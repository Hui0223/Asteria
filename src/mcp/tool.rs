use crate::tools::{AgentTool, ToolExecutionContext, ToolOutput};
use rmcp::{
    Peer, RoleClient,
    model::{CallToolRequestParams, CallToolResult, ContentBlock, Tool},
};
use serde_json::{Value, json};
use std::time::Duration;

/// 将远端 MCP Tool 适配为 Asteria 现有 AgentTool。
pub struct McpToolAdapter {
    exposed_name: String,
    server_name: String,
    remote_name: String,
    description: String,
    input_schema: Value,
    peer: Peer<RoleClient>,
    timeout: Duration,
}

impl McpToolAdapter {
    pub fn new(server_name: &str, tool: Tool, peer: Peer<RoleClient>, timeout: Duration) -> Self {
        let remote_name = tool.name.into_owned();
        Self {
            exposed_name: namespaced_tool_name(server_name, &remote_name),
            server_name: server_name.to_owned(),
            remote_name,
            description: tool
                .description
                .map(|description| description.into_owned())
                .unwrap_or_else(|| "No description provided.".into()),
            input_schema: Value::Object(tool.input_schema.as_ref().clone()),
            peer,
            timeout,
        }
    }

    async fn call(
        &self,
        raw_args: &str,
        cancel: Option<&crate::agent_loop::CancelToken>,
    ) -> ToolOutput {
        let arguments = match serde_json::from_str::<Value>(raw_args) {
            Ok(Value::Object(arguments)) => arguments,
            Ok(_) => {
                return ToolOutput {
                    content: "MCP 工具执行失败: 参数必须是 JSON Object".into(),
                    is_error: true,
                };
            }
            Err(error) => {
                return ToolOutput {
                    content: format!("MCP 工具执行失败: 无法解析参数: {error}"),
                    is_error: true,
                };
            }
        };
        let request =
            CallToolRequestParams::new(self.remote_name.clone()).with_arguments(arguments);
        let response = self.peer.call_tool(request);
        let result = match cancel {
            Some(cancel) => {
                tokio::select! {
                    result = response => result,
                    _ = cancel.cancelled() => {
                        return ToolOutput {
                            content: format!("MCP 工具调用已取消: {}", self.exposed_name),
                            is_error: true,
                        };
                    }
                }
            }
            None => response.await,
        };
        match result {
            Ok(result) => normalize_result(result),
            Err(error) => ToolOutput {
                content: format!(
                    "MCP 工具调用失败（Server={}，Tool={}）: {error}",
                    self.server_name, self.remote_name
                ),
                is_error: true,
            },
        }
    }
}

#[async_trait::async_trait]
impl AgentTool for McpToolAdapter {
    fn name(&self) -> &str {
        &self.exposed_name
    }

    fn schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.exposed_name,
                "description": format!(
                    "MCP tool from server `{}`.\n\n{}",
                    self.server_name, self.description
                ),
                "parameters": self.input_schema
            }
        })
    }

    fn execution_timeout(&self) -> Option<Duration> {
        Some(self.timeout)
    }

    async fn execute(&self, raw_args: &str) -> ToolOutput {
        self.call(raw_args, None).await
    }

    async fn execute_with_context(
        &self,
        raw_args: &str,
        context: ToolExecutionContext,
    ) -> ToolOutput {
        self.call(raw_args, Some(&context.cancel)).await
    }
}

pub fn namespaced_tool_name(server_name: &str, tool_name: &str) -> String {
    let name = format!(
        "mcp__{}__{}",
        sanitize_name(server_name),
        sanitize_name(tool_name)
    );
    if name.len() <= 64 {
        return name;
    }
    let hash = stable_hash(&format!("{server_name}\0{tool_name}"));
    format!("{}_{}", &name[..55], hash)
}

fn sanitize_name(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "unnamed".into()
    } else {
        sanitized
    }
}

fn stable_hash(value: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:08x}").chars().take(8).collect()
}

fn normalize_result(result: CallToolResult) -> ToolOutput {
    let mut sections: Vec<String> = result.content.iter().map(content_to_text).collect();
    sections.retain(|section| !section.is_empty());
    if sections.is_empty()
        && let Some(structured) = result.structured_content
    {
        sections.push(structured.to_string());
    }
    if sections.is_empty() {
        sections.push("(MCP tool returned no textual content)".into());
    }
    ToolOutput {
        content: sections.join("\n\n"),
        is_error: result.is_error.unwrap_or(false),
    }
}

fn content_to_text(content: &ContentBlock) -> String {
    if let Some(text) = content.as_text() {
        text.text.clone()
    } else if let Some(resource) = content.as_resource() {
        let text = resource.get_text();
        if text.is_empty() {
            "[MCP binary resource omitted]".into()
        } else {
            text
        }
    } else if let Some(link) = content.as_resource_link() {
        format!("[MCP resource: {} ({})]", link.name, link.uri)
    } else if let Some(image) = content.as_image() {
        format!("[MCP image omitted: {}]", image.mime_type)
    } else if let Some(audio) = content.as_audio() {
        format!("[MCP audio omitted: {}]", audio.mime_type)
    } else {
        "[Unsupported MCP content omitted]".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespaces_and_sanitizes_remote_tool_names() {
        assert_eq!(
            namespaced_tool_name("github server", "list/issues"),
            "mcp__github_server__list_issues"
        );
        let long = namespaced_tool_name(&"server".repeat(12), &"tool".repeat(12));
        assert_eq!(long.len(), 64);
        assert!(long.starts_with("mcp__"));
    }

    #[test]
    fn normalizes_text_without_embedding_binary_payloads() {
        let output = normalize_result(CallToolResult::success(vec![
            ContentBlock::text("hello"),
            ContentBlock::image("very-large-base64", "image/png"),
        ]));
        assert!(output.content.contains("hello"));
        assert!(output.content.contains("image/png"));
        assert!(!output.content.contains("very-large-base64"));
        assert!(!output.is_error);
    }
}
