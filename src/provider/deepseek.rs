use crate::{
    context::ContextMemory,
    message::{Message, ToolCall},
};
use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use std::env;

const API_URL: &str = "https://api.deepseek.com/chat/completions";

/// 表示模型完成一个推理步骤后返回的统一助手消息。
#[derive(Clone, Debug)]
pub struct AssistantTurn {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

/// 封装 DeepSeek 的鉴权、请求协议和响应解析。
pub struct DeepSeekProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl DeepSeekProvider {
    /// 从环境变量创建 DeepSeek 适配器，并使用默认模型作为后备值。
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            api_key: env::var("DEEPSEEK_API_KEY").context("请先在 .env 中设置 DEEPSEEK_API_KEY")?,
            model: env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".into()),
        })
    }

    /// 返回当前适配器实际使用的模型名称。
    pub fn model(&self) -> &str {
        &self.model
    }

    /// 把中立上下文转换成 DeepSeek 请求，并返回统一格式的助手消息。
    pub fn complete(&self, context: &ContextMemory, tools: Value) -> Result<AssistantTurn> {
        let response: ChatResponse = self
            .client
            .post(API_URL)
            .bearer_auth(&self.api_key)
            .json(&json!({
                "model": self.model,
                "messages": project(context)?,
                "tools": tools,
                "tool_choice": "auto",
                "thinking": {"type": "disabled"}
            }))
            .send()
            .context("无法连接 DeepSeek API")?
            .error_for_status()
            .context("DeepSeek API 返回错误")?
            .json()
            .context("无法解析 DeepSeek 响应")?;

        let message = response
            .choices
            .into_iter()
            .next()
            .context("DeepSeek 响应中没有 message")?
            .message;
        Ok(AssistantTurn {
            content: message.content,
            tool_calls: message
                .tool_calls
                .into_iter()
                .map(|call| ToolCall {
                    id: call.id,
                    name: call.function.name,
                    arguments: call.function.arguments,
                })
                .collect(),
        })
    }
}

/// 把内部强类型消息投影为 DeepSeek Chat Completions 的 JSON 消息。
fn project(context: &ContextMemory) -> Result<Vec<Value>> {
    context.validate()?;
    let mut wire = vec![json!({"role": "system", "content": context.system_prompt()})];
    for message in context.messages() {
        wire.push(match message {
            Message::User { content } => json!({"role": "user", "content": content}),
            Message::Assistant {
                content,
                tool_calls,
            } if tool_calls.is_empty() => json!({"role": "assistant", "content": content}),
            Message::Assistant {
                content,
                tool_calls,
            } => json!({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls.iter().map(|call| json!({
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments}
                })).collect::<Vec<_>>()
            }),
            Message::Tool {
                call_id, content, ..
            } => json!({"role": "tool", "tool_call_id": call_id, "content": content}),
        });
    }
    Ok(wire)
}

/// DeepSeek 顶层响应中当前任务需要读取的字段。
#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

/// DeepSeek 候选答案中的消息包装层。
#[derive(Deserialize)]
struct Choice {
    message: AssistantResponse,
}

/// DeepSeek 返回的助手正文和工具调用列表。
#[derive(Deserialize)]
struct AssistantResponse {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ResponseToolCall>,
}

/// DeepSeek 工具调用的 ID 与函数描述。
#[derive(Deserialize)]
struct ResponseToolCall {
    id: String,
    function: ResponseFunction,
}

/// DeepSeek 函数调用中的名称和原始 JSON 参数。
#[derive(Deserialize)]
struct ResponseFunction {
    name: String,
    arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造投影测试所需的工具调用。
    fn call(id: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "calculate".into(),
            arguments: "{\"expression\":\"1+1\"}".into(),
        }
    }

    #[test]
    /// 验证普通助手消息不会携带空的 tool_calls 字段。
    fn projection_omits_empty_tool_calls() {
        let mut context = ContextMemory::new("system");
        context
            .append_assistant(Some("done".into()), Vec::new())
            .unwrap();
        let projected = project(&context).unwrap();
        assert!(projected[1].get("tool_calls").is_none());
    }

    #[test]
    /// 验证工具调用和结果会被转换成 DeepSeek 要求的配对格式。
    fn projection_preserves_tool_exchange() {
        let mut context = ContextMemory::new("system");
        context.append_assistant(None, vec![call("a")]).unwrap();
        context.append_tool_result("a", "2", false).unwrap();
        let projected = project(&context).unwrap();
        assert_eq!(projected[1]["tool_calls"][0]["id"], "a");
        assert_eq!(projected[2]["tool_call_id"], "a");
    }
}
