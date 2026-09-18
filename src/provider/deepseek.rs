use crate::{
    context_builder::PreparedContext,
    message::{Message, ToolCall},
    provider::{AssistantTurn, ModelProvider, TextDeltaSink, TokenUsage},
};
use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::{Value, json};
use std::{collections::BTreeMap, env};

const API_URL: &str = "https://api.deepseek.com/chat/completions";

/// 封装 DeepSeek 的鉴权、请求协议和响应解析。
pub struct DeepSeekProvider {
    client: Client,
    api_key: String,
    model: String,
    api_url: String,
}

impl DeepSeekProvider {
    /// 从环境变量创建 DeepSeek 适配器，并使用默认模型作为后备值。
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            client: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(10))
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .context("无法创建 HTTP 客户端")?,
            api_key: env::var("DEEPSEEK_API_KEY").context("请先在 .env 中设置 DEEPSEEK_API_KEY")?,
            model: env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".into()),
            api_url: API_URL.into(),
        })
    }

    /// 执行一次 DeepSeek SSE 请求，边读边推送正文增量。
    async fn request(
        &self,
        context: &PreparedContext,
        tools: Value,
        on_delta: Option<&TextDeltaSink>,
    ) -> Result<AssistantTurn> {
        let (tools, tool_choice) = select_tools(context, tools);
        let mut response = self
            .client
            .post(&self.api_url)
            .bearer_auth(&self.api_key)
            .json(&json!({
                "model": self.model,
                "messages": project(context),
                "tools": tools,
                "tool_choice": tool_choice,
                "thinking": {"type": "disabled"},
                "stream": true,
                "stream_options": {"include_usage": true}
            }))
            .send()
            .await
            .context("无法连接 DeepSeek API")?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("DeepSeek API 返回错误: {status} {body}");
        }

        let mut assembler = StreamAssembler::default();
        let mut leftover = String::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .context("读取 DeepSeek 流式响应失败")?
        {
            leftover.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(idx) = leftover.find('\n') {
                let mut line: String = leftover.drain(..=idx).collect();
                if line.ends_with('\n') {
                    line.pop();
                }
                if line.ends_with('\r') {
                    line.pop();
                }
                if assembler.ingest_line(&line, on_delta)? {
                    return Ok(assembler.finish());
                }
            }
        }
        if !leftover.trim().is_empty() {
            let _ = assembler.ingest_line(leftover.trim(), on_delta)?;
        }
        Ok(assembler.finish())
    }
}

fn select_tools(context: &PreparedContext, tools: Value) -> (Value, Value) {
    if let Some(forced_name) = context.forced_tool_name() {
        let selected = tools
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter(|item| item["function"]["name"] == forced_name)
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        (Value::Array(selected), json!("required"))
    } else if context.force_calculation() {
        let calculate_only = tools
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter(|item| item["function"]["name"] == "calculate")
                    .cloned()
                    .collect()
            })
            .map(Value::Array)
            .unwrap_or(tools);
        (calculate_only, json!("required"))
    } else if context.force_search_docs() {
        let search_only = tools
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter(|item| item["function"]["name"] == "search_docs")
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if search_only.is_empty() {
            (tools, json!("auto"))
        } else {
            (Value::Array(search_only), json!("required"))
        }
    } else {
        (tools, json!("auto"))
    }
}

impl ModelProvider for DeepSeekProvider {
    /// 向 Agent Loop 暴露当前 DeepSeek 模型名称。
    fn model(&self) -> &str {
        &self.model
    }

    /// 把中立上下文转换成 DeepSeek 协议并完成一个模型 Step。
    async fn complete(&self, context: &PreparedContext, tools: Value) -> Result<AssistantTurn> {
        self.request(context, tools, None).await
    }

    /// 通过 Chat Completions SSE 推送 assistant.delta。
    async fn complete_streaming(
        &self,
        context: &PreparedContext,
        tools: Value,
        on_delta: TextDeltaSink,
    ) -> Result<AssistantTurn> {
        self.request(context, tools, Some(&on_delta)).await
    }
}

/// 把内部强类型消息投影为 DeepSeek Chat Completions 的 JSON 消息。
fn project(context: &PreparedContext) -> Vec<Value> {
    let mut wire = vec![json!({"role": "system", "content": context.system_prompt()})];
    if let Some(summary) = context.summary() {
        wire.push(json!({
            "role": "user",
            "content": format!("[Asteria 历史摘要]\n{summary}")
        }));
    }
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
    wire
}

/// 把 Chat Completions SSE 分片拼成完整 AssistantTurn。
#[derive(Default)]
struct StreamAssembler {
    content: String,
    tools: BTreeMap<usize, PartialToolCall>,
    usage: Option<TokenUsage>,
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl StreamAssembler {
    /// 解析一行 SSE；遇到 `[DONE]` 时返回 true。
    fn ingest_line(&mut self, line: &str, on_delta: Option<&TextDeltaSink>) -> Result<bool> {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            return Ok(false);
        }
        let Some(data) = line.strip_prefix("data:") else {
            return Ok(false);
        };
        let data = data.trim();
        if data == "[DONE]" {
            return Ok(true);
        }
        self.ingest_json(data, on_delta)?;
        Ok(false)
    }

    fn ingest_json(&mut self, data: &str, on_delta: Option<&TextDeltaSink>) -> Result<()> {
        let value: Value = serde_json::from_str(data).context("无法解析 DeepSeek 流式分片")?;
        if let Some(usage) = value.get("usage").filter(|item| !item.is_null()) {
            self.usage = Some(TokenUsage {
                prompt_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0) as usize,
                completion_tokens: usage["completion_tokens"].as_u64().unwrap_or(0) as usize,
                total_tokens: usage["total_tokens"].as_u64().unwrap_or(0) as usize,
            });
        }
        let Some(choice) = value.get("choices").and_then(|choices| choices.get(0)) else {
            return Ok(());
        };
        let delta = choice
            .get("delta")
            .or_else(|| choice.get("message"))
            .unwrap_or(&Value::Null);
        if let Some(text) = delta.get("content").and_then(Value::as_str)
            && !text.is_empty()
        {
            self.content.push_str(text);
            if let Some(sink) = on_delta {
                sink(text);
            }
        }
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let entry = self.tools.entry(index).or_default();
                if let Some(id) = call.get("id").and_then(Value::as_str)
                    && !id.is_empty()
                {
                    entry.id = id.to_owned();
                }
                if let Some(name) = call.pointer("/function/name").and_then(Value::as_str) {
                    entry.name.push_str(name);
                }
                if let Some(arguments) = call.pointer("/function/arguments").and_then(Value::as_str)
                {
                    entry.arguments.push_str(arguments);
                }
            }
        }
        Ok(())
    }

    fn finish(self) -> AssistantTurn {
        AssistantTurn {
            content: if self.content.is_empty() {
                None
            } else {
                Some(self.content)
            },
            tool_calls: self
                .tools
                .into_values()
                .map(|call| ToolCall {
                    id: call.id,
                    name: call.name,
                    arguments: call.arguments,
                })
                .collect(),
            usage: self.usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::ContextMemory,
        context_builder::{ContextBuilder, ContextPolicy, HeuristicTokenEstimator},
    };

    /// 把测试记忆转换为不会裁剪消息的预算化上下文。
    fn prepare(context: &ContextMemory) -> PreparedContext {
        ContextBuilder::new(ContextPolicy::default(), HeuristicTokenEstimator)
            .prepare(context, &json!([]))
            .unwrap()
    }

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
        let projected = project(&prepare(&context));
        assert!(projected[1].get("tool_calls").is_none());
    }

    #[test]
    /// 验证工具调用和结果会被转换成 DeepSeek 要求的配对格式。
    fn projection_preserves_tool_exchange() {
        let mut context = ContextMemory::new("system");
        context.append_assistant(None, vec![call("a")]).unwrap();
        context.append_tool_result("a", "2", false).unwrap();
        let projected = project(&prepare(&context));
        assert_eq!(projected[1]["tool_calls"][0]["id"], "a");
        assert_eq!(projected[2]["tool_call_id"], "a");
    }

    #[test]
    /// 验证上下文摘要会在 DeepSeek JSON 中出现在历史消息之前。
    fn projection_includes_compaction_summary() {
        let mut context = ContextMemory::new("system");
        context.append_user("old question").unwrap();
        context
            .append_assistant(Some("old answer".into()), Vec::new())
            .unwrap();
        context.append_user("new question").unwrap();
        context
            .append_assistant(Some("new answer".into()), Vec::new())
            .unwrap();
        let prepared = ContextBuilder::new(
            ContextPolicy {
                max_context_tokens: 100,
                reserved_output_tokens: 10,
                compact_at_tokens: 20,
                recent_turns_to_keep: 1,
            },
            HeuristicTokenEstimator,
        )
        .prepare(&context, &json!([]))
        .unwrap();
        let projected = project(&prepared);
        assert!(prepared.summary().is_some());
        assert_eq!(projected[1]["role"], "user");
        assert!(
            projected[1]["content"]
                .as_str()
                .unwrap()
                .contains("历史摘要")
        );
    }

    #[test]
    fn explicit_tool_request_sends_only_named_tool_as_required() {
        let mut context = ContextMemory::new("system");
        context
            .append_user("调用 mcp__fixture__echo，参数 text 为 MCP-RECONNECT-002")
            .unwrap();
        let tools = json!([
            {"type":"function","function":{"name":"calculate","parameters":{"type":"object"}}},
            {"type":"function","function":{"name":"mcp__fixture__echo","parameters":{"type":"object"}}}
        ]);
        let prepared = ContextBuilder::new(ContextPolicy::default(), HeuristicTokenEstimator)
            .prepare(&context, &tools)
            .unwrap();
        let (selected, choice) = select_tools(&prepared, tools);
        assert_eq!(choice, "required");
        assert_eq!(selected.as_array().unwrap().len(), 1);
        assert_eq!(
            selected[0]["function"]["name"],
            Value::String("mcp__fixture__echo".into())
        );
    }

    /// 用真实 TCP 连接让服务停在响应头或正文读取阶段，再取消 Agent。
    async fn check_http_cancellation(partial_body: bool) {
        use crate::agent_loop::{AgentLoop, CancelToken, LoopConfig, TurnState};
        use std::time::Duration;
        use tokio::{
            io::{AsyncReadExt, AsyncWriteExt},
            net::TcpListener,
        };

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let api_url = format!("http://{}/chat/completions", listener.local_addr().unwrap());
        let (ready, started) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut data = Vec::new();
            let mut buffer = [0; 1024];
            loop {
                let count = stream.read(&mut buffer).await.unwrap();
                assert!(count > 0);
                data.extend_from_slice(&buffer[..count]);
                if data.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                    break;
                }
            }
            if partial_body {
                stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 10000\r\nConnection: close\r\n\r\n{\"choices\": [").await.unwrap();
            }
            ready.send(()).unwrap();
            // 保持连接打开，直到测试取消服务任务，客户端不能靠 EOF 提前返回。
            std::future::pending::<()>().await;
            drop(stream);
        });
        let provider = DeepSeekProvider {
            client: Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(120))
                .build()
                .unwrap(),
            api_key: "test-only".into(),
            model: "test".into(),
            api_url,
        };
        let mut engine = AgentLoop::new(provider, LoopConfig::default());
        let mut memory = ContextMemory::new("system");
        let cancel = CancelToken::new();
        let outcome = tokio::time::timeout(Duration::from_secs(3), async {
            tokio::join!(engine.run_turn(&mut memory, "hi", &cancel), async {
                started.await.unwrap();
                // 给 reqwest 机会进入正文读取，而不是只停留在连接阶段。
                tokio::time::sleep(Duration::from_millis(30)).await;
                cancel.cancel();
            })
            .0
        })
        .await;
        server.abort();
        let _ = server.await;
        assert!(outcome.expect("取消不能等到 120 秒请求超时").is_err());
        let report = engine.last_turn().unwrap();
        assert_eq!(report.state, TurnState::Cancelled);
        assert_eq!(report.retries, 0);
        assert_eq!(report.usage.total_tokens, 0);
        assert!(memory.messages().is_empty());
    }

    #[test]
    fn stream_assembler_merges_content_and_tool_calls() {
        let mut assembler = StreamAssembler::default();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let sink: TextDeltaSink = {
            let seen = seen.clone();
            std::sync::Arc::new(move |delta: &str| seen.lock().unwrap().push_str(delta))
        };
        assembler
            .ingest_line(
                r#"data: {"choices":[{"delta":{"content":"你好"}}]}"#,
                Some(&sink),
            )
            .unwrap();
        assembler
            .ingest_line(
                r#"data: {"choices":[{"delta":{"content":"世界"}}]}"#,
                Some(&sink),
            )
            .unwrap();
        assembler
            .ingest_line(
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"Read","arguments":""}}]}}]}"#,
                Some(&sink),
            )
            .unwrap();
        assembler
            .ingest_line(
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":\"a.rs\"}"}}]}}]}"#,
                Some(&sink),
            )
            .unwrap();
        assembler
            .ingest_line(
                r#"data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#,
                Some(&sink),
            )
            .unwrap();
        assert!(assembler.ingest_line("data: [DONE]", Some(&sink)).unwrap());
        let turn = assembler.finish();
        assert_eq!(turn.content.as_deref(), Some("你好世界"));
        assert_eq!(seen.lock().unwrap().as_str(), "你好世界");
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].id, "c1");
        assert_eq!(turn.tool_calls[0].name, "Read");
        assert_eq!(turn.tool_calls[0].arguments, r#"{"path":"a.rs"}"#);
        assert_eq!(
            turn.usage,
            Some(TokenUsage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5
            })
        );
    }

    #[tokio::test]
    /// 实际 HTTP 连接等待响应头时可取消。
    async fn cancels_http_before_headers() {
        check_http_cancellation(false).await;
    }

    #[tokio::test]
    /// 已收到响应头，但 JSON 正文尚未读完时也可取消。
    async fn cancels_http_during_body() {
        check_http_cancellation(true).await;
    }
}
