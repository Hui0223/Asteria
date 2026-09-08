use crate::{context::ContextMemory, message::ToolCall, tools};
use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::json;
use std::env;

const API_URL: &str = "https://api.deepseek.com/chat/completions";
const SYSTEM: &str = "你是 Asteria，一个可靠、简洁的中文 AI 助手。需要精确计算或当前时间时调用工具，不要猜测工具结果。";

pub struct Asteria {
    client: Client,
    api_key: String,
    model: String,
    context: ContextMemory,
}

impl Asteria {
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            api_key: env::var("DEEPSEEK_API_KEY").context("请先在 .env 中设置 DEEPSEEK_API_KEY")?,
            model: env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".into()),
            context: ContextMemory::new(SYSTEM),
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn reset(&mut self) {
        self.context.reset();
    }

    pub fn ask(&mut self, input: &str) -> Result<String> {
        let checkpoint = self.context.checkpoint();
        self.context.append_user(input)?;
        let result = self.complete_turn();
        if result.is_err() {
            self.context.rollback(checkpoint);
        }
        result
    }

    fn complete_turn(&mut self) -> Result<String> {
        for _ in 0..8 {
            let response: ChatResponse = self
                .client
                .post(API_URL)
                .bearer_auth(&self.api_key)
                .json(&json!({
                    "model":self.model,
                    "messages":self.context.project_deepseek()?,
                    "tools":tools::schema(), "tool_choice":"auto",
                    "thinking":{"type":"disabled"}
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
            let calls = message
                .tool_calls
                .into_iter()
                .map(|call| ToolCall {
                    id: call.id,
                    name: call.function.name,
                    arguments: call.function.arguments,
                })
                .collect::<Vec<_>>();
            self.context
                .append_assistant(message.content.clone(), calls.clone())?;
            if calls.is_empty() {
                return Ok(message.content.unwrap_or_default());
            }
            for call in calls {
                let output = tools::execute(&call.name, &call.arguments);
                self.context
                    .append_tool_result(call.id, output.content, output.is_error)?;
            }
        }
        bail!("工具调用轮次过多，已停止")
    }
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: AssistantResponse,
}

#[derive(Deserialize)]
struct AssistantResponse {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ResponseToolCall>,
}

#[derive(Deserialize)]
struct ResponseToolCall {
    id: String,
    function: ResponseFunction,
}

#[derive(Deserialize)]
struct ResponseFunction {
    name: String,
    arguments: String,
}
