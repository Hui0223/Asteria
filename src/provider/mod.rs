use crate::{context_builder::PreparedContext, message::ToolCall};
use anyhow::Result;
use serde_json::Value;

pub mod deepseek;

/// 记录模型服务端返回的本次请求 Token 消耗。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// 模型完成一个 Step 后返回的供应商无关结果。
#[derive(Clone, Debug)]
pub struct AssistantTurn {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<TokenUsage>,
}

/// 定义 Agent Loop 所依赖的最小模型能力，便于替换供应商和编写假模型测试。
pub trait ModelProvider {
    /// 返回当前供应商实际使用的模型名称。
    fn model(&self) -> &str;

    /// 根据预算化上下文和工具定义生成下一个助手消息。
    fn complete(&self, context: &PreparedContext, tools: Value) -> Result<AssistantTurn>;
}
