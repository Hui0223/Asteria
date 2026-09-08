use crate::{context::ContextMemory, message::ToolCall};
use anyhow::Result;
use serde_json::Value;

pub mod deepseek;

/// 模型完成一个 Step 后返回的供应商无关结果。
#[derive(Clone, Debug)]
pub struct AssistantTurn {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

/// 定义 Agent Loop 所依赖的最小模型能力，便于替换供应商和编写假模型测试。
pub trait ModelProvider {
    /// 返回当前供应商实际使用的模型名称。
    fn model(&self) -> &str;

    /// 根据完整上下文和工具定义生成下一个助手消息。
    fn complete(&self, context: &ContextMemory, tools: Value) -> Result<AssistantTurn>;
}
