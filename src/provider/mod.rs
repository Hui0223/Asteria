use crate::{context_builder::PreparedContext, message::ToolCall};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

pub mod deepseek;

/// 模型流式正文回调；实现方必须快速返回，不得阻塞 HTTP 读取。
pub type TextDeltaSink = Arc<dyn Fn(&str) + Send + Sync>;

/// 记录模型服务端返回的本次请求 Token 消耗。
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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

    /// 异步生成助手消息；取消时会丢弃该 Future，实现方不得在后台继续写上下文。
    fn complete(
        &self,
        context: &PreparedContext,
        tools: Value,
    ) -> impl std::future::Future<Output = Result<AssistantTurn>>;

    /// 在生成过程中推送正文增量；默认实现在完成后把整段答案作为一次 delta。
    fn complete_streaming(
        &self,
        context: &PreparedContext,
        tools: Value,
        on_delta: TextDeltaSink,
    ) -> impl std::future::Future<Output = Result<AssistantTurn>> {
        async move {
            let result = self.complete(context, tools).await?;
            if let Some(content) = result.content.as_deref()
                && !content.is_empty()
            {
                on_delta(content);
            }
            Ok(result)
        }
    }
}
