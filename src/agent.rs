use crate::{
    agent_loop::{AgentLoop, CancelToken, LoopConfig},
    context::ContextMemory,
    provider::TokenUsage,
    provider::deepseek::DeepSeekProvider,
};
use anyhow::Result;

const SYSTEM: &str = "你是 Asteria，一个可靠、简洁的中文 AI 助手。需要精确计算或当前时间时调用工具，不要猜测工具结果。";

/// 对外提供简单问答接口，并组合上下文与 Agent Loop。
pub struct Asteria {
    agent_loop: AgentLoop<DeepSeekProvider>,
    context: ContextMemory,
}

impl Asteria {
    /// 根据环境变量创建 DeepSeek Agent，并采用默认 Loop 配置。
    pub fn new() -> Result<Self> {
        Ok(Self {
            agent_loop: AgentLoop::new(DeepSeekProvider::from_env()?, LoopConfig::default()),
            context: ContextMemory::new(SYSTEM),
        })
    }

    /// 返回当前 Agent 使用的模型名称。
    pub fn model(&self) -> &str {
        self.agent_loop.model()
    }

    /// 返回最近一个 Turn 的真实 Token 使用量；尚未请求模型时返回 None。
    pub fn last_usage(&self) -> Option<&TokenUsage> {
        self.agent_loop.last_turn().map(|turn| &turn.usage)
    }

    /// 返回当前进程内所有 Turn 的累计 Token 使用量。
    pub fn session_usage(&self) -> &TokenUsage {
        self.agent_loop.session_usage()
    }

    /// 清空对话历史，但保留 Agent 的系统设定。
    pub fn reset(&mut self) {
        self.context.reset();
    }

    /// 使用一个新的取消令牌执行完整用户 Turn。
    pub fn ask(&mut self, input: &str) -> Result<String> {
        self.ask_with_cancel(input, &CancelToken::default())
    }

    /// 使用调用方提供的令牌执行 Turn，以支持协作式取消。
    pub fn ask_with_cancel(&mut self, input: &str, cancel: &CancelToken) -> Result<String> {
        self.agent_loop.run_turn(&mut self.context, input, cancel)
    }
}
