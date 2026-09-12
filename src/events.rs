use crate::provider::TokenUsage;

/// Asteria 执行过程中的稳定事件类型，供 TUI、Transcript、日志和评估使用。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentEvent {
    TurnStarted {
        turn_id: u64,
        input: String,
    },
    StepStarted {
        turn_id: u64,
        step: usize,
    },
    ToolCallStarted {
        turn_id: u64,
        call_id: String,
        name: String,
    },
    ToolResult {
        turn_id: u64,
        call_id: String,
        is_error: bool,
    },
    StepCompleted {
        turn_id: u64,
        step: usize,
        usage: TokenUsage,
    },
    /// 一次模型请求失败但准备重试；不会增加 Step 数。
    StepRetrying {
        turn_id: u64,
        step: usize,
        failed_attempt: usize,
        next_attempt: usize,
        max_attempts: usize,
        delay_ms: u128,
    },
    TurnCompleted {
        turn_id: u64,
        steps: usize,
    },
    TurnCancelled {
        turn_id: u64,
    },
    TurnFailed {
        turn_id: u64,
        message: String,
    },
}

/// 接收 Agent 事件的最小接口；实现方可以写日志、更新 TUI 或保存 Transcript。
pub trait EventSink: Send + Sync {
    /// 消费一个事件；实现方不应阻塞模型 Loop。
    fn publish(&self, event: AgentEvent);
}

/// 默认空事件接收器，保证没有 UI 时 Loop 无额外行为。
#[derive(Default)]
pub struct NoopEventSink;

impl EventSink for NoopEventSink {
    /// 丢弃事件。
    fn publish(&self, _: AgentEvent) {}
}
