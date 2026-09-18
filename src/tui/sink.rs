use super::{SharedTuiState, reducer};
use crate::terminal::Output;
use asteria_agent::events::{AgentEvent, EventSink};

/// 将 Agent 事件同步归约后送入统一终端输出队列。
pub struct TuiEventSink {
    output: Output,
    state: SharedTuiState,
}

impl TuiEventSink {
    pub fn new(output: Output, state: SharedTuiState) -> Self {
        Self { output, state }
    }
}

impl EventSink for TuiEventSink {
    fn publish(&self, event: AgentEvent) {
        let messages = self
            .state
            .lock()
            .map(|mut state| reducer::reduce(&mut state, &event))
            .unwrap_or_default();
        for message in messages {
            self.output.print(message);
        }
    }
}
