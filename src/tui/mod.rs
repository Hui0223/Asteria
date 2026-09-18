mod approval;
mod reducer;
pub mod render;
mod sink;
mod state;

use crate::terminal::Output;
use std::sync::{Arc, Mutex};

pub use approval::ApprovalUi;
pub use sink::TuiEventSink;
pub type SharedTuiState = Arc<Mutex<state::TuiState>>;

pub fn new_state() -> SharedTuiState {
    Arc::new(Mutex::new(state::TuiState::default()))
}

pub fn set_verbose(state: &SharedTuiState, enabled: bool, output: &Output) {
    if let Ok(mut state) = state.lock() {
        state.verbose = enabled;
    }
    output.print(format!(
        "[界面] 详细事件输出已{}。",
        if enabled { "开启" } else { "关闭" }
    ));
}

pub fn is_verbose(state: &SharedTuiState) -> bool {
    state.lock().is_ok_and(|state| state.verbose)
}

pub fn take_turn_elapsed(state: &SharedTuiState, turn_id: u64) -> Option<std::time::Duration> {
    state
        .lock()
        .ok()
        .and_then(|mut state| state.take_elapsed(turn_id))
}

pub fn take_answer_streamed(state: &SharedTuiState) -> bool {
    state
        .lock()
        .ok()
        .is_some_and(|mut state| state.take_answer_streamed())
}
