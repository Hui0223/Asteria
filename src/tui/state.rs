use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

/// Reedline 按整行打印，达到该宽度或换行时才刷新，避免每个 token 占一行。
const LIVE_FLUSH_CHARS: usize = 80;

/// TUI 的瞬时展示状态；不写入 Session，也不进入模型上下文。
#[derive(Default)]
pub struct TuiState {
    pub verbose: bool,
    pub tool_names: HashMap<String, String>,
    pub live: LiveAnswer,
    turn_started_at: HashMap<u64, Instant>,
    completed_elapsed: HashMap<u64, Duration>,
}

impl TuiState {
    pub fn begin_turn(&mut self, turn_id: u64) {
        self.live.reset();
        self.turn_started_at.insert(turn_id, Instant::now());
    }

    pub fn complete_turn(&mut self, turn_id: u64) {
        if let Some(started) = self.turn_started_at.remove(&turn_id) {
            self.completed_elapsed.insert(turn_id, started.elapsed());
        }
        self.tool_names.clear();
    }

    pub fn take_elapsed(&mut self, turn_id: u64) -> Option<Duration> {
        self.completed_elapsed.remove(&turn_id)
    }

    pub fn take_answer_streamed(&mut self) -> bool {
        let streamed = self.live.streamed;
        self.live.streamed = false;
        streamed
    }

    pub fn clear_turn(&mut self, turn_id: u64) {
        self.turn_started_at.remove(&turn_id);
        self.tool_names.clear();
    }
}

/// 合并 assistant.delta，按行刷新；对应 Kimi 的 LiveView draft。
#[derive(Default)]
pub struct LiveAnswer {
    pub streamed: bool,
    header_shown: bool,
    buffer: String,
    next_offset: usize,
}

impl LiveAnswer {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// 工具调用或重试后，下一段生成重新显示 Asteria 标题。
    pub fn begin_generation(&mut self) {
        self.header_shown = false;
        self.buffer.clear();
        self.next_offset = 0;
    }

    pub fn append(&mut self, delta: &str, offset: usize) -> Vec<String> {
        if delta.is_empty() || offset < self.next_offset {
            return Vec::new();
        }
        self.next_offset = offset.saturating_add(delta.len());
        self.buffer.push_str(delta);
        self.flush(false)
    }

    pub fn flush(&mut self, force: bool) -> Vec<String> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let mut content = Vec::new();
        while let Some(idx) = self.buffer.find('\n') {
            let mut line: String = self.buffer.drain(..=idx).collect();
            if line.ends_with('\n') {
                line.pop();
            }
            if line.ends_with('\r') {
                line.pop();
            }
            content.push(line);
        }
        if (force || self.buffer.chars().count() >= LIVE_FLUSH_CHARS) && !self.buffer.is_empty() {
            content.push(std::mem::take(&mut self.buffer));
        }
        if content.is_empty() {
            return Vec::new();
        }
        let mut lines = Vec::new();
        if !self.header_shown {
            lines.push("\nAsteria".into());
            self.header_shown = true;
        }
        self.streamed = true;
        lines.extend(content);
        lines
    }
}
