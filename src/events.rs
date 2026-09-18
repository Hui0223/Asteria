use crate::{message::ToolCall, provider::TokenUsage};
use serde_json::Value;
use std::sync::{
    Arc, Mutex,
    mpsc::{self, SyncSender},
};
use std::thread::JoinHandle;

/// 不参与模型上下文的工具审计记录。
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolTrace {
    pub turn_id: u64,
    pub step: usize,
    pub call_id: String,
    pub tool_name: String,
    pub arguments_preview: String,
    pub arguments_hash: String,
    pub status: ToolTraceStatus,
    pub duration_ms: u64,
    pub result_hash: String,
    pub completed_at: String,
}

/// 工具执行结果状态；权限拒绝、参数错误和超时均记为 Error。
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ToolTraceStatus {
    Success,
    Error,
}

impl ToolTrace {
    /// 从一次已完成的调用创建脱敏审计记录，不保存完整工具结果。
    pub fn completed(
        turn_id: u64,
        step: usize,
        call: &ToolCall,
        result: &str,
        is_error: bool,
        duration_ms: u128,
    ) -> Self {
        Self {
            turn_id,
            step,
            call_id: call.id.clone(),
            tool_name: call.name.clone(),
            arguments_preview: arguments_preview(&call.arguments),
            arguments_hash: stable_hash(&call.arguments),
            status: if is_error {
                ToolTraceStatus::Error
            } else {
                ToolTraceStatus::Success
            },
            duration_ms: u64::try_from(duration_ms).unwrap_or(u64::MAX),
            result_hash: stable_hash(result),
            completed_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Asteria 执行过程中的稳定事件类型，供 TUI、Transcript、日志和评估使用。
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
        duration_ms: u128,
    },
    /// 工具权限为 ask，正在等待用户批准。
    PermissionRequested {
        turn_id: u64,
        call_id: String,
        name: String,
    },
    /// 用户对本次工具调用作出批准或拒绝。
    PermissionResolved {
        turn_id: u64,
        call_id: String,
        allowed: bool,
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
    /// 模型正在生成的增量正文；易失事件，不写入 Session。
    AssistantDelta {
        turn_id: u64,
        step: usize,
        delta: String,
        offset: usize,
    },
}

impl AgentEvent {
    /// 流式分片只服务 TUI，不进入 trace.jsonl。
    pub fn is_volatile(&self) -> bool {
        matches!(self, Self::AssistantDelta { .. })
    }
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

/// 将同一事件同时发送给持久化、TUI 或评估等多个接收器。
pub struct FanoutEventSink {
    pub sinks: Vec<std::sync::Arc<dyn EventSink>>,
}

impl EventSink for FanoutEventSink {
    /// 顺序通知所有接收器。
    fn publish(&self, event: AgentEvent) {
        for sink in &self.sinks {
            sink.publish(event.clone());
        }
    }
}

/// 有界异步事件队列；生产者只入队，磁盘和 TUI 输出由后台消费者处理。
pub struct QueuedEventSink {
    sender: Mutex<Option<SyncSender<AgentEvent>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl QueuedEventSink {
    /// 创建容量受限的事件队列，并启动一个后台消费线程。
    pub fn new(sinks: Vec<Arc<dyn EventSink>>) -> Arc<Self> {
        let (sender, receiver) = mpsc::sync_channel::<AgentEvent>(1024);
        let worker = std::thread::Builder::new()
            .name("asteria-events".into())
            .spawn(move || {
                while let Ok(event) = receiver.recv() {
                    for sink in &sinks {
                        sink.publish(event.clone());
                    }
                }
            })
            .expect("无法启动事件消费线程");
        Arc::new(Self {
            sender: Mutex::new(Some(sender)),
            worker: Mutex::new(Some(worker)),
        })
    }
}

impl EventSink for QueuedEventSink {
    /// 非阻塞发布事件；队列满时丢弃事件，避免拖住 Agent 主循环。
    fn publish(&self, event: AgentEvent) {
        if event.is_volatile() {
            return;
        }
        let sender = self
            .sender
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().cloned());
        if let Some(sender) = sender {
            let _ = sender.try_send(event);
        }
    }
}

impl Drop for QueuedEventSink {
    /// 关闭发送端并等待队列中的事件消费完成。
    fn drop(&mut self) {
        self.sender.lock().ok().and_then(|mut sender| sender.take());
        if let Ok(mut worker) = self.worker.lock()
            && let Some(worker) = worker.take()
        {
            let _ = worker.join();
        }
    }
}

/// 只展示经过键名脱敏的 JSON 参数；无效 JSON 仅保留格式说明和哈希。
fn arguments_preview(raw: &str) -> String {
    let Ok(mut value) = serde_json::from_str::<Value>(raw) else {
        return "<invalid-json; hash-only>".into();
    };
    redact_sensitive_values(&mut value);
    let preview = serde_json::to_string(&value).unwrap_or_else(|_| "<unavailable>".into());
    truncate_chars(&preview, 240)
}

fn redact_sensitive_values(value: &mut Value) {
    match value {
        Value::Object(fields) => {
            for (key, value) in fields {
                if is_sensitive_key(key) {
                    *value = Value::String("[REDACTED]".into());
                } else {
                    redact_sensitive_values(value);
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                redact_sensitive_values(value);
            }
        }
        _ => {}
    }
}

fn is_sensitive_key(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase().replace(['_', '-'], "");
    normalized == "key"
        || matches!(normalized.as_str(), "content" | "oldstring" | "newstring")
        || [
            "password",
            "passwd",
            "secret",
            "token",
            "apikey",
            "authorization",
            "credential",
            "privatekey",
            "accesskey",
        ]
        .iter()
        .any(|marker| normalized.contains(marker))
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let mut result: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    result.push('…');
    result
}

fn stable_hash(text: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Trace 参数预览保留普通字段，但必须递归隐藏凭据。
    fn tool_trace_redacts_sensitive_arguments() {
        let trace = ToolTrace::completed(
            1,
            2,
            &ToolCall {
                id: "call-1".into(),
                name: "example".into(),
                arguments:
                    r#"{"query":"chapter 14","api_key":"secret","nested":{"password":"pw"}}"#.into(),
            },
            "result",
            false,
            12,
        );
        assert!(trace.arguments_preview.contains("chapter 14"));
        assert!(!trace.arguments_preview.contains("secret"));
        assert!(!trace.arguments_preview.contains("\"pw\""));
        assert_eq!(trace.status, ToolTraceStatus::Success);
        assert_eq!(trace.arguments_hash.len(), 16);
        assert_eq!(trace.result_hash.len(), 16);
    }

    #[test]
    /// 非 JSON 参数不能原样进入审计日志。
    fn invalid_json_trace_keeps_hash_only() {
        let trace = ToolTrace::completed(
            1,
            1,
            &ToolCall {
                id: "call".into(),
                name: "tool".into(),
                arguments: "secret raw text".into(),
            },
            "result",
            true,
            0,
        );
        assert_eq!(trace.arguments_preview, "<invalid-json; hash-only>");
        assert_eq!(trace.status, ToolTraceStatus::Error);
    }

    #[test]
    fn tool_trace_redacts_file_write_bodies() {
        let trace = ToolTrace::completed(
            1,
            1,
            &ToolCall {
                id: "write".into(),
                name: "Edit".into(),
                arguments:
                    r#"{"path":"src/lib.rs","old_string":"private old","new_string":"private new"}"#
                        .into(),
            },
            "ok",
            false,
            1,
        );
        assert!(trace.arguments_preview.contains("src/lib.rs"));
        assert!(!trace.arguments_preview.contains("private"));
    }

    #[test]
    fn assistant_delta_is_volatile() {
        assert!(
            AgentEvent::AssistantDelta {
                turn_id: 1,
                step: 1,
                delta: "hi".into(),
                offset: 0,
            }
            .is_volatile()
        );
        assert!(
            !AgentEvent::TurnCompleted {
                turn_id: 1,
                steps: 1
            }
            .is_volatile()
        );
    }
}
