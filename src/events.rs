use crate::provider::TokenUsage;
use std::sync::{
    Arc, Mutex,
    mpsc::{self, SyncSender},
};
use std::thread::JoinHandle;

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
