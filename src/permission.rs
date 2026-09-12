use std::{
    fmt,
    str::FromStr,
    sync::atomic::{AtomicU64, Ordering},
};
use tokio::sync::{mpsc, oneshot};

/// 单个工具在当前会话中的权限；新注册工具默认需要人工确认。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ToolPermission {
    Allow,
    Deny,
    #[default]
    Ask,
}

impl fmt::Display for ToolPermission {
    /// 输出与终端命令一致的权限名称。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Allow => "allow",
            Self::Deny => "deny",
            Self::Ask => "ask",
        })
    }
}

impl FromStr for ToolPermission {
    type Err = anyhow::Error;
    /// 仅接受三种明确的权限值，拼写错误不能默认为允许。
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "allow" => Ok(Self::Allow),
            "deny" => Ok(Self::Deny),
            "ask" => Ok(Self::Ask),
            _ => anyhow::bail!("权限必须是 allow、deny 或 ask"),
        }
    }
}

/// 一次审批请求；reply 仅能消费一次，模型输出不能直接改变审批结果。
pub struct ApprovalRequest {
    pub id: u64,
    pub tool_name: String,
    pub arguments: String,
    pub call_id: Option<String>,
    pub reply: oneshot::Sender<bool>,
}

/// 工具权限入口，UI 或测试实现可异步给出本次调用的批准结果。
#[async_trait::async_trait]
pub trait ToolApprover: Send + Sync {
    /// 返回 true 才允许执行；丢弃 Future 表示该次审批已被取消。
    async fn approve(&self, tool: &str, arguments: &str, call_id: Option<&str>) -> bool;
}

/// 将审批请求投递给 CLI，使用单调编号隔离旧请求与新请求。
pub struct ChannelApprover {
    sender: mpsc::UnboundedSender<ApprovalRequest>,
    next_id: AtomicU64,
}

impl ChannelApprover {
    /// 创建审批通道；每个 Agent 会话只创建一次，取消不重置编号。
    pub fn channel() -> (Self, mpsc::UnboundedReceiver<ApprovalRequest>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        (
            Self {
                sender,
                next_id: AtomicU64::new(1),
            },
            receiver,
        )
    }
}

#[async_trait::async_trait]
impl ToolApprover for ChannelApprover {
    /// UI 不可用、回复丢失或通道关闭时拒绝执行，不隐式授权。
    async fn approve(&self, tool: &str, arguments: &str, call_id: Option<&str>) -> bool {
        let (reply, response) = oneshot::channel();
        let request = ApprovalRequest {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            tool_name: tool.into(),
            arguments: arguments.into(),
            call_id: call_id.map(str::to_owned),
            reply,
        };
        if self.sender.send(request).is_err() {
            return false;
        }
        response.await.unwrap_or(false)
    }
}
