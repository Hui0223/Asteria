use crate::terminal::Output;
use asteria_agent::permission::ApprovalRequest;
use std::collections::HashMap;
use tokio::sync::{mpsc, oneshot};

/// 当前 Turn 的待批准调用；通过唯一审批编号支持同批次多个工具。
pub struct ApprovalUi {
    pub receiver: mpsc::UnboundedReceiver<ApprovalRequest>,
    pending: HashMap<u64, oneshot::Sender<bool>>,
}

impl ApprovalUi {
    /// 连接 Agent 的审批通道。
    pub fn new(receiver: mpsc::UnboundedReceiver<ApprovalRequest>) -> Self {
        Self {
            receiver,
            pending: HashMap::new(),
        }
    }

    /// 显示仍有效的审批请求；输入已关闭时明确拒绝，避免管道模式永久等待。
    pub fn present(&mut self, request: ApprovalRequest, output: &Output, input_open: bool) {
        if request.reply.is_closed() {
            return;
        }
        if !input_open {
            let _ = request.reply.send(false);
            output.print("[权限拒绝] 输入已关闭，无法取得人工批准。");
            return;
        }
        output.print(format!(
            "[待批准 #{}] 工具={:?} call_id={:?}\n参数：{:?}\n输入 /approve {} 批准本次，/deny {} 拒绝，或 /cancel 取消整个 Turn。",
            request.id, request.tool_name, request.call_id, request.arguments, request.id, request.id,
        ));
        self.pending.insert(request.id, request.reply);
    }

    /// 精确识别用户批准命令；无效、过期或重复编号不授权，也不发给模型。
    pub fn respond(&mut self, line: &str, output: &Output) -> bool {
        let parts: Vec<_> = line.split_whitespace().collect();
        let Some(command @ ("/approve" | "/deny")) = parts.first().copied() else {
            return false;
        };
        if parts.len() != 2 {
            output.print("用法：/approve <审批编号> 或 /deny <审批编号>");
            return true;
        }
        let Ok(id) = parts[1].parse::<u64>() else {
            output.print("审批编号必须是整数。");
            return true;
        };
        match self.pending.remove(&id) {
            Some(reply) if !reply.is_closed() => {
                let allowed = command == "/approve";
                if reply.send(allowed).is_ok() {
                    output.print(format!(
                        "[审批 #{id}] {}（仅本次调用）",
                        if allowed { "已批准" } else { "已拒绝" }
                    ));
                } else {
                    output.print("审批已失效，工具不会因此启动。");
                }
            }
            _ => output.print("没有该编号的有效审批，请使用当前显示的编号。"),
        }
        true
    }

    /// 结束 Turn 或输入 EOF 时拒绝所有尚未处理的批准请求。
    pub fn clear(&mut self) {
        for (_, reply) in self.pending.drain() {
            let _ = reply.send(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    /// 普通文本不算批准，只有精确命令和有效编号才能消费一次审批。
    async fn approval_requires_explicit_command() {
        let (_, receiver) = mpsc::unbounded_channel();
        let mut ui = ApprovalUi::new(receiver);
        let (reply, mut response) = oneshot::channel();
        ui.pending.insert(7, reply);
        assert!(!ui.respond("请批准7", &Output::default()));
        assert!(response.try_recv().is_err());
        assert!(ui.respond("/approve 6", &Output::default()));
        assert!(response.try_recv().is_err());
        assert!(ui.respond("/approve 7 extra", &Output::default()));
        assert!(response.try_recv().is_err());
        assert!(ui.respond("/approve 7", &Output::default()));
        assert!(response.await.unwrap());
        assert!(ui.pending.is_empty());
        assert!(ui.respond("/approve 7", &Output::default()));
    }

    #[tokio::test]
    /// 清理或输入关闭都拒绝待批准工具，不能留下挂起的批准通道。
    async fn closed_input_denies_pending_approvals() {
        let (_, receiver) = mpsc::unbounded_channel();
        let mut ui = ApprovalUi::new(receiver);
        let (reply, response) = oneshot::channel();
        ui.pending.insert(1, reply);
        ui.clear();
        assert!(!response.await.unwrap());
        let (reply, response) = oneshot::channel();
        ui.present(
            ApprovalRequest {
                id: 2,
                tool_name: "test".into(),
                arguments: "{}".into(),
                call_id: None,
                reply,
            },
            &Output::default(),
            false,
        );
        assert!(!response.await.unwrap());
        assert!(ui.pending.is_empty());
    }
}
