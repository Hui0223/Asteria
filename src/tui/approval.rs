use crate::terminal::Output;
use asteria_agent::permission::ApprovalRequest;
use std::collections::HashMap;
use tokio::sync::{mpsc, oneshot};

const MAX_ARGUMENT_LINES: usize = 6;
const MAX_ARGUMENT_CHARS: usize = 800;

/// 当前 Turn 的待批准调用；通过唯一审批编号支持同批次多个工具。
pub struct ApprovalUi {
    pub receiver: mpsc::UnboundedReceiver<ApprovalRequest>,
    pending: HashMap<u64, oneshot::Sender<bool>>,
}

impl ApprovalUi {
    pub fn new(receiver: mpsc::UnboundedReceiver<ApprovalRequest>) -> Self {
        Self {
            receiver,
            pending: HashMap::new(),
        }
    }

    /// 以紧凑面板展示工具与参数；输入关闭时安全拒绝。
    pub fn present(&mut self, request: ApprovalRequest, output: &Output, input_open: bool) {
        if request.reply.is_closed() {
            return;
        }
        if !input_open {
            let _ = request.reply.send(false);
            output.print("× 输入已关闭，工具请求已拒绝。");
            return;
        }
        let arguments = format_arguments(&request.arguments)
            .lines()
            .map(|line| format!("│   {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        output.print(format!(
            "╭─ 需要批准 #{}\n│ 工具：{}\n│ 参数：\n{}\n╰─ /approve {} 批准一次 · /deny {} 拒绝 · /cancel 取消 Turn",
            request.id,
            display_tool_name(&request.tool_name),
            arguments,
            request.id,
            request.id,
        ));
        self.pending.insert(request.id, request.reply);
    }

    /// 只有精确命令和仍有效的编号才能消费一次审批。
    pub fn respond(&mut self, line: &str, output: &Output) -> bool {
        let parts = line.split_whitespace().collect::<Vec<_>>();
        let Some(command @ ("/approve" | "/deny")) = parts.first().copied() else {
            return false;
        };
        if parts.len() != 2 {
            output.print("用法：/approve <编号> 或 /deny <编号>");
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
                        "{} 审批 #{id} {}（仅本次）",
                        if allowed { "✓" } else { "×" },
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

    pub fn clear(&mut self) {
        for (_, reply) in self.pending.drain() {
            let _ = reply.send(false);
        }
    }
}

fn display_tool_name(name: &str) -> String {
    name.strip_prefix("mcp__")
        .unwrap_or(name)
        .replacen("__", ".", 1)
}

fn format_arguments(raw: &str) -> String {
    let formatted = serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .and_then(|value| serde_json::to_string_pretty(&value).ok())
        .unwrap_or_else(|| raw.to_owned());
    let mut output = String::new();
    let mut truncated = false;
    for (index, line) in formatted.lines().enumerate() {
        if index >= MAX_ARGUMENT_LINES
            || output.chars().count() + line.chars().count() > MAX_ARGUMENT_CHARS
        {
            truncated = true;
            break;
        }
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(line);
    }
    if truncated {
        output.push_str("\n… 参数预览已截断");
    }
    if output.is_empty() {
        "{}".into()
    } else {
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
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

    #[test]
    fn formats_json_and_truncates_long_arguments() {
        assert_eq!(
            format_arguments(r#"{"path":"src/lib.rs"}"#),
            "{\n  \"path\": \"src/lib.rs\"\n}"
        );
        let long = serde_json::json!({"content": "x".repeat(1_000)}).to_string();
        assert!(format_arguments(&long).contains("参数预览已截断"));
    }
}
