mod approval_ui;
mod terminal;

use anyhow::{Context, Result, bail};
use approval_ui::ApprovalUi;
use asteria_agent::{
    agent::Asteria,
    agent_loop::{CancelToken, TurnState},
    events::{AgentEvent, EventSink},
    permission::ChannelApprover,
};
use std::collections::VecDeque;
use terminal::{InputEvent, Output, Terminal};

/// 启动命令行 Agent，由行编辑器负责输入与屏幕重绘。
#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let mut agent = Asteria::new()?;
    let (approver, requests) = ChannelApprover::channel();
    agent.set_tool_approver(std::sync::Arc::new(approver));
    let mut approvals = ApprovalUi::new(requests);
    let mut terminal = Terminal::start()?;
    let retry_output = terminal.output.clone();
    agent.set_retry_output(move |message| retry_output.print(message));
    agent.set_event_sink(std::sync::Arc::new(TuiEventSink {
        output: terminal.output.clone(),
    }));
    terminal.output.print(format!(
        "=== Asteria 已启动 ===\n模型：{}\n命令：/context 记忆 | /usage 用量 | /reset 清空 | /exit 退出\n取消：Ctrl+C 或 /cancel",
        agent.model()
    ));
    terminal.output.print(
        "工具权限：/permissions 查看；/permission <工具名> allow|deny|ask 设置；待批准时使用 /approve <编号> 或 /deny <编号>。",
    );
    let mut queued: VecDeque<String> = VecDeque::new();
    loop {
        let input = if let Some(line) = queued.pop_front() {
            terminal
                .output
                .print(format!("--- 开始处理排队问题 ---\n用户：{}", line.trim()));
            line
        } else {
            tokio::select! {
                event = terminal.input.recv() => match event {
                    Some(InputEvent::Line(line)) => line,
                    Some(InputEvent::Cancel) => {
                        terminal.output.print("当前没有运行中的 Turn，已清空输入行。");
                        continue;
                    }
                    Some(InputEvent::Error(error)) => { terminal.output.print(format!("输入错误: {error}")); break; }
                    None => break,
                },
                // Ctrl+C 按键由编辑器投递；也接收宿主直接发送的 SIGINT。
                signal = tokio::signal::ctrl_c() => {
                    signal.context("无法监听 Ctrl+C")?;
                    terminal.output.print("当前没有运行中的 Turn，输入 /exit 退出。");
                    continue;
                }
            }
        };
        if approvals.respond(&input, &terminal.output) {
            continue;
        }
        if permission_command(&mut agent, &input, &terminal.output) {
            continue;
        }
        match input.trim() {
            "/exit" => break,
            "/cancel" => terminal.output.print("当前没有运行中的 Turn。"),
            "/context" => print_context(&agent, &terminal.output),
            "/usage" => print_usage(&agent, &terminal.output),
            "/session" => terminal
                .output
                .print(format!("[Session] {}", agent.session_path().display())),
            "/new-session" => match agent.new_session() {
                Ok(()) => terminal.output.print("[Session] 已创建新的空会话。"),
                Err(error) => terminal.output.print(format!("创建新会话失败: {error:#}")),
            },
            "/reset" => {
                agent.reset();
                terminal.output.print("Asteria: 记忆已清空。");
            }
            "" => {}
            text => {
                match ask_interruptible(
                    &mut agent,
                    text,
                    &mut terminal,
                    &mut queued,
                    &mut approvals,
                )
                .await
                {
                    Ok(answer) => terminal
                        .output
                        .print(format!("=== Asteria 回答 ===\n{answer}")),
                    Err(_)
                        if agent
                            .last_turn()
                            .is_some_and(|turn| turn.state == TurnState::Cancelled) =>
                    {
                        terminal
                            .output
                            .print("=== Turn 已取消 ===\n当前问题没有完成，可以继续提问。");
                    }
                    Err(error) => terminal
                        .output
                        .print(format!("=== 执行失败 ===\n原因：{error:#}")),
                }
                approvals.clear();
                print_usage(&agent, &terminal.output);
            }
        }
    }
    terminal.finish();
    Ok(())
}

/// 将结构化 AgentEvent 转换成清晰的 TUI 事件日志。
struct TuiEventSink {
    output: Output,
}

impl EventSink for TuiEventSink {
    /// 将事件交给 Reedline 外部输出通道，避免破坏当前输入行。
    fn publish(&self, event: AgentEvent) {
        let line = match event {
            AgentEvent::TurnStarted { turn_id, input } => {
                format!("[执行] Turn {turn_id} 开始\n用户：{}", preview(&input))
            }
            AgentEvent::StepStarted { turn_id, step } => {
                format!("[执行] Turn {turn_id} · Step {step}\n动作：请求模型")
            }
            AgentEvent::ToolCallStarted {
                turn_id,
                call_id,
                name,
            } => format!("[工具] Turn {turn_id} 开始调用\n工具：{name}\n调用：{call_id}"),
            AgentEvent::ToolResult {
                turn_id,
                call_id,
                is_error,
                duration_ms,
            } => format!(
                "[工具] Turn {turn_id} 调用结束\n调用：{call_id}\n结果：{} · 耗时：{}ms",
                if is_error { "失败" } else { "成功" },
                duration_ms
            ),
            AgentEvent::PermissionRequested {
                turn_id,
                call_id,
                name,
            } => {
                format!("[权限] Turn {turn_id} 等待确认\n工具：{name}\n调用：{call_id}")
            }
            AgentEvent::PermissionResolved {
                turn_id,
                call_id,
                allowed,
            } => {
                format!(
                    "[权限] Turn {turn_id} {}\n调用：{call_id}",
                    if allowed { "已批准" } else { "已拒绝" }
                )
            }
            AgentEvent::StepCompleted {
                turn_id,
                step,
                usage,
            } => format!(
                "[执行] Turn {turn_id} · Step {step} 完成\n本次 Token：{}",
                usage.total_tokens
            ),
            AgentEvent::StepRetrying {
                turn_id,
                step,
                failed_attempt,
                next_attempt,
                max_attempts,
                delay_ms,
            } => format!(
                "[重试] Turn {turn_id} · Step {step}\n第 {failed_attempt} 次失败，准备第 {next_attempt}/{max_attempts} 次\n等待：{delay_ms}ms"
            ),
            AgentEvent::TurnCompleted { turn_id, steps } => {
                format!("[完成] Turn {turn_id}\n共执行：{steps} 个 Step")
            }
            AgentEvent::TurnCancelled { turn_id } => format!("[取消] Turn {turn_id}\n状态：已取消"),
            AgentEvent::TurnFailed { turn_id, message } => {
                format!("[失败] Turn {turn_id}\n原因：{message}")
            }
        };
        self.output.print(line);
    }
}

/// 压缩事件中的用户输入，避免长问题淹没 TUI。
fn preview(input: &str) -> String {
    let compact = input.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut result = compact.chars().take(120).collect::<String>();
    if compact.chars().count() > 120 {
        result.push('…');
    }
    result
}

/// 同时等待回答、输入与取消；普通输入排队，取消后等待回滚完成。
async fn ask_interruptible(
    agent: &mut Asteria,
    input: &str,
    terminal: &mut Terminal,
    queued: &mut VecDeque<String>,
    approvals: &mut ApprovalUi,
) -> Result<String> {
    terminal
        .output
        .print("--- 正在处理 ---\n可以继续输入并回车排队；输入 /cancel 或按 Ctrl+C 取消当前轮。");
    let cancel = CancelToken::new();
    let request = agent.ask_with_cancel(input, &cancel);
    tokio::pin!(request);
    let mut input_open = true;
    let mut approvals_open = true;
    loop {
        tokio::select! {
            biased;
            signal = tokio::signal::ctrl_c() => {
                terminal.output.print("[取消] 收到 Ctrl+C，正在取消当前 Turn。");
                cancel.cancel();
                let result = request.await;
                signal.context("无法监听 Ctrl+C")?;
                return result;
            }
            result = &mut request => return result,
            approval = approvals.receiver.recv(), if approvals_open => {
                match approval {
                    Some(approval) => approvals.present(approval, &terminal.output, input_open),
                    None => approvals_open = false,
                }
            }
            event = terminal.input.recv(), if input_open => {
                match event {
                    Some(InputEvent::Cancel) => {
                        terminal.output.print("[取消] 收到 Ctrl+C，正在取消当前 Turn。");
                        cancel.cancel();
                        return request.await;
                    }
                    Some(InputEvent::Line(line)) if line.trim() == "/cancel" => {
                        terminal.output.print("[取消] 收到 /cancel，正在取消当前 Turn。");
                        cancel.cancel();
                        return request.await;
                    }
                    Some(InputEvent::Line(line)) if line.trim().is_empty() => {},
                    Some(InputEvent::Line(line)) => {
                        if approvals.respond(&line, &terminal.output) { continue; }
                        queued.push_back(line);
                        terminal.output.print(format!("[已排队] 当前有 {} 条待处理输入，将按顺序执行。", queued.len()));
                    },
                    Some(InputEvent::Error(error)) => {
                        cancel.cancel();
                        let _ = request.await;
                        bail!("输入错误: {error}");
                    }
                    // EOF 不取消已提交的问题，仍等待答案并处理已排队的消息。
                    None => { input_open = false; approvals.clear(); },
                }
            }
        }
    }
}

/// 只在空闲时更改会话权限；命令错误不会成为模型提示词，也不会默认允许。
fn permission_command(agent: &mut Asteria, input: &str, output: &Output) -> bool {
    let parts: Vec<_> = input.split_whitespace().collect();
    match parts.as_slice() {
        ["/permissions"] => output.print(
            agent
                .tool_permissions()
                .into_iter()
                .map(|(name, permission)| format!("{name}: {permission}"))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        ["/permission", name, value] => {
            match value
                .parse()
                .and_then(|permission| agent.set_tool_permission(name, permission))
            {
                Ok(()) => output.print(format!("[权限] {name} = {value}（当前会话；/reset 保留）")),
                Err(error) => output.print(format!("权限设置失败: {error}")),
            }
        }
        ["/permissions" | "/permission", ..] => {
            output.print("用法：/permissions 或 /permission <工具名> allow|deny|ask")
        }
        _ => return false,
    }
    true
}

/// 显示原始记忆；Debug 转义控制字符，整块输出后由编辑器恢复输入行。
fn print_context(agent: &Asteria, output: &Output) {
    let context = agent.context();
    let mut lines = vec![
        format!(
            "[ContextMemory] messages={}（不含 System；原始历史，不是本次模型请求）",
            context.messages().len()
        ),
        format!("System: {:?}", context.system_prompt()),
    ];
    for (index, message) in context.messages().iter().enumerate() {
        lines.push(format!("{}: {:?}", index + 1, message));
    }
    output.print(lines.join("\n"));
}

/// 成功或失败都显示报告；合并输出，避免逐行打印打断正在编辑的文字。
fn print_usage(agent: &Asteria, output: &Output) {
    let mut lines = Vec::new();
    if let Some(turn) = agent.last_turn() {
        lines.push(format!(
            "=== 最近一次 Turn：{} ===\n状态：{:?} · 步骤：{} · 重试：{}\n上下文消息：{}",
            turn.id,
            turn.state,
            turn.steps,
            turn.retries,
            agent.context().messages().len()
        ));
        lines.push(format!(
            "本轮用量：输入 {} · 输出 {} · 合计 {}",
            turn.usage.prompt_tokens, turn.usage.completion_tokens, turn.usage.total_tokens
        ));
    } else {
        lines.push("尚未执行 Turn。".into());
    }
    let usage = agent.session_usage();
    lines.push(format!(
        "=== 当前会话累计用量 ===\n输入 {} · 输出 {} · 合计 {}",
        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
    ));
    lines
        .push("说明：只累计模型实际返回的 usage；未返回的用量未知。/reset 不清空累计用量。".into());
    output.print(lines.join("\n"));
}
