mod boot;
mod commands;
mod rpc;
mod terminal;
mod tui;

use anyhow::{Context, Result, bail};
use asteria_agent::{
    agent::Asteria,
    agent_loop::{CancelToken, TurnState},
    permission::ChannelApprover,
};
use boot::PreparedAgent;
use std::collections::VecDeque;
use terminal::{InputEvent, Output, Terminal};
use tui::ApprovalUi;

/// 默认启动 TUI；`--rpc` 给编辑器侧栏用，不创建 Reedline。
#[tokio::main]
async fn main() -> Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    dotenvy::dotenv().ok();
    if arguments.iter().any(|argument| argument == "--rpc") {
        return rpc::run(&arguments).await;
    }
    let PreparedAgent {
        mut agent,
        rag_chunks,
        mut trust_project_mcp,
    } = boot::prepare_agent(&arguments)?;
    let (approver, requests) = ChannelApprover::channel();
    agent.set_tool_approver(std::sync::Arc::new(approver));
    let mut approvals = ApprovalUi::new(requests);
    let mut terminal = Terminal::start()?;
    let tui_state = tui::new_state();
    agent.set_event_sink(std::sync::Arc::new(tui::TuiEventSink::new(
        terminal.output.clone(),
        tui_state.clone(),
    )));
    terminal.output.print("[MCP] 正在加载配置并连接 Server...");
    if let Err(error) = agent.enable_mcp(trust_project_mcp).await {
        terminal
            .output
            .print(format!("[MCP] 配置加载失败，Asteria 将继续启动：{error:#}"));
    }
    terminal.set_busy(false);
    terminal.output.print(format!(
        "Asteria · {}\n{} · {}\n/usage /trace /mcp /permissions /verbose · Ctrl+C 取消 · /exit 退出",
        agent.model(),
        boot::rag_label(rag_chunks),
        boot::mcp_label(&agent)
    ));
    let mut queued: VecDeque<String> = VecDeque::new();
    loop {
        let input = if let Some(line) = queued.pop_front() {
            terminal.output.print("↳ 开始处理排队问题");
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
        if verbose_command(&input, &tui_state, &terminal.output) {
            continue;
        }
        match input.trim() {
            "/exit" => break,
            "/cancel" => terminal.output.print("当前没有运行中的 Turn。"),
            "/context" => terminal.output.print(commands::format_context(&agent)),
            "/usage" => terminal.output.print(commands::format_usage(&agent)),
            command if command.starts_with("/mcp") => terminal.output.print(
                commands::run_mcp_command(&mut agent, command, &mut trust_project_mcp).await,
            ),
            command if command.starts_with("/trace") => terminal
                .output
                .print(commands::format_trace(&agent, command)),
            "/session" => terminal.output.print(commands::format_session(&agent)),
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
                    Ok(answer) => {
                        terminal.set_busy(false);
                        if !tui::take_answer_streamed(&tui_state) {
                            terminal.output.print(format!("\nAsteria\n{answer}"));
                        }
                        print_turn_summary(&agent, &tui_state, &terminal.output);
                    }
                    Err(_)
                        if agent
                            .last_turn()
                            .is_some_and(|turn| turn.state == TurnState::Cancelled) =>
                    {
                        terminal.set_busy(false);
                        terminal
                            .output
                            .print("=== Turn 已取消 ===\n当前问题没有完成，可以继续提问。");
                    }
                    Err(error) => {
                        terminal.set_busy(false);
                        terminal
                            .output
                            .print(format!("=== 执行失败 ===\n原因：{error:#}"));
                    }
                }
                approvals.clear();
            }
        }
    }
    terminal.finish();
    Ok(())
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
        .print("… 正在处理 · Enter 排队 · Ctrl+C 或 /cancel 取消");
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
        ["/permissions"] => output.print(commands::format_permissions(agent)),
        ["/permission", name, value] => output.print(commands::set_permission(agent, name, value)),
        ["/permissions" | "/permission", ..] => output.print(commands::permission_usage()),
        _ => return false,
    }
    true
}

/// 切换默认紧凑视图与完整事件日志；只影响显示，不改变持久化事件。
fn verbose_command(input: &str, state: &tui::SharedTuiState, output: &Output) -> bool {
    let parts = input.split_whitespace().collect::<Vec<_>>();
    match parts.as_slice() {
        ["/verbose"] => output.print(commands::format_verbose(
            input,
            Some(tui::is_verbose(state)),
        )),
        ["/verbose", "on"] => tui::set_verbose(state, true, output),
        ["/verbose", "off"] => tui::set_verbose(state, false, output),
        ["/verbose", ..] => output.print("用法：/verbose on|off"),
        _ => return false,
    }
    true
}

/// Turn 结束后只显示一行摘要，完整统计仍由 /usage 提供。
fn print_turn_summary(agent: &Asteria, state: &tui::SharedTuiState, output: &Output) {
    let Some(turn) = agent.last_turn() else {
        return;
    };
    let elapsed = tui::take_turn_elapsed(state, turn.id);
    output.print(format!(
        "  {}",
        tui::render::turn_summary(turn.steps, turn.retries, turn.usage.total_tokens, elapsed)
    ));
}
