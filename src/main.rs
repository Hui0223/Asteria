mod terminal;
mod tui;

use anyhow::{Context, Result, bail};
use asteria_agent::{
    agent::Asteria,
    agent_loop::{CancelToken, TurnState},
    permission::ChannelApprover,
    rag::RagStore,
};
use std::collections::VecDeque;
use terminal::{InputEvent, Output, Terminal};
use tui::ApprovalUi;

/// 启动命令行 Agent，由行编辑器负责输入与屏幕重绘。
#[tokio::main]
async fn main() -> Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    dotenvy::dotenv().ok();
    let mut agent = Asteria::new()?;
    let rag_chunks = match load_default_rag_store()? {
        Some(store) => {
            let count = store.len();
            agent.enable_search_docs(store);
            Some(count)
        }
        None => None,
    };
    let (approver, requests) = ChannelApprover::channel();
    agent.set_tool_approver(std::sync::Arc::new(approver));
    let mut approvals = ApprovalUi::new(requests);
    let mut terminal = Terminal::start()?;
    let tui_state = tui::new_state();
    agent.set_event_sink(std::sync::Arc::new(tui::TuiEventSink::new(
        terminal.output.clone(),
        tui_state.clone(),
    )));
    let trust_project_mcp = arguments
        .iter()
        .any(|argument| argument == "--trust-project-mcp");
    terminal.output.print("[MCP] 正在加载配置并连接 Server...");
    if let Err(error) = agent.enable_mcp(trust_project_mcp).await {
        terminal
            .output
            .print(format!("[MCP] 配置加载失败，Asteria 将继续启动：{error:#}"));
    }
    terminal.output.print(format!(
        "Asteria · {}\n{} · {}\n/usage /trace /mcp /permissions /verbose · Ctrl+C 取消 · /exit 退出",
        agent.model(),
        rag_chunks.map_or_else(|| "RAG 未加载".into(), |count| format!("RAG {count} chunks")),
        mcp_summary(&agent)
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
            "/context" => print_context(&agent, &terminal.output),
            "/usage" => print_usage(&agent, &terminal.output),
            command if command.starts_with("/mcp") => {
                mcp_command(&mut agent, command, trust_project_mcp, &terminal.output).await
            }
            command if command.starts_with("/trace") => {
                print_trace(&agent, command, &terminal.output)
            }
            "/session" => {
                let paths = agent.session_paths();
                terminal.output.print(format!(
                    "[Session]\n目录：{}\n上下文：{}\n审计：{}\n状态：{}",
                    paths.directory.display(),
                    paths.context.display(),
                    paths.trace.display(),
                    paths.state.display()
                ));
            }
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
                        if !tui::take_answer_streamed(&tui_state) {
                            terminal.output.print(format!("Asteria\n{answer}"));
                        }
                        print_turn_summary(&agent, &tui_state, &terminal.output);
                    }
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
            }
        }
    }
    terminal.finish();
    Ok(())
}

/// 加载主 TUI 默认使用的本地 RAG 文档目录。
fn load_default_rag_store() -> Result<Option<RagStore>> {
    let path = std::path::Path::new("docs/rag-docs");
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(RagStore::from_dir(path, 500, 50)?))
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

/// 切换默认紧凑视图与完整事件日志；只影响显示，不改变持久化事件。
fn verbose_command(input: &str, state: &tui::SharedTuiState, output: &Output) -> bool {
    let parts = input.split_whitespace().collect::<Vec<_>>();
    match parts.as_slice() {
        ["/verbose"] => output.print(format!(
            "[界面] 当前为{}模式。用法：/verbose on|off",
            if tui::is_verbose(state) {
                "详细"
            } else {
                "紧凑"
            }
        )),
        ["/verbose", "on"] => tui::set_verbose(state, true, output),
        ["/verbose", "off"] => tui::set_verbose(state, false, output),
        ["/verbose", ..] => output.print("用法：/verbose on|off"),
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

/// 显示 MCP 配置来源、连接状态和注册后的命名空间工具。
fn print_mcp(agent: &Asteria, output: &Output) {
    let Some(manager) = agent.mcp_manager() else {
        output.print("[MCP] 尚未加载。");
        return;
    };
    let mut lines = vec![format!(
        "[MCP] 已连接 {}/{} 个 Server，注册 {} 个工具",
        manager.active_connections(),
        manager.statuses().len(),
        manager
            .statuses()
            .iter()
            .map(|server| server.tools.len())
            .sum::<usize>()
    )];
    if manager.loaded_files().is_empty() {
        lines.push("配置：未找到 ~/.asteria/mcp.json".into());
    } else {
        lines.push(format!(
            "配置：{}",
            manager
                .loaded_files()
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if let Some(path) = manager.skipped_project_file() {
        lines.push(format!(
            "项目配置已跳过：{}（使用 --trust-project-mcp 显式信任）",
            path.display()
        ));
    }
    if manager.statuses().is_empty() && !manager.loaded_files().is_empty() {
        lines
            .push("提示：配置文件已读取，但 mcpServers 为空；当前没有可连接的 MCP Server。".into());
    }
    for server in manager.statuses() {
        lines.push(format!(
            "- {}: {} · {} 个工具{}",
            server.name,
            server.state,
            server.tools.len(),
            server
                .detail
                .as_ref()
                .map(|detail| format!(" · {detail}"))
                .unwrap_or_default()
        ));
        if !server.tools.is_empty() {
            lines.push(format!("  {}", server.tools.join(", ")));
        }
    }
    output.print(lines.join("\n"));
}

fn mcp_summary(agent: &Asteria) -> String {
    let Some(manager) = agent.mcp_manager() else {
        return "MCP 未加载".into();
    };
    let tools = manager
        .statuses()
        .iter()
        .map(|server| server.tools.len())
        .sum::<usize>();
    format!(
        "MCP {}/{} servers · {tools} tools",
        manager.active_connections(),
        manager.statuses().len()
    )
}

/// 在空闲状态执行 MCP 生命周期命令，避免 Turn 进行中修改模型工具列表。
async fn mcp_command(agent: &mut Asteria, command: &str, trust_project: bool, output: &Output) {
    let parts = command.split_whitespace().collect::<Vec<_>>();
    match parts.as_slice() {
        ["/mcp"] | ["/mcp", "status"] => print_mcp(agent, output),
        ["/mcp", "tools"] => {
            let Some(manager) = agent.mcp_manager() else {
                output.print("[MCP] 尚未加载。");
                return;
            };
            let tools = manager
                .statuses()
                .iter()
                .flat_map(|server| {
                    server
                        .tools
                        .iter()
                        .map(move |tool| format!("{}: {tool}", server.name))
                })
                .collect::<Vec<_>>();
            output.print(if tools.is_empty() {
                "[MCP] 当前没有已注册工具。".into()
            } else {
                format!("[MCP] 已注册工具\n{}", tools.join("\n"))
            });
        }
        ["/mcp", "reload"] => match agent.reload_mcp(trust_project).await {
            Ok(()) => {
                output.print("[MCP] 配置已原子重载。");
                print_mcp(agent, output);
            }
            Err(error) => output.print(format!("[MCP] 重载失败：{error:#}")),
        },
        ["/mcp", "connect", server] => {
            match agent.connect_mcp_server(trust_project, server).await {
                Ok(()) => {
                    output.print(format!("[MCP] Server `{server}` 已连接。"));
                    print_mcp(agent, output);
                }
                Err(error) => output.print(format!("[MCP] 连接失败：{error:#}")),
            }
        }
        ["/mcp", "disconnect", server] => match agent.disconnect_mcp_server(server).await {
            Ok(()) => {
                output.print(format!("[MCP] Server `{server}` 已断开。"));
                print_mcp(agent, output);
            }
            Err(error) => output.print(format!("[MCP] 断开失败：{error:#}")),
        },
        _ => output.print("用法：/mcp [status|tools|reload|connect <server>|disconnect <server>]"),
    }
}

/// 显示最近 Turn 或指定 Turn 的脱敏工具调用记录，不读取工具结果正文。
fn print_trace(agent: &Asteria, command: &str, output: &Output) {
    let parts: Vec<_> = command.split_whitespace().collect();
    let turn_id = match parts.as_slice() {
        ["/trace"] => agent.last_turn().map(|turn| turn.id),
        ["/trace", value] => match value.parse::<u64>() {
            Ok(turn_id) => Some(turn_id),
            Err(_) => {
                output.print("用法：/trace 或 /trace <turn_id>");
                return;
            }
        },
        _ => {
            output.print("用法：/trace 或 /trace <turn_id>");
            return;
        }
    };
    let Some(turn_id) = turn_id else {
        output.print("[ToolTrace] 尚未执行 Turn。");
        return;
    };
    let traces = match agent.tool_traces(Some(turn_id)) {
        Ok(traces) => traces,
        Err(error) => {
            output.print(format!("读取 ToolTrace 失败: {error:#}"));
            return;
        }
    };
    if traces.is_empty() {
        output.print(format!("[ToolTrace] Turn {turn_id} 没有工具调用记录。"));
        return;
    }
    let mut lines = vec![format!(
        "[ToolTrace] Turn {turn_id} · {} 次调用（审计记录不参与模型上下文）",
        traces.len()
    )];
    for (index, trace) in traces.iter().enumerate() {
        lines.push(format!(
            "{}. Step {} · {} · {}\n调用：{}\n参数：{}\n参数哈希：{} · 结果哈希：{}\n耗时：{}ms · 完成：{}",
            index + 1,
            trace.step,
            trace.tool_name,
            match trace.status {
                asteria_agent::events::ToolTraceStatus::Success => "成功",
                asteria_agent::events::ToolTraceStatus::Error => "失败",
            },
            trace.call_id,
            trace.arguments_preview,
            trace.arguments_hash,
            trace.result_hash,
            trace.duration_ms,
            trace.completed_at,
        ));
    }
    output.print(lines.join("\n"));
}
