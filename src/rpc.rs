use crate::boot::{self, PreparedAgent};
use crate::commands::{self, SlashAction};
use anyhow::{Context, Result};
use asteria_agent::{
    agent::Asteria,
    agent_loop::{CancelToken, TurnState},
    events::{AgentEvent, EventSink},
    permission::{ApprovalRequest, ChannelApprover},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, VecDeque},
    io::{Write, stdout},
    sync::{Arc, Mutex},
};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    sync::{mpsc, oneshot},
};

/// 启动 JSONL RPC 宿主：stdin 收命令，stdout 推事件。不创建 TUI。
pub async fn run(arguments: &[String]) -> Result<()> {
    let PreparedAgent {
        mut agent,
        rag_chunks,
        mut trust_project_mcp,
    } = boot::prepare_agent(arguments)?;
    let (approver, mut approvals) = ChannelApprover::channel();
    agent.set_tool_approver(Arc::new(approver));
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    agent.set_event_sink(Arc::new(RpcEventSink::new(event_tx)));
    if let Err(error) = agent.enable_mcp(trust_project_mcp).await {
        emit(&ClientEvent::Log {
            text: format!("MCP 配置加载失败，Asteria 将继续启动：{error:#}"),
        });
    }
    emit(&ClientEvent::Ready {
        model: agent.model().to_owned(),
        rag: boot::rag_label(rag_chunks),
        mcp: boot::mcp_label(&agent),
    });

    let (command_tx, mut command_rx) = mpsc::unbounded_channel();
    tokio::spawn(read_commands(command_tx));

    let mut pending: HashMap<u64, oneshot::Sender<bool>> = HashMap::new();
    let mut queued: VecDeque<String> = VecDeque::new();
    loop {
        let text = if let Some(text) = queued.pop_front() {
            emit(&ClientEvent::Log {
                text: "开始处理排队问题".into(),
            });
            text
        } else {
            loop {
                tokio::select! {
                    event = event_rx.recv() => {
                        if let Some(event) = event {
                            emit(&event);
                        }
                    }
                    request = approvals.recv() => {
                        if let Some(request) = request {
                            present_approval(request, &mut pending);
                        }
                    }
                    command = command_rx.recv() => {
                        let Some(command) = command else { return Ok(()) };
                        match command {
                            RpcCommand::Ask { text } => match handle_user_input(
                                &mut agent,
                                &text,
                                rag_chunks,
                                &mut trust_project_mcp,
                                &mut pending,
                                &mut queued,
                            )
                            .await
                            {
                                UserInput::Ask(text) => break text,
                                UserInput::Quit => return Ok(()),
                                UserInput::Handled => {}
                            },
                            other => {
                                idle_command(other, &mut agent, &mut pending, &mut queued);
                            }
                        }
                    }
                }
            }
        };
        match handle_user_input(
            &mut agent,
            &text,
            rag_chunks,
            &mut trust_project_mcp,
            &mut pending,
            &mut queued,
        )
        .await
        {
            UserInput::Ask(text) => {
                run_turn(
                    &mut agent,
                    &text,
                    &mut pending,
                    &mut queued,
                    &mut approvals,
                    &mut command_rx,
                    &mut event_rx,
                )
                .await;
            }
            UserInput::Quit => return Ok(()),
            UserInput::Handled => {}
        }
    }
}

enum UserInput {
    Ask(String),
    Handled,
    Quit,
}

async fn handle_user_input(
    agent: &mut Asteria,
    text: &str,
    rag_chunks: Option<usize>,
    trust_project_mcp: &mut bool,
    pending: &mut HashMap<u64, oneshot::Sender<bool>>,
    queued: &mut VecDeque<String>,
) -> UserInput {
    let text = text.trim();
    if text.is_empty() {
        return UserInput::Handled;
    }
    let Some(action) = commands::classify_slash(text) else {
        return UserInput::Ask(text.to_owned());
    };
    emit(&ClientEvent::Command {
        text: text.to_owned(),
    });
    match action {
        SlashAction::Context => emit_result(commands::format_context(agent)),
        SlashAction::Usage => emit_result(commands::format_usage(agent)),
        SlashAction::Session => emit_result(commands::format_session(agent)),
        SlashAction::Trace { command } => emit_result(commands::format_trace(agent, &command)),
        SlashAction::Mcp { command } => {
            emit_result(commands::run_mcp_command(agent, &command, trust_project_mcp).await);
            emit(&ClientEvent::Ready {
                model: agent.model().to_owned(),
                rag: boot::rag_label(rag_chunks),
                mcp: boot::mcp_label(agent),
            });
        }
        SlashAction::Permissions => emit_result(commands::format_permissions(agent)),
        SlashAction::Permission { name, value } => {
            emit_result(commands::set_permission(agent, &name, &value));
        }
        SlashAction::PermissionUsage => emit_result(commands::permission_usage()),
        SlashAction::Verbose { command } => emit_result(commands::format_verbose(&command, None)),
        SlashAction::Reset => {
            agent.reset();
            emit_result("Asteria: 记忆已清空。");
        }
        SlashAction::NewSession => reset_session(agent, pending, queued),
        SlashAction::Cancel => emit(&ClientEvent::Log {
            text: "当前没有运行中的 Turn。".into(),
        }),
        SlashAction::Exit => {
            emit(&ClientEvent::Exit);
            return UserInput::Quit;
        }
        SlashAction::Approve { id } => resolve_approval(pending, id, true),
        SlashAction::Deny { id } => resolve_approval(pending, id, false),
        SlashAction::ApprovalUsage => emit_result(commands::approval_usage()),
    }
    UserInput::Handled
}

fn emit_result(text: impl Into<String>) {
    emit(&ClientEvent::CommandResult { text: text.into() });
}

fn idle_command(
    command: RpcCommand,
    agent: &mut Asteria,
    pending: &mut HashMap<u64, oneshot::Sender<bool>>,
    queued: &mut VecDeque<String>,
) -> Option<String> {
    match command {
        RpcCommand::Ask { text } => {
            let text = text.trim().to_owned();
            (!text.is_empty()).then_some(text)
        }
        RpcCommand::Approve { approval_id } => {
            resolve_approval(pending, approval_id, true);
            None
        }
        RpcCommand::Deny { approval_id } => {
            resolve_approval(pending, approval_id, false);
            None
        }
        RpcCommand::NewSession => {
            reset_session(agent, pending, queued);
            None
        }
        RpcCommand::Cancel => {
            emit(&ClientEvent::Log {
                text: "当前没有运行中的 Turn。".into(),
            });
            None
        }
        RpcCommand::Status => {
            emit(&ClientEvent::Log {
                text: "空闲。".into(),
            });
            None
        }
        RpcCommand::Unknown { method } => {
            emit(&ClientEvent::Error {
                message: format!("未知命令：{method}"),
            });
            None
        }
    }
}

fn reset_session(
    agent: &mut Asteria,
    pending: &mut HashMap<u64, oneshot::Sender<bool>>,
    queued: &mut VecDeque<String>,
) {
    for (_, reply) in pending.drain() {
        let _ = reply.send(false);
    }
    queued.clear();
    match agent.new_session() {
        Ok(()) => emit(&ClientEvent::SessionCleared),
        Err(error) => emit(&ClientEvent::Error {
            message: format!("{error:#}"),
        }),
    }
}

async fn run_turn(
    agent: &mut Asteria,
    text: &str,
    pending: &mut HashMap<u64, oneshot::Sender<bool>>,
    queued: &mut VecDeque<String>,
    approvals: &mut mpsc::UnboundedReceiver<ApprovalRequest>,
    commands: &mut mpsc::UnboundedReceiver<RpcCommand>,
    events: &mut mpsc::UnboundedReceiver<ClientEvent>,
) {
    emit(&ClientEvent::User {
        text: text.to_owned(),
    });
    emit(&ClientEvent::Status {
        text: "正在处理 · 可继续输入排队".into(),
    });
    let cancel = CancelToken::new();
    let result = {
        let request = agent.ask_with_cancel(text, &cancel);
        tokio::pin!(request);
        loop {
            tokio::select! {
                biased;
                event = events.recv() => {
                    if let Some(event) = event {
                        emit(&event);
                    }
                }
                request = approvals.recv() => {
                    if let Some(request) = request {
                        present_approval(request, pending);
                    }
                }
                command = commands.recv() => {
                    match command {
                        None => {
                            cancel.cancel();
                            break request.await;
                        }
                        Some(RpcCommand::Cancel) => {
                            cancel.cancel();
                        }
                        Some(RpcCommand::Approve { approval_id }) => {
                            resolve_approval(pending, approval_id, true);
                        }
                        Some(RpcCommand::Deny { approval_id }) => {
                            resolve_approval(pending, approval_id, false);
                        }
                        Some(RpcCommand::Ask { text }) => {
                            let text = text.trim().to_owned();
                            if text.is_empty() {
                                continue;
                            }
                            match commands::classify_slash(&text) {
                                Some(SlashAction::Cancel) => cancel.cancel(),
                                Some(SlashAction::Approve { id }) => {
                                    resolve_approval(pending, id, true);
                                }
                                Some(SlashAction::Deny { id }) => {
                                    resolve_approval(pending, id, false);
                                }
                                Some(SlashAction::ApprovalUsage) => {
                                    emit_result(commands::approval_usage());
                                }
                                Some(SlashAction::NewSession) => {
                                    emit(&ClientEvent::Log {
                                        text: "当前回合进行中，结束后再新建会话。".into(),
                                    });
                                }
                                Some(_) | None => {
                                    queued.push_back(text);
                                    emit(&ClientEvent::Log {
                                        text: format!(
                                            "已排队，当前有 {} 条待处理输入",
                                            queued.len()
                                        ),
                                    });
                                }
                            }
                        }
                        Some(RpcCommand::NewSession) => {
                            emit(&ClientEvent::Log {
                                text: "当前回合进行中，结束后再新建会话。".into(),
                            });
                        }
                        Some(_) => {}
                    }
                }
                result = &mut request => break result,
            }
        }
    };
    for (_, reply) in pending.drain() {
        let _ = reply.send(false);
    }
    match result {
        Ok(_) => {
            if let Some(turn) = agent.last_turn() {
                emit(&ClientEvent::TurnCompleted {
                    steps: turn.steps,
                    tokens: turn.usage.total_tokens,
                    retries: turn.retries,
                });
            }
        }
        Err(_)
            if agent
                .last_turn()
                .is_some_and(|turn| turn.state == TurnState::Cancelled) =>
        {
            emit(&ClientEvent::TurnCancelled);
        }
        Err(error) => emit(&ClientEvent::Error {
            message: format!("{error:#}"),
        }),
    }
}

fn present_approval(request: ApprovalRequest, pending: &mut HashMap<u64, oneshot::Sender<bool>>) {
    if request.reply.is_closed() {
        return;
    }
    emit(&ClientEvent::Approval {
        id: request.id,
        tool: display_tool_name(&request.tool_name),
        arguments: preview_arguments(&request.arguments),
    });
    pending.insert(request.id, request.reply);
}

fn resolve_approval(pending: &mut HashMap<u64, oneshot::Sender<bool>>, id: u64, allowed: bool) {
    match pending.remove(&id) {
        Some(reply) if !reply.is_closed() => {
            let _ = reply.send(allowed);
            emit(&ClientEvent::Log {
                text: format!(
                    "审批 #{id} {}（仅本次）",
                    if allowed { "已批准" } else { "已拒绝" }
                ),
            });
        }
        _ => emit(&ClientEvent::Error {
            message: format!("没有编号 #{id} 的有效审批"),
        }),
    }
}

async fn read_commands(sender: mpsc::UnboundedSender<RpcCommand>) {
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_command(line) {
            Ok(command) => {
                if sender.send(command).is_err() {
                    break;
                }
            }
            Err(error) => emit(&ClientEvent::Error {
                message: error.to_string(),
            }),
        }
    }
}

fn emit(event: &ClientEvent) {
    if let Ok(json) = serde_json::to_string(event) {
        let mut out = stdout().lock();
        let _ = writeln!(out, "{json}");
        let _ = out.flush();
    }
}

fn display_tool_name(name: &str) -> String {
    name.strip_prefix("mcp__")
        .unwrap_or(name)
        .replacen("__", ".", 1)
}

fn preview_arguments(raw: &str) -> String {
    serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|value| serde_json::to_string_pretty(&value).ok())
        .unwrap_or_else(|| raw.to_owned())
        .chars()
        .take(800)
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
pub enum RpcCommand {
    Ask { text: String },
    Approve { approval_id: u64 },
    Deny { approval_id: u64 },
    Cancel,
    Status,
    NewSession,
    Unknown { method: String },
}

#[derive(Deserialize)]
struct WireCommand {
    method: String,
    #[serde(default)]
    params: Value,
}

pub fn parse_command(line: &str) -> Result<RpcCommand> {
    let wire: WireCommand = serde_json::from_str(line).context("RPC 命令必须是 JSON 对象")?;
    Ok(match wire.method.as_str() {
        "ask" => RpcCommand::Ask {
            text: wire
                .params
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
        },
        "approve" => RpcCommand::Approve {
            approval_id: wire
                .params
                .get("approval_id")
                .and_then(Value::as_u64)
                .context("approve 需要 approval_id")?,
        },
        "deny" => RpcCommand::Deny {
            approval_id: wire
                .params
                .get("approval_id")
                .and_then(Value::as_u64)
                .context("deny 需要 approval_id")?,
        },
        "cancel" => RpcCommand::Cancel,
        "status" => RpcCommand::Status,
        "new_session" => RpcCommand::NewSession,
        other => RpcCommand::Unknown {
            method: other.to_owned(),
        },
    })
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum ClientEvent {
    Ready {
        model: String,
        rag: String,
        mcp: String,
    },
    User {
        text: String,
    },
    Command {
        text: String,
    },
    CommandResult {
        text: String,
    },
    Exit,
    Status {
        text: String,
    },
    AssistantBegin,
    AssistantDelta {
        text: String,
    },
    ToolStarted {
        name: String,
    },
    ToolResult {
        name: String,
        ok: bool,
        ms: u128,
    },
    Approval {
        id: u64,
        tool: String,
        arguments: String,
    },
    TurnCompleted {
        steps: usize,
        tokens: usize,
        retries: usize,
    },
    TurnCancelled,
    SessionCleared,
    Error {
        message: String,
    },
    Log {
        text: String,
    },
}

struct RpcEventSink {
    sender: mpsc::UnboundedSender<ClientEvent>,
    tool_names: Mutex<HashMap<String, String>>,
    assistant_open: Mutex<bool>,
}

impl RpcEventSink {
    fn new(sender: mpsc::UnboundedSender<ClientEvent>) -> Self {
        Self {
            sender,
            tool_names: Mutex::new(HashMap::new()),
            assistant_open: Mutex::new(false),
        }
    }
}

impl EventSink for RpcEventSink {
    fn publish(&self, event: AgentEvent) {
        match &event {
            AgentEvent::ToolCallStarted { .. }
            | AgentEvent::StepRetrying { .. }
            | AgentEvent::TurnStarted { .. } => {
                if let Ok(mut open) = self.assistant_open.lock() {
                    *open = false;
                }
            }
            _ => {}
        }
        let mapped = match event {
            AgentEvent::AssistantDelta { delta, .. } => {
                if delta.is_empty() {
                    return;
                }
                if let Ok(mut open) = self.assistant_open.lock()
                    && !*open
                {
                    *open = true;
                    let _ = self.sender.send(ClientEvent::AssistantBegin);
                }
                ClientEvent::AssistantDelta { text: delta }
            }
            AgentEvent::ToolCallStarted { call_id, name, .. } => {
                if let Ok(mut names) = self.tool_names.lock() {
                    names.insert(call_id, name.clone());
                }
                ClientEvent::ToolStarted {
                    name: display_tool_name(&name),
                }
            }
            AgentEvent::ToolResult {
                call_id,
                is_error,
                duration_ms,
                ..
            } => {
                let name = self
                    .tool_names
                    .lock()
                    .ok()
                    .and_then(|mut names| names.remove(&call_id))
                    .unwrap_or_else(|| "tool".into());
                ClientEvent::ToolResult {
                    name: display_tool_name(&name),
                    ok: !is_error,
                    ms: duration_ms,
                }
            }
            AgentEvent::TurnFailed { message, .. } => ClientEvent::Error { message },
            AgentEvent::TurnCancelled { .. } => ClientEvent::TurnCancelled,
            AgentEvent::StepRetrying {
                next_attempt,
                max_attempts,
                delay_ms,
                ..
            } => ClientEvent::Log {
                text: format!("重试 {next_attempt}/{max_attempts} · 等待 {delay_ms}ms"),
            },
            _ => return,
        };
        let _ = self.sender.send(mapped);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ask_approve_and_cancel() {
        let ask = parse_command(r#"{"method":"ask","params":{"text":"计算1+1"}}"#).unwrap();
        assert_eq!(
            ask,
            RpcCommand::Ask {
                text: "计算1+1".into()
            }
        );
        let approve = parse_command(r#"{"method":"approve","params":{"approval_id":2}}"#).unwrap();
        assert_eq!(approve, RpcCommand::Approve { approval_id: 2 });
        assert_eq!(
            parse_command(r#"{"method":"cancel"}"#).unwrap(),
            RpcCommand::Cancel
        );
        assert_eq!(
            parse_command(r#"{"method":"new_session"}"#).unwrap(),
            RpcCommand::NewSession
        );
    }

    #[test]
    fn maps_tool_and_delta_events() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let sink = RpcEventSink::new(sender);
        sink.publish(AgentEvent::ToolCallStarted {
            turn_id: 1,
            call_id: "c1".into(),
            name: "calculate".into(),
        });
        sink.publish(AgentEvent::ToolResult {
            turn_id: 1,
            call_id: "c1".into(),
            is_error: false,
            duration_ms: 3,
        });
        sink.publish(AgentEvent::AssistantDelta {
            turn_id: 1,
            step: 1,
            delta: "2".into(),
            offset: 0,
        });
        assert!(matches!(
            receiver.try_recv().unwrap(),
            ClientEvent::ToolStarted { name } if name == "calculate"
        ));
        assert!(matches!(
            receiver.try_recv().unwrap(),
            ClientEvent::ToolResult {
                ok: true,
                ms: 3,
                ..
            }
        ));
        assert!(matches!(
            receiver.try_recv().unwrap(),
            ClientEvent::AssistantBegin
        ));
        assert!(matches!(
            receiver.try_recv().unwrap(),
            ClientEvent::AssistantDelta { text } if text == "2"
        ));
    }
}
