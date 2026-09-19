use asteria_agent::agent::Asteria;

/// 与 TUI 相同的斜杠命令分类；未知 `/foo` 视为普通问题，交给模型。
#[derive(Debug, PartialEq, Eq)]
pub enum SlashAction {
    Context,
    Usage,
    Session,
    Trace { command: String },
    Mcp { command: String },
    Permissions,
    Permission { name: String, value: String },
    PermissionUsage,
    Verbose { command: String },
    Reset,
    NewSession,
    Cancel,
    Exit,
    Approve { id: u64 },
    Deny { id: u64 },
    ApprovalUsage,
}

pub fn classify_slash(input: &str) -> Option<SlashAction> {
    let text = input.trim();
    if !text.starts_with('/') {
        return None;
    }
    let parts = text.split_whitespace().collect::<Vec<_>>();
    Some(match parts.as_slice() {
        ["/exit"] => SlashAction::Exit,
        ["/cancel"] => SlashAction::Cancel,
        ["/context"] => SlashAction::Context,
        ["/usage"] => SlashAction::Usage,
        ["/session"] => SlashAction::Session,
        ["/new-session"] => SlashAction::NewSession,
        ["/reset"] => SlashAction::Reset,
        ["/permissions"] => SlashAction::Permissions,
        ["/permission", name, value] => SlashAction::Permission {
            name: (*name).to_owned(),
            value: (*value).to_owned(),
        },
        ["/permissions" | "/permission", ..] => SlashAction::PermissionUsage,
        command if command.first() == Some(&"/mcp") => SlashAction::Mcp {
            command: text.to_owned(),
        },
        command if command.first() == Some(&"/trace") => SlashAction::Trace {
            command: text.to_owned(),
        },
        command if command.first() == Some(&"/verbose") => SlashAction::Verbose {
            command: text.to_owned(),
        },
        ["/approve", id] => match id.parse() {
            Ok(id) => SlashAction::Approve { id },
            Err(_) => SlashAction::ApprovalUsage,
        },
        ["/deny", id] => match id.parse() {
            Ok(id) => SlashAction::Deny { id },
            Err(_) => SlashAction::ApprovalUsage,
        },
        ["/approve" | "/deny", ..] => SlashAction::ApprovalUsage,
        _ => return None,
    })
}

pub fn format_context(agent: &Asteria) -> String {
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
    lines.join("\n")
}

pub fn format_usage(agent: &Asteria) -> String {
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
    lines.join("\n")
}

pub fn format_session(agent: &Asteria) -> String {
    let paths = agent.session_paths();
    format!(
        "[Session]\n目录：{}\n上下文：{}\n审计：{}\n状态：{}",
        paths.directory.display(),
        paths.context.display(),
        paths.trace.display(),
        paths.state.display()
    )
}

pub fn format_permissions(agent: &Asteria) -> String {
    agent
        .tool_permissions()
        .into_iter()
        .map(|(name, permission)| format!("{name}: {permission}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn set_permission(agent: &mut Asteria, name: &str, value: &str) -> String {
    match value
        .parse()
        .and_then(|permission| agent.set_tool_permission(name, permission))
    {
        Ok(()) => format!("[权限] {name} = {value}（当前会话；/reset 保留）"),
        Err(error) => format!("权限设置失败: {error}"),
    }
}

pub fn permission_usage() -> &'static str {
    "用法：/permissions 或 /permission <工具名> allow|deny|ask"
}

pub fn approval_usage() -> &'static str {
    "用法：/approve <编号> 或 /deny <编号>"
}

pub fn format_verbose(command: &str, current: Option<bool>) -> String {
    let parts = command.split_whitespace().collect::<Vec<_>>();
    match (parts.as_slice(), current) {
        (["/verbose"], Some(verbose)) => format!(
            "[界面] 当前为{}模式。用法：/verbose on|off",
            if verbose { "详细" } else { "紧凑" }
        ),
        (["/verbose"], None) => {
            "独立窗口始终显示对话事件。用法：/verbose on|off（仅 TUI 切换紧凑/详细）。".into()
        }
        (["/verbose", "on" | "off"], None) => {
            "独立窗口没有紧凑/详细切换，对话事件会直接显示。".into()
        }
        (["/verbose", "on" | "off"], Some(_)) => String::new(),
        _ => "用法：/verbose on|off".into(),
    }
}

pub fn format_trace(agent: &Asteria, command: &str) -> String {
    let parts: Vec<_> = command.split_whitespace().collect();
    let turn_id = match parts.as_slice() {
        ["/trace"] => agent.last_turn().map(|turn| turn.id),
        ["/trace", value] => match value.parse::<u64>() {
            Ok(turn_id) => Some(turn_id),
            Err(_) => return "用法：/trace 或 /trace <turn_id>".into(),
        },
        _ => return "用法：/trace 或 /trace <turn_id>".into(),
    };
    let Some(turn_id) = turn_id else {
        return "[ToolTrace] 尚未执行 Turn。".into();
    };
    let traces = match agent.tool_traces(Some(turn_id)) {
        Ok(traces) => traces,
        Err(error) => return format!("读取 ToolTrace 失败: {error:#}"),
    };
    if traces.is_empty() {
        return format!("[ToolTrace] Turn {turn_id} 没有工具调用记录。");
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
    lines.join("\n")
}

pub fn format_mcp(agent: &Asteria) -> String {
    let Some(manager) = agent.mcp_manager() else {
        return "[MCP] 尚未加载。".into();
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
    lines.push(mcp_source_line(
        "用户配置",
        manager.global_config(),
        manager.loaded_files(),
        manager.skipped_project_file(),
    ));
    lines.push(mcp_source_line(
        "项目配置",
        manager.project_config(),
        manager.loaded_files(),
        manager.skipped_project_file(),
    ));
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
    lines.join("\n")
}

pub fn mcp_source_line(
    label: &str,
    path: Option<&std::path::Path>,
    loaded: &[std::path::PathBuf],
    skipped: Option<&std::path::Path>,
) -> String {
    let Some(path) = path else {
        return format!("{label}：未设置");
    };
    let state = if loaded.iter().any(|loaded| loaded == path) {
        "已加载"
    } else if skipped == Some(path) {
        "已跳过，输入 /mcp trust 启用"
    } else if path.is_file() {
        "存在但未加载"
    } else {
        "未找到"
    };
    format!("{label}：{}（{state}）", path.display())
}

pub async fn run_mcp_command(
    agent: &mut Asteria,
    command: &str,
    trust_project: &mut bool,
) -> String {
    let parts = command.split_whitespace().collect::<Vec<_>>();
    match parts.as_slice() {
        ["/mcp"] | ["/mcp", "status"] => format_mcp(agent),
        ["/mcp", "tools"] => {
            let Some(manager) = agent.mcp_manager() else {
                return "[MCP] 尚未加载。".into();
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
            if tools.is_empty() {
                "[MCP] 当前没有已注册工具。".into()
            } else {
                format!("[MCP] 已注册工具\n{}", tools.join("\n"))
            }
        }
        ["/mcp", "trust"] => {
            *trust_project = true;
            match agent.reload_mcp(true).await {
                Ok(()) => format!("[MCP] 已信任并加载项目配置。\n{}", format_mcp(agent)),
                Err(error) => format!("[MCP] 启用项目配置失败：{error:#}"),
            }
        }
        ["/mcp", "untrust"] => {
            *trust_project = false;
            match agent.reload_mcp(false).await {
                Ok(()) => format!("[MCP] 已跳过项目配置。\n{}", format_mcp(agent)),
                Err(error) => format!("[MCP] 取消项目配置失败：{error:#}"),
            }
        }
        ["/mcp", "reload"] => match agent.reload_mcp(*trust_project).await {
            Ok(()) => format!("[MCP] 配置已原子重载。\n{}", format_mcp(agent)),
            Err(error) => format!("[MCP] 重载失败：{error:#}"),
        },
        ["/mcp", "connect", server] => {
            match agent.connect_mcp_server(*trust_project, server).await {
                Ok(()) => format!("[MCP] Server `{server}` 已连接。\n{}", format_mcp(agent)),
                Err(error) => format!("[MCP] 连接失败：{error:#}"),
            }
        }
        ["/mcp", "auth", server] => match agent.authorize_mcp_server(*trust_project, server).await {
            Ok(()) => format!("[MCP] Server `{server}` 已授权并连接。\n{}", format_mcp(agent)),
            Err(error) => format!("[MCP] 授权失败：{error:#}"),
        },
        ["/mcp", "disconnect", server] => match agent.disconnect_mcp_server(server).await {
            Ok(()) => format!("[MCP] Server `{server}` 已断开。\n{}", format_mcp(agent)),
            Err(error) => format!("[MCP] 断开失败：{error:#}"),
        },
        ["/mcp", "logout", server] => match agent.logout_mcp_server(server).await {
            Ok(()) => format!("[MCP] Server `{server}` 已退出登录。\n{}", format_mcp(agent)),
            Err(error) => format!("[MCP] 退出登录失败：{error:#}"),
        },
        _ => "用法：/mcp [status|tools|trust|untrust|reload|connect <server>|disconnect <server>|auth <server>|logout <server>]"
            .into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_tui_slash_commands() {
        assert_eq!(classify_slash("/reset"), Some(SlashAction::Reset));
        assert_eq!(classify_slash("/context"), Some(SlashAction::Context));
        assert_eq!(
            classify_slash("/trace 3"),
            Some(SlashAction::Trace {
                command: "/trace 3".into()
            })
        );
        assert_eq!(
            classify_slash("/approve 7"),
            Some(SlashAction::Approve { id: 7 })
        );
        assert_eq!(classify_slash("计算1+1"), None);
        assert_eq!(classify_slash("/unknown"), None);
    }
}
