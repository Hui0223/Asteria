use asteria_agent::events::AgentEvent;

pub fn compact_tool_result(name: &str, is_error: bool, duration_ms: u128) -> String {
    format!(
        "  {} {} · {} · {}",
        if is_error { "×" } else { "✓" },
        display_tool_name(name),
        if is_error { "失败" } else { "完成" },
        format_duration_ms(duration_ms)
    )
}

pub fn compact_retry(
    step: usize,
    next_attempt: usize,
    max_attempts: usize,
    delay_ms: u128,
) -> String {
    format!(
        "  ↻ Step {step} 重试 {next_attempt}/{max_attempts} · 等待 {}",
        format_duration_ms(delay_ms)
    )
}

pub fn verbose_event(event: &AgentEvent) -> String {
    match event {
        AgentEvent::TurnStarted { turn_id, input } => {
            format!("[执行] Turn {turn_id} 开始\n用户：{}", preview(input))
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
            if *is_error { "失败" } else { "成功" },
            duration_ms
        ),
        AgentEvent::PermissionRequested {
            turn_id,
            call_id,
            name,
        } => format!("[权限] Turn {turn_id} 等待确认\n工具：{name}\n调用：{call_id}"),
        AgentEvent::PermissionResolved {
            turn_id,
            call_id,
            allowed,
        } => format!(
            "[权限] Turn {turn_id} {}\n调用：{call_id}",
            if *allowed { "已批准" } else { "已拒绝" }
        ),
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
        AgentEvent::TurnCancelled { turn_id } => {
            format!("[取消] Turn {turn_id}\n状态：已取消")
        }
        AgentEvent::TurnFailed { turn_id, message } => {
            format!("[失败] Turn {turn_id}\n原因：{message}")
        }
        AgentEvent::AssistantDelta {
            turn_id,
            step,
            delta,
            ..
        } => format!("[流式] Turn {turn_id} · Step {step}\n{}", preview(delta)),
    }
}

pub fn turn_summary(
    steps: usize,
    retries: usize,
    total_tokens: usize,
    elapsed: Option<std::time::Duration>,
) -> String {
    let mut parts = vec![
        format!("{steps} step{}", if steps == 1 { "" } else { "s" }),
        format_token_count(total_tokens),
    ];
    if retries > 0 {
        parts.push(format!("{retries} 次重试"));
    }
    if let Some(elapsed) = elapsed {
        parts.push(format_duration_ms(elapsed.as_millis()));
    }
    parts.join(" · ")
}

fn display_tool_name(name: &str) -> String {
    name.strip_prefix("mcp__")
        .unwrap_or(name)
        .replacen("__", ".", 1)
}

fn format_token_count(tokens: usize) -> String {
    if tokens >= 1_000 {
        format!("{:.1}k tokens", tokens as f64 / 1_000.0)
    } else {
        format!("{tokens} tokens")
    }
}

fn format_duration_ms(duration_ms: u128) -> String {
    if duration_ms >= 1_000 {
        format!("{:.1}s", duration_ms as f64 / 1_000.0)
    } else {
        format!("{duration_ms}ms")
    }
}

fn preview(input: &str) -> String {
    let compact = input.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut result = compact.chars().take(120).collect::<String>();
    if compact.chars().count() > 120 {
        result.push('…');
    }
    result
}
