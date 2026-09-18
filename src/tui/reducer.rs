use super::{render, state::TuiState};
use asteria_agent::events::AgentEvent;

/// 把细粒度 AgentEvent 归约为少量用户可见输出。
pub fn reduce(state: &mut TuiState, event: &AgentEvent) -> Vec<String> {
    let mut lines = match event {
        AgentEvent::TurnStarted { turn_id, .. } => {
            state.begin_turn(*turn_id);
            Vec::new()
        }
        AgentEvent::AssistantDelta { delta, offset, .. } => state.live.append(delta, *offset),
        AgentEvent::ToolCallStarted { call_id, name, .. } => {
            let flushed = state.live.flush(true);
            state.live.begin_generation();
            state.tool_names.insert(call_id.clone(), name.clone());
            flushed
        }
        AgentEvent::StepCompleted { .. } => state.live.flush(true),
        AgentEvent::StepRetrying { .. } => {
            let flushed = state.live.flush(true);
            state.live.begin_generation();
            flushed
        }
        AgentEvent::TurnCompleted { turn_id, .. } => {
            let flushed = state.live.flush(true);
            state.complete_turn(*turn_id);
            flushed
        }
        AgentEvent::TurnCancelled { turn_id } | AgentEvent::TurnFailed { turn_id, .. } => {
            let flushed = state.live.flush(true);
            state.clear_turn(*turn_id);
            flushed
        }
        _ => Vec::new(),
    };

    if state.verbose && !matches!(event, AgentEvent::AssistantDelta { .. }) {
        lines.push(render::verbose_event(event));
        return lines;
    }

    match event {
        AgentEvent::ToolResult {
            call_id,
            is_error,
            duration_ms,
            ..
        } => {
            let name = state
                .tool_names
                .remove(call_id)
                .unwrap_or_else(|| "unknown_tool".into());
            lines.push(render::compact_tool_result(&name, *is_error, *duration_ms));
        }
        AgentEvent::StepRetrying {
            step,
            next_attempt,
            max_attempts,
            delay_ms,
            ..
        } => lines.push(render::compact_retry(
            *step,
            *next_attempt,
            *max_attempts,
            *delay_ms,
        )),
        _ => {}
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use asteria_agent::provider::TokenUsage;

    #[test]
    fn compact_view_collapses_tool_events_and_hides_step_noise() {
        let mut state = TuiState::default();
        assert!(
            reduce(
                &mut state,
                &AgentEvent::StepStarted {
                    turn_id: 1,
                    step: 1
                }
            )
            .is_empty()
        );
        assert!(
            reduce(
                &mut state,
                &AgentEvent::ToolCallStarted {
                    turn_id: 1,
                    call_id: "call-1".into(),
                    name: "mcp__github__get_file_contents".into(),
                }
            )
            .is_empty()
        );
        assert!(
            reduce(
                &mut state,
                &AgentEvent::PermissionRequested {
                    turn_id: 1,
                    call_id: "call-1".into(),
                    name: "mcp__github__get_file_contents".into(),
                }
            )
            .is_empty()
        );
        assert_eq!(
            reduce(
                &mut state,
                &AgentEvent::ToolResult {
                    turn_id: 1,
                    call_id: "call-1".into(),
                    is_error: false,
                    duration_ms: 5_400,
                }
            ),
            vec!["  ✓ github.get_file_contents · 完成 · 5.4s"]
        );
        assert!(
            reduce(
                &mut state,
                &AgentEvent::StepCompleted {
                    turn_id: 1,
                    step: 1,
                    usage: TokenUsage::default(),
                }
            )
            .is_empty()
        );
    }

    #[test]
    fn verbose_view_keeps_detailed_events() {
        let mut state = TuiState::default();
        state.verbose = true;
        let output = reduce(
            &mut state,
            &AgentEvent::StepStarted {
                turn_id: 2,
                step: 3,
            },
        );
        assert_eq!(output, vec!["[执行] Turn 2 · Step 3\n动作：请求模型"]);
    }

    #[test]
    fn live_answer_coalesces_deltas_and_prints_header_once() {
        let mut state = TuiState::default();
        reduce(
            &mut state,
            &AgentEvent::TurnStarted {
                turn_id: 3,
                input: "hi".into(),
            },
        );
        assert!(
            reduce(
                &mut state,
                &AgentEvent::AssistantDelta {
                    turn_id: 3,
                    step: 1,
                    delta: "你".into(),
                    offset: 0,
                }
            )
            .is_empty()
        );
        assert_eq!(
            reduce(
                &mut state,
                &AgentEvent::AssistantDelta {
                    turn_id: 3,
                    step: 1,
                    delta: "好\n世界".into(),
                    offset: "你".len(),
                }
            ),
            vec!["Asteria".to_string(), "你好".into()]
        );
        assert_eq!(
            reduce(
                &mut state,
                &AgentEvent::StepCompleted {
                    turn_id: 3,
                    step: 1,
                    usage: TokenUsage::default(),
                }
            ),
            vec!["世界".to_string()]
        );
        assert!(state.live.streamed);
        assert!(state.take_answer_streamed());
        assert!(!state.take_answer_streamed());
    }

    #[test]
    fn live_answer_ignores_stale_offsets() {
        let mut state = TuiState::default();
        reduce(
            &mut state,
            &AgentEvent::AssistantDelta {
                turn_id: 1,
                step: 1,
                delta: "abc".into(),
                offset: 0,
            },
        );
        assert!(
            reduce(
                &mut state,
                &AgentEvent::AssistantDelta {
                    turn_id: 1,
                    step: 1,
                    delta: "xx".into(),
                    offset: 1,
                }
            )
            .is_empty()
        );
        assert_eq!(
            reduce(
                &mut state,
                &AgentEvent::StepCompleted {
                    turn_id: 1,
                    step: 1,
                    usage: TokenUsage::default(),
                }
            ),
            vec!["Asteria".to_string(), "abc".into()]
        );
    }
}
