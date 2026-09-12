use crate::{
    context::ContextMemory,
    message::{Message, ToolCall},
};
use anyhow::{Result, bail};
use serde_json::Value;

/// 定义模型窗口、输出预留和触发上下文整理的软阈值。
#[derive(Clone, Copy, Debug)]
pub struct ContextPolicy {
    pub max_context_tokens: usize,
    pub reserved_output_tokens: usize,
    pub compact_at_tokens: usize,
    pub recent_turns_to_keep: usize,
}

impl Default for ContextPolicy {
    /// 为 DeepSeek 百万 Token 窗口提供保守的默认预算。
    fn default() -> Self {
        Self {
            max_context_tokens: 1_000_000,
            reserved_output_tokens: 100_000,
            compact_at_tokens: 800_000,
            recent_turns_to_keep: 20,
        }
    }
}

impl ContextPolicy {
    /// 返回扣除模型输出预留后的最大输入 Token 数。
    pub fn hard_input_limit(&self) -> Result<usize> {
        if self.max_context_tokens <= self.reserved_output_tokens {
            bail!("上下文窗口必须大于输出预留 Token");
        }
        Ok(self.max_context_tokens - self.reserved_output_tokens)
    }

    /// 返回不超过硬限制的上下文整理目标值。
    fn target_input_tokens(&self) -> Result<usize> {
        if self.compact_at_tokens == 0 {
            bail!("上下文整理阈值必须大于 0");
        }
        Ok(self.compact_at_tokens.min(self.hard_input_limit()?))
    }
}

/// 把文本转换成近似 Token 数，使预算算法可以替换成更精确的实现。
pub trait TokenEstimator {
    /// 估算一段文本占用的 Token 数。
    fn estimate_text(&self, text: &str) -> usize;
}

/// 使用中英文字符特征进行保守估算，不依赖特定模型的 tokenizer。
#[derive(Clone, Copy, Debug, Default)]
pub struct HeuristicTokenEstimator;

impl TokenEstimator for HeuristicTokenEstimator {
    /// 将非 ASCII 字符按一个 Token、ASCII 字符按约三个字符一个 Token 估算。
    fn estimate_text(&self, text: &str) -> usize {
        let mut ascii = 0usize;
        let mut non_ascii = 0usize;
        for character in text.chars() {
            if character.is_ascii() {
                ascii += 1;
            } else {
                non_ascii += 1;
            }
        }
        non_ascii + ascii.div_ceil(3)
    }
}

/// 表示经过预算控制、可以安全交给模型适配器的上下文视图。
#[derive(Clone, Debug)]
pub struct PreparedContext {
    system_prompt: String,
    summary: Option<String>,
    messages: Vec<Message>,
    estimated_tokens: usize,
    truncated: bool,
    force_calculation: bool,
}

impl PreparedContext {
    /// 返回本次请求必须携带的系统提示词。
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// 返回本次允许模型看到的消息，不代表完整历史已被删除。
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// 返回被裁剪旧 Turn 的摘要；没有裁剪时返回 None。
    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    /// 返回包含系统提示、工具定义和消息的估算 Token 数。
    pub fn estimated_tokens(&self) -> usize {
        self.estimated_tokens
    }

    /// 表示本次视图是否省略了较早的完整 Turn。
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// 表示当前用户问题是否应强制使用 calculate 工具。
    pub fn force_calculation(&self) -> bool {
        self.force_calculation
    }
}

/// 根据预算从完整 ContextMemory 中构建模型本次可见的上下文。
pub struct ContextBuilder<E> {
    policy: ContextPolicy,
    estimator: E,
}

impl<E: TokenEstimator> ContextBuilder<E> {
    /// 使用指定策略和 Token 估算器创建上下文构建器。
    pub fn new(policy: ContextPolicy, estimator: E) -> Self {
        Self { policy, estimator }
    }

    /// 保留系统提示和最近完整 Turn，并在软阈值前构建预算化视图。
    pub fn prepare(&self, memory: &ContextMemory, tools: &Value) -> Result<PreparedContext> {
        memory.validate()?;
        let hard_limit = self.policy.hard_input_limit()?;
        let target = self.policy.target_input_tokens()?;
        let base_tokens = self.estimate_text(memory.system_prompt())
            + self.estimate_text(&serde_json::to_string(tools)?);
        if base_tokens > hard_limit {
            bail!("系统提示和工具定义已经超过上下文输入上限");
        }

        let ranges = turn_ranges(memory.messages());
        let all_tokens = base_tokens + self.estimate_messages(memory.messages());
        if all_tokens <= target {
            return Ok(PreparedContext {
                system_prompt: memory.system_prompt().to_owned(),
                summary: None,
                messages: memory.messages().to_vec(),
                estimated_tokens: all_tokens,
                truncated: false,
                force_calculation: latest_requires_calculation(memory.messages()),
            });
        }

        let Some(&(current_start, current_end)) = ranges.last() else {
            return Ok(PreparedContext {
                system_prompt: memory.system_prompt().to_owned(),
                summary: None,
                messages: Vec::new(),
                estimated_tokens: base_tokens,
                truncated: false,
                force_calculation: latest_requires_calculation(memory.messages()),
            });
        };
        let current_tokens = self.estimate_messages(&memory.messages()[current_start..current_end]);
        if base_tokens + current_tokens > hard_limit {
            bail!("系统提示、工具定义和当前 Turn 已超过上下文输入上限");
        }

        let mut selected_start = ranges.len() - 1;
        let mut selected_tokens = base_tokens + current_tokens;
        let keep_limit = self.policy.recent_turns_to_keep.max(1);
        while selected_start > 0 && ranges.len() - selected_start < keep_limit {
            let (start, end) = ranges[selected_start - 1];
            let turn_tokens = self.estimate_messages(&memory.messages()[start..end]);
            if selected_tokens + turn_tokens > target {
                break;
            }
            selected_start -= 1;
            selected_tokens += turn_tokens;
        }

        let first_message = ranges[selected_start].0;
        let summary = summarize_messages(&memory.messages()[..first_message]).and_then(|summary| {
            let messages_tokens = self.estimate_messages(&memory.messages()[first_message..]);
            let remaining = hard_limit.saturating_sub(base_tokens + messages_tokens);
            (remaining > 0).then(|| fit_summary(&self.estimator, &summary, remaining))
        });
        let selected_tokens = base_tokens
            + summary
                .as_deref()
                .map_or(0, |text| self.estimate_text(text))
            + self.estimate_messages(&memory.messages()[first_message..]);
        if selected_tokens > hard_limit {
            bail!("摘要和当前上下文超过输入上限");
        }
        Ok(PreparedContext {
            system_prompt: memory.system_prompt().to_owned(),
            summary,
            messages: memory.messages()[first_message..].to_vec(),
            estimated_tokens: selected_tokens,
            truncated: first_message > 0,
            force_calculation: latest_requires_calculation(memory.messages()),
        })
    }

    /// 给文本估算结果至少保留一个 Token，避免空值完全没有协议开销。
    fn estimate_text(&self, text: &str) -> usize {
        self.estimator.estimate_text(text).max(1)
    }

    /// 估算一组强类型消息及其角色、工具调用等协议开销。
    fn estimate_messages(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|message| match message {
                Message::User { content } => 4 + self.estimate_text(content),
                Message::Assistant {
                    content,
                    tool_calls,
                } => {
                    4 + content
                        .as_deref()
                        .map_or(0, |text| self.estimate_text(text))
                        + tool_calls
                            .iter()
                            .map(|call| self.estimate_tool_call(call))
                            .sum::<usize>()
                }
                Message::Tool {
                    call_id, content, ..
                } => 4 + self.estimate_text(call_id) + self.estimate_text(content),
            })
            .sum()
    }

    /// 估算一个工具调用的 ID、名称、参数和协议字段开销。
    fn estimate_tool_call(&self, call: &ToolCall) -> usize {
        8 + self.estimate_text(&call.id)
            + self.estimate_text(&call.name)
            + self.estimate_text(&call.arguments)
    }
}

/// 用轻量规则识别明确的数学请求，避免把普通聊天错误强制成工具调用。
fn latest_requires_calculation(messages: &[Message]) -> bool {
    let Some(user_index) = messages
        .iter()
        .rposition(|message| matches!(message, Message::User { .. }))
    else {
        return false;
    };
    // 工具结果已经回写后，当前 Step 允许模型生成最终文本。
    if messages[user_index + 1..]
        .iter()
        .any(|message| matches!(message, Message::Tool { .. }))
    {
        return false;
    }
    let Message::User { content } = &messages[user_index] else {
        return false;
    };
    // “打印一行文字：计算1+1”是在复述文字，不是在请求计算。
    if (content.contains("打印") || content.contains("输出")) && content.contains("文字") {
        return false;
    }
    let has_math_marker = content.contains("计算")
        || content.contains("算出")
        || content.contains("求")
        || content.contains("calculate");
    let has_operator = content
        .chars()
        .any(|character| "+-*/^×÷=".contains(character));
    let has_number = content.chars().any(|character| character.is_ascii_digit());
    let compact = content
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    let is_bare_expression = has_number
        && has_operator
        && compact
            .chars()
            .all(|character| character.is_ascii_digit() || ".+-*/^()×÷".contains(character));
    (has_math_marker && has_operator && has_number) || is_bare_expression
}

/// 为被省略的旧消息生成不依赖模型的保守摘要，避免摘要过程递归调用模型。
fn summarize_messages(messages: &[Message]) -> Option<String> {
    if messages.is_empty() {
        return None;
    }
    let turn_count = turn_ranges(messages).len();
    let first_user = messages.iter().find_map(|message| match message {
        Message::User { content } => Some(compact_text(content)),
        _ => None,
    });
    let last_assistant = messages.iter().rev().find_map(|message| match message {
        Message::Assistant { content, .. } => content.as_deref().map(compact_text),
        _ => None,
    });
    let mut parts = vec![format!("已省略 {turn_count} 个较早 Turn")];
    if let Some(user) = first_user {
        parts.push(format!("早期用户主题：{user}"));
    }
    if let Some(assistant) = last_assistant {
        parts.push(format!("早期助手结论片段：{assistant}"));
    }
    Some(parts.join("；") + "。如需精确信息，请重新询问。")
}

/// 压缩摘要片段的长度并折叠空白，防止摘要反过来占满上下文预算。
fn compact_text(text: &str) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut compacted = normalized.chars().take(120).collect::<String>();
    if normalized.chars().count() > 120 {
        compacted.push('…');
    }
    compacted
}

/// 将摘要裁剪到剩余预算内，同时保留摘要开头的 Turn 数量信息。
fn fit_summary<E: TokenEstimator>(estimator: &E, summary: &str, max_tokens: usize) -> String {
    if estimator.estimate_text(summary).max(1) <= max_tokens {
        return summary.to_owned();
    }
    let mut chars = summary.chars().collect::<Vec<_>>();
    while !chars.is_empty()
        && estimator.estimate_text(&chars.iter().collect::<String>()) > max_tokens
    {
        let remove_count = (chars.len() / 10).max(1);
        let new_len = chars.len().saturating_sub(remove_count);
        chars.truncate(new_len);
    }
    let fitted = chars.into_iter().collect::<String>();
    if fitted.is_empty() {
        "…".to_owned()
    } else {
        fitted
    }
}

/// 按用户消息边界划分 Turn，保证工具调用和对应结果不会被拆开。
fn turn_ranges(messages: &[Message]) -> Vec<(usize, usize)> {
    if messages.is_empty() {
        return Vec::new();
    }
    let mut starts = vec![0];
    for (index, message) in messages.iter().enumerate().skip(1) {
        if matches!(message, Message::User { .. }) {
            starts.push(index);
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(index, start)| {
            (
                *start,
                starts.get(index + 1).copied().unwrap_or(messages.len()),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// 使用字符数作为 Token 数，让预算测试具有完全确定的结果。
    struct CharacterEstimator;

    impl TokenEstimator for CharacterEstimator {
        /// 返回 Unicode 字符数量作为测试 Token 数。
        fn estimate_text(&self, text: &str) -> usize {
            text.chars().count()
        }
    }

    /// 创建使用小窗口的测试构建器，方便触发上下文整理。
    fn builder(compact_at_tokens: usize) -> ContextBuilder<CharacterEstimator> {
        ContextBuilder::new(
            ContextPolicy {
                max_context_tokens: 100,
                reserved_output_tokens: 10,
                compact_at_tokens,
                recent_turns_to_keep: 20,
            },
            CharacterEstimator,
        )
    }

    /// 向上下文添加一个不包含工具调用的完整 Turn。
    fn append_turn(memory: &mut ContextMemory, user: &str, assistant: &str) {
        memory.append_user(user).unwrap();
        memory
            .append_assistant(Some(assistant.into()), Vec::new())
            .unwrap();
    }

    #[test]
    /// 验证预算充足时不会裁剪任何历史。
    fn keeps_all_messages_when_under_budget() {
        let mut memory = ContextMemory::new("s");
        append_turn(&mut memory, "one", "first");
        let prepared = builder(80).prepare(&memory, &json!([])).unwrap();
        assert_eq!(prepared.messages(), memory.messages());
        assert!(!prepared.truncated());
    }

    #[test]
    /// 验证超过软阈值时只保留最近的完整 Turn。
    fn drops_old_turns_when_over_budget() {
        let mut memory = ContextMemory::new("s");
        append_turn(&mut memory, "old-user", "old-assistant");
        append_turn(&mut memory, "new-user", "new-assistant");
        let prepared = builder(35).prepare(&memory, &json!([])).unwrap();
        assert!(prepared.truncated());
        assert_eq!(prepared.messages(), &memory.messages()[2..]);
        assert!(prepared.summary().is_some());
    }

    #[test]
    /// 验证工具调用和结果在预算裁剪后仍作为同一个 Turn 保留。
    fn keeps_tool_call_and_result_together() {
        let mut memory = ContextMemory::new("s");
        append_turn(&mut memory, "old-user", "old-assistant");
        memory.append_user("calculate").unwrap();
        memory
            .append_assistant(
                None,
                vec![ToolCall {
                    id: "a".into(),
                    name: "calculate".into(),
                    arguments: "{}".into(),
                }],
            )
            .unwrap();
        memory.append_tool_result("a", "2", false).unwrap();
        let prepared = builder(40).prepare(&memory, &json!([])).unwrap();
        assert!(matches!(prepared.messages()[1], Message::Assistant { .. }));
        assert!(matches!(prepared.messages()[2], Message::Tool { .. }));
    }

    #[test]
    /// 验证构建预算视图不会删除 ContextMemory 中的原始消息。
    fn does_not_mutate_raw_memory() {
        let mut memory = ContextMemory::new("s");
        append_turn(&mut memory, "old-user", "old-assistant");
        append_turn(&mut memory, "new-user", "new-assistant");
        let original = memory.messages().to_vec();
        let _ = builder(35).prepare(&memory, &json!([])).unwrap();
        assert_eq!(memory.messages(), original);
    }

    #[test]
    /// 验证当前 Turn 自身超过硬上限时返回明确错误。
    fn rejects_current_turn_over_hard_limit() {
        let mut memory = ContextMemory::new("s");
        append_turn(&mut memory, &"x".repeat(90), "answer");
        assert!(builder(30).prepare(&memory, &json!([])).is_err());
    }

    #[test]
    /// 验证无效的窗口与输出预留配置会被拒绝。
    fn rejects_invalid_policy() {
        let policy = ContextPolicy {
            max_context_tokens: 10,
            reserved_output_tokens: 10,
            compact_at_tokens: 5,
            recent_turns_to_keep: 1,
        };
        assert!(policy.hard_input_limit().is_err());
    }

    #[test]
    /// 验证估算值刚好等于软阈值时不会误触发裁剪。
    fn keeps_context_exactly_at_soft_limit() {
        let mut memory = ContextMemory::new("s");
        append_turn(&mut memory, "a", "b");
        let prepared = builder(13).prepare(&memory, &json!([])).unwrap();
        assert_eq!(prepared.estimated_tokens(), 13);
        assert!(!prepared.truncated());
        assert_eq!(prepared.messages(), memory.messages());
    }

    #[test]
    /// 验证工具 Schema 也计入预算，并能单独触发旧 Turn 裁剪。
    fn counts_tool_schema_when_selecting_history() {
        let mut memory = ContextMemory::new("s");
        append_turn(&mut memory, "a", "b");
        append_turn(&mut memory, "c", "d");
        let context_builder = builder(30);
        let without_large_tools = context_builder.prepare(&memory, &json!([])).unwrap();
        let with_large_tools = context_builder
            .prepare(
                &memory,
                &json!([{"name": "large", "description": "xxxxxxxxxxxxxxxxxxxx"}]),
            )
            .unwrap();
        assert!(!without_large_tools.truncated());
        assert!(with_large_tools.truncated());
        assert_eq!(with_large_tools.messages(), &memory.messages()[2..]);
    }

    #[test]
    /// 验证触发整理后，即使预算仍有空间也遵守最近 Turn 数量上限。
    fn respects_recent_turn_count_limit_after_compaction() {
        let mut memory = ContextMemory::new("s");
        append_turn(&mut memory, "a", "1");
        append_turn(&mut memory, "b", "2");
        append_turn(&mut memory, "c", "3");
        append_turn(&mut memory, "d", "4");
        let context_builder = ContextBuilder::new(
            ContextPolicy {
                max_context_tokens: 100,
                reserved_output_tokens: 10,
                compact_at_tokens: 25,
                recent_turns_to_keep: 2,
            },
            CharacterEstimator,
        );
        let prepared = context_builder.prepare(&memory, &json!([])).unwrap();
        assert!(prepared.truncated());
        assert_eq!(prepared.messages(), &memory.messages()[4..]);
    }

    #[test]
    /// 验证当前 Turn 可超过软阈值，只要仍未超过硬输入上限。
    fn keeps_current_turn_above_soft_limit() {
        let mut memory = ContextMemory::new("s");
        append_turn(&mut memory, &"x".repeat(30), &"y".repeat(10));
        let prepared = builder(20).prepare(&memory, &json!([])).unwrap();
        assert!(prepared.estimated_tokens() > 20);
        assert!(!prepared.truncated());
        assert_eq!(prepared.messages(), memory.messages());
    }

    #[test]
    /// 验证明确数学请求触发代码级 calculate 强制策略。
    fn detects_calculation_request() {
        let mut memory = ContextMemory::new("system");
        memory.append_user("计算 1+1").unwrap();
        let prepared = ContextBuilder::new(ContextPolicy::default(), CharacterEstimator)
            .prepare(&memory, &json!([]))
            .unwrap();
        assert!(prepared.force_calculation());
    }

    #[test]
    /// 普通对话即使包含数字也不应被错误强制调用计算工具。
    fn does_not_force_calculation_for_normal_chat() {
        let mut memory = ContextMemory::new("system");
        memory.append_user("我有 1 个问题，今天心情很好").unwrap();
        let prepared = ContextBuilder::new(ContextPolicy::default(), CharacterEstimator)
            .prepare(&memory, &json!([]))
            .unwrap();
        assert!(!prepared.force_calculation());
    }

    #[test]
    /// 裸算式没有“计算”关键词，也必须触发 calculate。
    fn forces_bare_arithmetic_expression() {
        let mut memory = ContextMemory::new("system");
        memory.append_user("1+1").unwrap();
        let prepared = ContextBuilder::new(ContextPolicy::default(), CharacterEstimator)
            .prepare(&memory, &json!([]))
            .unwrap();
        assert!(prepared.force_calculation());
    }

    #[test]
    /// 打印包含算式的文字时，必须保留文本任务语义，不能误调用 calculate。
    fn does_not_calculate_quoted_text() {
        let mut memory = ContextMemory::new("system");
        memory.append_user("帮我打印一行文字‘计算1+1’").unwrap();
        let prepared = ContextBuilder::new(ContextPolicy::default(), CharacterEstimator)
            .prepare(&memory, &json!([]))
            .unwrap();
        assert!(!prepared.force_calculation());
    }
}
