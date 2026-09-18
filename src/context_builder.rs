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
    forced_tool_name: Option<String>,
    force_calculation: bool,
    force_search_docs: bool,
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

    /// 返回用户明确点名并要求调用的已注册工具。
    pub fn forced_tool_name(&self) -> Option<&str> {
        self.forced_tool_name.as_deref()
    }

    /// 表示当前用户问题是否应强制使用 calculate 工具。
    pub fn force_calculation(&self) -> bool {
        self.force_calculation
    }

    /// 表示当前用户问题是否应强制检索本地文档。
    pub fn force_search_docs(&self) -> bool {
        self.force_search_docs
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
        let messages = visible_messages(memory.messages());
        let forced_tool_name = latest_explicit_tool_request(&messages, tools);
        let force_calculation =
            forced_tool_name.is_none() && latest_requires_calculation(&messages);
        let force_search_docs = forced_tool_name.is_none()
            && !force_calculation
            && latest_requires_search_docs(&messages);
        let base_tokens = self.estimate_text(memory.system_prompt())
            + self.estimate_text(&serde_json::to_string(tools)?);
        if base_tokens > hard_limit {
            bail!("系统提示和工具定义已经超过上下文输入上限");
        }

        let ranges = turn_ranges(&messages);
        let all_tokens = base_tokens + self.estimate_messages(&messages);
        if all_tokens <= target {
            return Ok(PreparedContext {
                system_prompt: memory.system_prompt().to_owned(),
                summary: None,
                messages,
                estimated_tokens: all_tokens,
                truncated: false,
                forced_tool_name,
                force_calculation,
                force_search_docs,
            });
        }

        let Some(&(current_start, current_end)) = ranges.last() else {
            return Ok(PreparedContext {
                system_prompt: memory.system_prompt().to_owned(),
                summary: None,
                messages: Vec::new(),
                estimated_tokens: base_tokens,
                truncated: false,
                forced_tool_name,
                force_calculation,
                force_search_docs,
            });
        };
        let current_tokens = self.estimate_messages(&messages[current_start..current_end]);
        if base_tokens + current_tokens > hard_limit {
            bail!("系统提示、工具定义和当前 Turn 已超过上下文输入上限");
        }

        let mut selected_start = ranges.len() - 1;
        let mut selected_tokens = base_tokens + current_tokens;
        let keep_limit = self.policy.recent_turns_to_keep.max(1);
        while selected_start > 0 && ranges.len() - selected_start < keep_limit {
            let (start, end) = ranges[selected_start - 1];
            let turn_tokens = self.estimate_messages(&messages[start..end]);
            if selected_tokens + turn_tokens > target {
                break;
            }
            selected_start -= 1;
            selected_tokens += turn_tokens;
        }

        let first_message = ranges[selected_start].0;
        let summary = summarize_messages(&messages[..first_message]).and_then(|summary| {
            let messages_tokens = self.estimate_messages(&messages[first_message..]);
            let remaining = hard_limit.saturating_sub(base_tokens + messages_tokens);
            (remaining > 0).then(|| fit_summary(&self.estimator, &summary, remaining))
        });
        let selected_tokens = base_tokens
            + summary
                .as_deref()
                .map_or(0, |text| self.estimate_text(text))
            + self.estimate_messages(&messages[first_message..]);
        if selected_tokens > hard_limit {
            bail!("摘要和当前上下文超过输入上限");
        }
        Ok(PreparedContext {
            system_prompt: memory.system_prompt().to_owned(),
            summary,
            messages: messages[first_message..].to_vec(),
            estimated_tokens: selected_tokens,
            truncated: first_message > 0,
            forced_tool_name,
            force_calculation,
            force_search_docs,
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

/// 发给模型前去掉旧版 RAG 把整份资料塞进用户消息的过期片段。
fn visible_messages(messages: &[Message]) -> Vec<Message> {
    messages
        .iter()
        .map(|message| match message {
            Message::User { content } => Message::User {
                content: unwrap_stale_rag_prompt(content),
            },
            other => other.clone(),
        })
        .collect()
}

fn unwrap_stale_rag_prompt(content: &str) -> String {
    let stale = content.contains("请只根据下面的本地资料回答问题")
        && (content.contains("本地资料：") || content.contains("[资料 "));
    if !stale {
        return content.to_owned();
    }
    content
        .rsplit("\n问题：")
        .next()
        .map(str::trim)
        .filter(|question| !question.is_empty() && *question != content)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| content.to_owned())
}

fn latest_user_index(messages: &[Message]) -> Option<usize> {
    messages
        .iter()
        .rposition(|message| matches!(message, Message::User { .. }))
}

fn latest_user_awaiting_tools(messages: &[Message]) -> Option<&str> {
    let user_index = latest_user_index(messages)?;
    if messages[user_index + 1..]
        .iter()
        .any(|message| matches!(message, Message::Tool { .. }))
    {
        return None;
    }
    match &messages[user_index] {
        Message::User { content } => Some(content.as_str()),
        _ => None,
    }
}

/// 用户明确要求调用一个真实存在的工具时，返回该工具名以消除 auto 选择歧义。
fn latest_explicit_tool_request(messages: &[Message], tools: &Value) -> Option<String> {
    let content = latest_user_awaiting_tools(messages)?;
    let lower = content.to_ascii_lowercase();
    let requests_call = content.contains("请调用")
        || content.contains("必须调用")
        || content.contains("帮我调用")
        || content.contains("调用工具")
        || content.contains("请使用")
        || content.contains("必须使用")
        || content.contains("使用工具")
        || lower.contains("call tool")
        || lower.contains("use tool");
    let normalized = content.replace('`', "");
    let trimmed = normalized.trim_start();
    tools
        .as_array()?
        .iter()
        .filter_map(|tool| tool["function"]["name"].as_str())
        .filter(|name| {
            normalized.contains(name)
                && (requests_call
                    || ["调用", "使用"].iter().any(|action| {
                        trimmed
                            .strip_prefix(action)
                            .is_some_and(|rest| rest.trim_start().starts_with(name))
                    }))
        })
        .max_by_key(|name| name.len())
        .map(ToOwned::to_owned)
}

/// 章节、目录和明确本地手册问题必须先检索，避免模型凭历史或文件名猜测。
fn latest_requires_search_docs(messages: &[Message]) -> bool {
    let Some(content) = latest_user_awaiting_tools(messages) else {
        return false;
    };
    let lower = content.to_ascii_lowercase();
    crate::rag::requested_chapter_number(content).is_some()
        || crate::rag::is_outline_query(content)
        || [
            "知识库",
            "本地资料",
            "故障手册",
            "troubleshooting",
            "trouble shooting",
        ]
        .iter()
        .any(|keyword| lower.contains(keyword))
}

/// 用轻量规则识别明确的数学请求，避免把普通聊天错误强制成工具调用。
fn latest_requires_calculation(messages: &[Message]) -> bool {
    let Some(content) = latest_user_awaiting_tools(messages) else {
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
    fn forces_explicitly_named_registered_tool() {
        let mut memory = ContextMemory::new("system");
        memory
            .append_user("必须调用 mcp__fixture__echo，参数 text 为 ASTERIA-MCP-001，并告诉我结果")
            .unwrap();
        let tools = json!([
            {"type":"function","function":{"name":"calculate","parameters":{"type":"object"}}},
            {"type":"function","function":{"name":"mcp__fixture__echo","parameters":{"type":"object"}}}
        ]);
        let prepared = ContextBuilder::new(ContextPolicy::default(), CharacterEstimator)
            .prepare(&memory, &tools)
            .unwrap();
        assert_eq!(prepared.forced_tool_name(), Some("mcp__fixture__echo"));
        assert!(!prepared.force_calculation());
        assert!(!prepared.force_search_docs());
    }

    #[test]
    fn forces_named_tool_for_direct_chinese_call_command() {
        let mut memory = ContextMemory::new("system");
        memory
            .append_user("调用 mcp__fixture__echo，参数 text 为 MCP-RECONNECT-002")
            .unwrap();
        let tools = json!([
            {"type":"function","function":{"name":"mcp__fixture__echo","parameters":{"type":"object"}}}
        ]);
        let prepared = ContextBuilder::new(ContextPolicy::default(), CharacterEstimator)
            .prepare(&memory, &tools)
            .unwrap();
        assert_eq!(prepared.forced_tool_name(), Some("mcp__fixture__echo"));
    }

    #[test]
    fn does_not_force_unregistered_named_tool() {
        let mut memory = ContextMemory::new("system");
        memory.append_user("必须调用 missing_tool").unwrap();
        let prepared = ContextBuilder::new(ContextPolicy::default(), CharacterEstimator)
            .prepare(&memory, &json!([]))
            .unwrap();
        assert_eq!(prepared.forced_tool_name(), None);
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

    #[test]
    /// 旧版把资料塞进用户消息的片段不能继续发给模型，章节问题要强制检索。
    fn unwraps_stale_rag_prompt_and_forces_search_docs() {
        let mut memory = ContextMemory::new("s");
        memory
            .append_user(
                "请只根据下面的本地资料回答问题；资料不足时明确说不知道。\n\n本地资料：\n[资料 1]\nPAGEREF toc\n\n问题：这篇文档主要讲什么？",
            )
            .unwrap();
        memory
            .append_assistant(Some("只有目录".into()), Vec::new())
            .unwrap();
        memory.append_user("第14章节写的什么内容").unwrap();
        let prepared = ContextBuilder::new(ContextPolicy::default(), CharacterEstimator)
            .prepare(&memory, &json!([]))
            .unwrap();
        assert!(prepared.force_search_docs());
        assert!(!prepared.force_calculation());
        let Message::User { content } = &prepared.messages()[0] else {
            panic!("expected user message");
        };
        assert_eq!(content, "这篇文档主要讲什么？");
        assert!(!content.contains("PAGEREF"));
        let Message::User { content } = prepared.messages().last().unwrap() else {
            panic!("expected user message");
        };
        assert_eq!(content, "第14章节写的什么内容");
    }

    #[test]
    fn troubleshooting_manual_question_forces_fresh_search() {
        let mut memory = ContextMemory::new("s");
        memory
            .append_user("查看 trouble shooting 文档，LT 反复重启时如何 debug？")
            .unwrap();
        let prepared = ContextBuilder::new(ContextPolicy::default(), CharacterEstimator)
            .prepare(&memory, &json!([]))
            .unwrap();
        assert!(prepared.force_search_docs());
        assert!(!prepared.force_calculation());
    }

    #[test]
    /// 当前 Turn 的检索结果必须保留到模型生成最终回答，且不能再次强制检索。
    fn keeps_current_turn_search_result_for_answer_step() {
        let mut memory = ContextMemory::new("s");
        memory.append_user("第14章节写的什么内容").unwrap();
        memory
            .append_assistant(
                None,
                vec![ToolCall {
                    id: "rag-1".into(),
                    name: "search_docs".into(),
                    arguments: r#"{"query":"第14章节写的什么内容"}"#.into(),
                }],
            )
            .unwrap();
        memory
            .append_tool_result("rag-1", "第14章正文", false)
            .unwrap();

        let prepared = ContextBuilder::new(ContextPolicy::default(), CharacterEstimator)
            .prepare(&memory, &json!([]))
            .unwrap();
        assert_eq!(prepared.messages(), memory.messages());
        assert!(!prepared.force_search_docs());
        assert!(matches!(
            prepared.messages().last(),
            Some(Message::Tool { content, .. }) if content == "第14章正文"
        ));
    }
}
