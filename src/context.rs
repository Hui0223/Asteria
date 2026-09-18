use crate::message::{Message, ToolCall};
use anyhow::{Result, bail};
use std::collections::HashSet;

/// 保存与模型供应商无关的对话历史，并维护消息顺序约束。
pub struct ContextMemory {
    system_prompt: String,
    messages: Vec<Message>,
}

impl ContextMemory {
    /// 从已重放的消息创建上下文，并验证工具消息仍然完整配对。
    pub fn restore(system_prompt: impl Into<String>, messages: Vec<Message>) -> Result<Self> {
        let context = Self {
            system_prompt: system_prompt.into(),
            messages,
        };
        context.validate()?;
        Ok(context)
    }
    /// 创建一份空的上下文记忆，并保存不会随重置丢失的系统提示词。
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            messages: Vec::new(),
        }
    }

    /// 返回当前消息数量，作为稍后回滚时使用的检查点。
    pub fn checkpoint(&self) -> usize {
        self.messages.len()
    }

    /// 删除检查点之后的所有消息，让上下文恢复到此前状态。
    pub fn rollback(&mut self, checkpoint: usize) {
        self.messages.truncate(checkpoint);
    }

    /// 删除指定检查点后的某类工具交换，保留用户问题、其他工具和最终回答。
    ///
    /// 返回被删除的工具结果正文，供调用方提取短来源引用并写入独立审计事件。
    pub fn compact_tool_exchanges_since(
        &mut self,
        checkpoint: usize,
        tool_name: &str,
    ) -> Result<Vec<String>> {
        self.compact_tool_exchanges_matching_since(checkpoint, |name| name == tool_name)
    }

    /// 删除指定检查点后名称带给定前缀的工具交换。
    pub fn compact_tool_exchanges_with_prefix_since(
        &mut self,
        checkpoint: usize,
        tool_prefix: &str,
    ) -> Result<Vec<String>> {
        self.compact_tool_exchanges_matching_since(checkpoint, |name| name.starts_with(tool_prefix))
    }

    fn compact_tool_exchanges_matching_since(
        &mut self,
        checkpoint: usize,
        matches_tool: impl Fn(&str) -> bool,
    ) -> Result<Vec<String>> {
        if checkpoint > self.messages.len() {
            bail!("上下文压缩检查点越界");
        }
        self.validate()?;
        let call_ids: HashSet<String> = self.messages[checkpoint..]
            .iter()
            .filter_map(|message| match message {
                Message::Assistant { tool_calls, .. } => Some(tool_calls),
                _ => None,
            })
            .flatten()
            .filter(|call| matches_tool(&call.name))
            .map(|call| call.id.clone())
            .collect();
        if call_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut removed_results = Vec::new();
        let mut compacted = Vec::with_capacity(self.messages.len());
        compacted.extend_from_slice(&self.messages[..checkpoint]);
        for message in &self.messages[checkpoint..] {
            match message {
                Message::Assistant {
                    content,
                    tool_calls,
                } => {
                    let retained_calls: Vec<_> = tool_calls
                        .iter()
                        .filter(|call| !call_ids.contains(&call.id))
                        .cloned()
                        .collect();
                    if content
                        .as_deref()
                        .is_some_and(|text| !text.trim().is_empty())
                        || !retained_calls.is_empty()
                    {
                        compacted.push(Message::Assistant {
                            content: content.clone(),
                            tool_calls: retained_calls,
                        });
                    }
                }
                Message::Tool {
                    call_id, content, ..
                } if call_ids.contains(call_id) => removed_results.push(content.clone()),
                other => compacted.push(other.clone()),
            }
        }
        let previous = std::mem::replace(&mut self.messages, compacted);
        if let Err(error) = self.validate() {
            self.messages = previous;
            return Err(error);
        }
        Ok(removed_results)
    }

    /// 清空对话消息，但保留系统提示词。
    pub fn reset(&mut self) {
        self.messages.clear();
    }

    /// 返回只读的系统提示词，供模型适配器组装请求。
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// 更新系统提示词，用于在运行时启用知识库等可选能力。
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = prompt.into();
    }

    /// 返回只读的结构化消息，避免外部绕过校验直接修改历史。
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// 检查当前历史是否完整，尤其是所有工具调用是否已有结果。
    pub fn validate(&self) -> Result<()> {
        self.ensure_no_pending_tools()
    }

    /// 追加用户消息；若上一批工具调用尚未完成，则拒绝写入。
    pub fn append_user(&mut self, content: impl Into<String>) -> Result<()> {
        self.ensure_no_pending_tools()?;
        self.messages.push(Message::User {
            content: content.into(),
        });
        Ok(())
    }

    /// 追加助手消息，并检查正文或工具调用至少存在一种。
    pub fn append_assistant(
        &mut self,
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Result<()> {
        self.ensure_no_pending_tools()?;
        if content.as_deref().is_none_or(|text| text.trim().is_empty()) && tool_calls.is_empty() {
            bail!("助手响应既没有正文，也没有工具调用");
        }
        self.messages.push(Message::Assistant {
            content,
            tool_calls,
        });
        Ok(())
    }

    /// 追加工具结果，并确保它只匹配一个仍在等待中的调用 ID。
    pub fn append_tool_result(
        &mut self,
        call_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Result<()> {
        let call_id = call_id.into();
        if !self.pending_tool_ids()?.contains(&call_id) {
            bail!("工具结果没有匹配的待处理调用: {call_id}");
        }
        self.messages.push(Message::Tool {
            call_id,
            content: content.into(),
            is_error,
        });
        Ok(())
    }

    /// 拒绝在工具调用结果不完整时开始下一段对话或发送模型请求。
    fn ensure_no_pending_tools(&self) -> Result<()> {
        let pending = self.pending_tool_ids()?;
        if !pending.is_empty() {
            bail!(
                "仍有未完成的工具调用: {}",
                pending.into_iter().collect::<Vec<_>>().join(", ")
            );
        }
        Ok(())
    }

    /// 扫描历史并返回尚未收到结果的工具调用 ID，同时检查配对顺序。
    fn pending_tool_ids(&self) -> Result<HashSet<String>> {
        let mut pending = HashSet::new();
        for message in &self.messages {
            match message {
                Message::User { .. } if !pending.is_empty() => {
                    bail!("工具结果必须紧跟对应的助手工具调用")
                }
                Message::Assistant { tool_calls, .. } => {
                    if !pending.is_empty() {
                        bail!("上一批工具调用尚未完成")
                    }
                    for call in tool_calls {
                        if call.id.is_empty()
                            || call.name.is_empty()
                            || !pending.insert(call.id.clone())
                        {
                            bail!("工具调用 ID 或名称无效")
                        }
                    }
                }
                Message::Tool { call_id, .. } if !pending.remove(call_id) => {
                    bail!("孤立或重复的工具结果: {call_id}")
                }
                _ => {}
            }
        }
        Ok(pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造测试使用的计算器工具调用。
    fn call(id: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "calculate".into(),
            arguments: "{\"expression\":\"1+1\"}".into(),
        }
    }

    #[test]
    /// 验证重置只删除消息，不删除系统提示词。
    fn reset_keeps_system_prompt() {
        let mut context = ContextMemory::new("system");
        context.append_user("hello").unwrap();
        context.reset();
        assert_eq!(context.system_prompt(), "system");
        assert!(context.messages().is_empty());
    }

    #[test]
    /// 运行时更新系统提示词后，后续请求应看到新设定。
    fn set_system_prompt_replaces_instructions() {
        let mut context = ContextMemory::new("system");
        context.set_system_prompt("system\nuse search_docs");
        assert_eq!(context.system_prompt(), "system\nuse search_docs");
    }

    #[test]
    /// 验证回滚会准确删除当前轮新增的消息。
    fn rollback_removes_the_current_turn() {
        let mut context = ContextMemory::new("system");
        let checkpoint = context.checkpoint();
        context.append_user("temporary").unwrap();
        context.rollback(checkpoint);
        assert!(context.messages().is_empty());
    }

    #[test]
    /// 验证孤立或重复的工具结果会被拒绝。
    fn rejects_orphan_and_duplicate_tool_results() {
        let mut context = ContextMemory::new("system");
        assert!(context.append_tool_result("missing", "x", true).is_err());
        context.append_assistant(None, vec![call("a")]).unwrap();
        context.append_tool_result("a", "2", false).unwrap();
        assert!(context.append_tool_result("a", "2", false).is_err());
    }

    #[test]
    /// 验证并行工具调用可以乱序返回，但必须全部完成。
    fn parallel_results_match_by_call_id() {
        let mut context = ContextMemory::new("system");
        context
            .append_assistant(None, vec![call("a"), call("b")])
            .unwrap();
        context.append_tool_result("b", "4", false).unwrap();
        assert!(context.validate().is_err());
        context.append_tool_result("a", "2", false).unwrap();
        assert!(context.validate().is_ok());
    }

    #[test]
    /// RAG 证据只在当前 Turn 使用，完成后应只保留问题和最终回答。
    fn compacts_search_docs_exchange_after_turn() {
        let mut context = ContextMemory::new("system");
        context.append_user("第14章是什么").unwrap();
        context
            .append_assistant(
                None,
                vec![ToolCall {
                    id: "rag-1".into(),
                    name: "search_docs".into(),
                    arguments: r#"{"query":"第14章是什么"}"#.into(),
                }],
            )
            .unwrap();
        context
            .append_tool_result(
                "rag-1",
                "[资料 1 | guide.docx | 片段 14 | score=100]\n很长的正文",
                false,
            )
            .unwrap();
        context
            .append_assistant(Some("第14章介绍持久化配置。".into()), Vec::new())
            .unwrap();

        let removed = context
            .compact_tool_exchanges_since(0, "search_docs")
            .unwrap();
        assert_eq!(removed.len(), 1);
        assert!(removed[0].contains("很长的正文"));
        assert_eq!(context.messages().len(), 2);
        assert!(matches!(context.messages()[0], Message::User { .. }));
        assert!(matches!(
            context.messages()[1],
            Message::Assistant { ref tool_calls, .. } if tool_calls.is_empty()
        ));
        assert!(context.validate().is_ok());
    }

    #[test]
    /// 删除 RAG 调用时不能破坏同批次中的其他工具调用。
    fn compaction_preserves_other_tools_in_mixed_batch() {
        let mut context = ContextMemory::new("system");
        context.append_user("检索并计算").unwrap();
        context
            .append_assistant(
                None,
                vec![
                    ToolCall {
                        id: "rag".into(),
                        name: "search_docs".into(),
                        arguments: "{}".into(),
                    },
                    call("calc"),
                ],
            )
            .unwrap();
        context
            .append_tool_result("rag", "资料正文", false)
            .unwrap();
        context.append_tool_result("calc", "2", false).unwrap();
        context
            .append_assistant(Some("完成".into()), Vec::new())
            .unwrap();

        context
            .compact_tool_exchanges_since(0, "search_docs")
            .unwrap();
        assert_eq!(context.messages().len(), 4);
        assert!(matches!(
            &context.messages()[1],
            Message::Assistant { tool_calls, .. }
                if tool_calls.len() == 1 && tool_calls[0].name == "calculate"
        ));
        assert!(matches!(
            &context.messages()[2],
            Message::Tool { call_id, .. } if call_id == "calc"
        ));
        assert!(context.validate().is_ok());
    }

    #[test]
    fn prefix_compaction_removes_mcp_exchange_only() {
        let mut context = ContextMemory::new("system");
        context.append_user("调用 MCP 并计算").unwrap();
        context
            .append_assistant(
                None,
                vec![
                    ToolCall {
                        id: "mcp".into(),
                        name: "mcp__fixture__echo".into(),
                        arguments: "{}".into(),
                    },
                    call("calc"),
                ],
            )
            .unwrap();
        context
            .append_tool_result("mcp", "远端结果", false)
            .unwrap();
        context.append_tool_result("calc", "2", false).unwrap();
        context
            .append_assistant(Some("完成".into()), Vec::new())
            .unwrap();

        let removed = context
            .compact_tool_exchanges_with_prefix_since(0, "mcp__")
            .unwrap();
        assert_eq!(removed, vec!["远端结果"]);
        assert!(context.messages().iter().any(|message| {
            matches!(message, Message::Tool { call_id, .. } if call_id == "calc")
        }));
        assert!(context.validate().is_ok());
    }
}
