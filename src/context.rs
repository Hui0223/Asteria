use crate::message::{Message, ToolCall};
use anyhow::{Result, bail};
use std::collections::HashSet;

/// 保存与模型供应商无关的对话历史，并维护消息顺序约束。
pub struct ContextMemory {
    system_prompt: String,
    messages: Vec<Message>,
}

impl ContextMemory {
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

    /// 清空对话消息，但保留系统提示词。
    pub fn reset(&mut self) {
        self.messages.clear();
    }

    /// 返回只读的系统提示词，供模型适配器组装请求。
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
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
}
