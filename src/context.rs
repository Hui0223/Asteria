use crate::message::{Message, ToolCall};
use anyhow::{Result, bail};
use serde_json::{Value, json};
use std::collections::HashSet;

pub struct ContextMemory {
    system_prompt: String,
    messages: Vec<Message>,
}

impl ContextMemory {
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            messages: Vec::new(),
        }
    }

    pub fn checkpoint(&self) -> usize {
        self.messages.len()
    }

    pub fn rollback(&mut self, checkpoint: usize) {
        self.messages.truncate(checkpoint);
    }

    pub fn reset(&mut self) {
        self.messages.clear();
    }

    pub fn append_user(&mut self, content: impl Into<String>) -> Result<()> {
        self.ensure_no_pending_tools()?;
        self.messages.push(Message::User {
            content: content.into(),
        });
        Ok(())
    }

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

    pub fn project_deepseek(&self) -> Result<Vec<Value>> {
        self.ensure_no_pending_tools()?;
        let mut wire = vec![json!({"role":"system", "content":self.system_prompt})];
        for message in &self.messages {
            wire.push(match message {
                Message::User { content } => json!({"role":"user", "content":content}),
                Message::Assistant {
                    content,
                    tool_calls,
                } if tool_calls.is_empty() => json!({"role":"assistant", "content":content}),
                Message::Assistant {
                    content,
                    tool_calls,
                } => {
                    json!({
                        "role":"assistant", "content":content,
                        "tool_calls":tool_calls.iter().map(|call| json!({
                            "id":call.id, "type":"function",
                            "function":{"name":call.name, "arguments":call.arguments}
                        })).collect::<Vec<_>>()
                    })
                }
                Message::Tool {
                    call_id, content, ..
                } => json!({"role":"tool", "tool_call_id":call_id, "content":content}),
            });
        }
        Ok(wire)
    }

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

    fn call(id: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "calculate".into(),
            arguments: "{\"expression\":\"1+1\"}".into(),
        }
    }

    #[test]
    fn reset_keeps_system_prompt() {
        let mut context = ContextMemory::new("system");
        context.append_user("hello").unwrap();
        context.reset();
        assert_eq!(
            context.project_deepseek().unwrap(),
            vec![json!({"role":"system","content":"system"})]
        );
    }

    #[test]
    fn rollback_removes_the_current_turn() {
        let mut context = ContextMemory::new("system");
        let checkpoint = context.checkpoint();
        context.append_user("temporary").unwrap();
        context.rollback(checkpoint);
        assert_eq!(context.project_deepseek().unwrap().len(), 1);
    }

    #[test]
    fn rejects_orphan_and_duplicate_tool_results() {
        let mut context = ContextMemory::new("system");
        assert!(context.append_tool_result("missing", "x", true).is_err());
        context.append_assistant(None, vec![call("a")]).unwrap();
        context.append_tool_result("a", "2", false).unwrap();
        assert!(context.append_tool_result("a", "2", false).is_err());
    }

    #[test]
    fn parallel_results_match_by_call_id() {
        let mut context = ContextMemory::new("system");
        context
            .append_assistant(None, vec![call("a"), call("b")])
            .unwrap();
        context.append_tool_result("b", "4", false).unwrap();
        assert!(context.project_deepseek().is_err());
        context.append_tool_result("a", "2", false).unwrap();
        assert!(context.project_deepseek().is_ok());
    }

    #[test]
    fn projection_omits_empty_tool_calls() {
        let mut context = ContextMemory::new("system");
        context
            .append_assistant(Some("done".into()), Vec::new())
            .unwrap();
        let projected = context.project_deepseek().unwrap();
        assert!(projected[1].get("tool_calls").is_none());
    }
}
