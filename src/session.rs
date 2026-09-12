use crate::{context::ContextMemory, message::Message, provider::TokenUsage};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

/// JSONL 中保存的可重放会话事件。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum SessionEvent {
    Message { message: Message },
    TurnCompleted { turn_id: u64, usage: TokenUsage },
}

/// 管理单个 Asteria 会话的 JSONL 追加日志和恢复状态。
pub struct SessionStore {
    path: PathBuf,
}

/// 从磁盘恢复出的上下文、Turn 编号和累计用量。
pub struct RestoredSession {
    pub context: ContextMemory,
    pub next_turn_id: u64,
    pub usage: TokenUsage,
}

impl SessionStore {
    /// 根据 ASTERIA_SESSION_PATH 创建会话存储，默认写入 .asteria/session.jsonl。
    pub fn from_env() -> Result<Self> {
        let path = std::env::var("ASTERIA_SESSION_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".asteria/session.jsonl"));
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).context("无法创建会话目录")?;
        }
        Ok(Self { path })
    }

    /// 读取历史事件并重建 ContextMemory；损坏的最后一行会被忽略。
    pub fn restore(&self, system_prompt: &str) -> Result<RestoredSession> {
        let mut messages = Vec::new();
        let mut pending_messages = Vec::new();
        let mut next_turn_id = 1;
        let mut usage = TokenUsage::default();
        let file = match File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(RestoredSession {
                    context: ContextMemory::new(system_prompt),
                    next_turn_id,
                    usage,
                });
            }
            Err(error) => return Err(error.into()),
        };
        for line in BufReader::new(file).lines() {
            let Ok(line) = line else { continue };
            let Ok(event) = serde_json::from_str::<SessionEvent>(&line) else {
                continue;
            };
            match event {
                SessionEvent::Message { message } => pending_messages.push(message),
                SessionEvent::TurnCompleted {
                    turn_id,
                    usage: turn_usage,
                } => {
                    messages.append(&mut pending_messages);
                    next_turn_id = next_turn_id.max(turn_id + 1);
                    usage.prompt_tokens += turn_usage.prompt_tokens;
                    usage.completion_tokens += turn_usage.completion_tokens;
                    usage.total_tokens += turn_usage.total_tokens;
                }
            }
        }
        Ok(RestoredSession {
            context: ContextMemory::restore(system_prompt, messages)?,
            next_turn_id,
            usage,
        })
    }

    /// 追加一批完整消息和 Turn 元数据，并在每条事件写入后刷新文件。
    pub fn append_turn(
        &self,
        messages: &[Message],
        turn_id: u64,
        usage: &TokenUsage,
    ) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        for message in messages {
            write_event(
                &mut file,
                &SessionEvent::Message {
                    message: message.clone(),
                },
            )?;
        }
        write_event(
            &mut file,
            &SessionEvent::TurnCompleted {
                turn_id,
                usage: usage.clone(),
            },
        )?;
        file.sync_data()?;
        Ok(())
    }

    /// 清空持久化日志，供用户执行 /reset 时同步清理会话。
    pub fn clear(&self) -> Result<()> {
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?
            .sync_data()?;
        Ok(())
    }

    /// 返回当前日志路径，便于诊断显示。
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// 序列化一行事件并立即写入文件。
fn write_event(file: &mut File, event: &SessionEvent) -> Result<()> {
    writeln!(file, "{}", serde_json::to_string(event)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// 创建本测试专用的临时 JSONL 路径，避免触碰用户会话。
    fn test_path() -> PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("asteria-session-test-{id}.jsonl"))
    }

    #[test]
    /// 验证消息、Turn ID 和 Token Usage 可以跨 Store 实例恢复。
    fn persists_and_restores_completed_turn() {
        let path = test_path();
        let store = SessionStore { path: path.clone() };
        let messages = vec![
            Message::User {
                content: "hello".into(),
            },
            Message::Assistant {
                content: Some("world".into()),
                tool_calls: Vec::new(),
            },
        ];
        let usage = TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 2,
            total_tokens: 12,
        };
        store.append_turn(&messages, 7, &usage).unwrap();
        let restored = store.restore("system").unwrap();
        assert_eq!(restored.next_turn_id, 8);
        assert_eq!(restored.usage, usage);
        assert_eq!(restored.context.messages(), messages);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    /// 验证没有 TurnCompleted 的半批次不会进入恢复后的 Context。
    fn ignores_incomplete_turn_batch() {
        let path = test_path();
        let store = SessionStore { path: path.clone() };
        let mut file = File::create(&path).unwrap();
        write_event(
            &mut file,
            &SessionEvent::Message {
                message: Message::User {
                    content: "unfinished".into(),
                },
            },
        )
        .unwrap();
        drop(file);
        let restored = store.restore("system").unwrap();
        assert!(restored.context.messages().is_empty());
        assert_eq!(restored.next_turn_id, 1);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    /// 验证损坏的最后一行不会阻止之前已完成 Turn 的恢复。
    fn skips_corrupted_tail_line() {
        let path = test_path();
        let store = SessionStore { path: path.clone() };
        let usage = TokenUsage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        };
        store
            .append_turn(
                &[Message::User {
                    content: "ok".into(),
                }],
                1,
                &usage,
            )
            .unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{broken-json\n")
            .unwrap();
        let restored = store.restore("system").unwrap();
        assert_eq!(restored.context.messages().len(), 1);
        assert_eq!(restored.next_turn_id, 2);
        let _ = std::fs::remove_file(path);
    }
}
