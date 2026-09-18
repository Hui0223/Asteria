use crate::{
    context::ContextMemory,
    events::{AgentEvent, EventSink, ToolTrace},
    message::Message,
    permission::ToolPermission,
    provider::TokenUsage,
    rag::RagSourceRef,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

const CONTEXT_FILE: &str = "context.jsonl";
const TRACE_FILE: &str = "trace.jsonl";
const STATE_FILE: &str = "state.json";

/// 旧版单文件会话格式，仅用于无损迁移。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum LegacySessionEvent {
    Message {
        message: Message,
    },
    TurnCompleted {
        turn_id: u64,
        usage: TokenUsage,
    },
    PermissionChanged {
        tool: String,
        permission: ToolPermission,
    },
    AgentEvent {
        event: AgentEvent,
    },
    RetrievalSources {
        turn_id: u64,
        sources: Vec<RagSourceRef>,
    },
    ToolTrace {
        trace: ToolTrace,
    },
}

/// context.jsonl 只保存可重放消息及其 Turn 提交边界。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum ContextRecord {
    Message { message: Message },
    TurnCompleted { turn_id: u64, usage: TokenUsage },
}

/// trace.jsonl 保存所有不参与模型上下文的运行审计信息。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum TraceRecord {
    AgentEvent {
        event: AgentEvent,
    },
    RetrievalSources {
        turn_id: u64,
        sources: Vec<RagSourceRef>,
    },
    ToolTrace {
        trace: ToolTrace,
    },
    McpLifecycle {
        action: String,
        server: Option<String>,
        succeeded: bool,
        detail: Option<String>,
        occurred_at: String,
    },
}

/// state.json 保存可独立恢复的轻量会话元数据。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
struct SessionState {
    version: u32,
    legacy_migration_version: u32,
    next_turn_id: u64,
    usage: TokenUsage,
    permissions: Vec<(String, ToolPermission)>,
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            version: 1,
            legacy_migration_version: 0,
            next_turn_id: 1,
            usage: TokenUsage::default(),
            permissions: Vec::new(),
        }
    }
}

/// 单个会话目录中的三个持久化文件。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionPaths {
    pub directory: PathBuf,
    pub context: PathBuf,
    pub trace: PathBuf,
    pub state: PathBuf,
}

/// 管理单个 Asteria 会话目录的上下文、审计和状态文件。
#[derive(Clone)]
pub struct SessionStore {
    paths: SessionPaths,
    write_lock: Arc<Mutex<()>>,
}

/// 从磁盘恢复出的上下文、Turn 编号和累计用量。
pub struct RestoredSession {
    pub context: ContextMemory,
    pub next_turn_id: u64,
    pub usage: TokenUsage,
    pub permissions: Vec<(String, ToolPermission)>,
}

impl SessionStore {
    /// 根据 ASTERIA_SESSION_PATH 创建会话目录，默认使用 .asteria/session/。
    ///
    /// 兼容旧配置：若变量值以 .jsonl 结尾，则将同名无扩展路径作为新目录，
    /// 并自动迁移原单文件日志。
    pub fn from_env() -> Result<Self> {
        let configured = std::env::var("ASTERIA_SESSION_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".asteria/session"));
        let (directory, legacy_path) = if configured.extension().is_some_and(|ext| ext == "jsonl") {
            (configured.with_extension(""), configured)
        } else {
            let legacy_path = configured.with_extension("jsonl");
            (configured, legacy_path)
        };
        let store = Self::at(directory)?;
        store.migrate_legacy(&legacy_path)?;
        store.ensure_layout()?;
        Ok(store)
    }

    fn at(directory: PathBuf) -> Result<Self> {
        fs::create_dir_all(&directory).context("无法创建会话目录")?;
        Ok(Self {
            paths: SessionPaths {
                context: directory.join(CONTEXT_FILE),
                trace: directory.join(TRACE_FILE),
                state: directory.join(STATE_FILE),
                directory,
            },
            write_lock: Arc::new(Mutex::new(())),
        })
    }

    /// 读取 context.jsonl 并重建 ContextMemory；审计文件不会参与恢复。
    pub fn restore(&self, system_prompt: &str) -> Result<RestoredSession> {
        let _guard = self.lock()?;
        let mut messages = Vec::new();
        let mut pending_messages = Vec::new();
        let mut next_turn_id = 1;
        let mut usage = TokenUsage::default();
        let mut completed_turns = 0usize;
        let state = self.load_state()?;
        let file = match File::open(&self.paths.context) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(RestoredSession {
                    context: ContextMemory::new(system_prompt),
                    next_turn_id: state.next_turn_id,
                    usage: state.usage,
                    permissions: state.permissions,
                });
            }
            Err(error) => return Err(error.into()),
        };
        for line in BufReader::new(file).lines() {
            let Ok(line) = line else { continue };
            let Ok(event) = serde_json::from_str::<ContextRecord>(&line) else {
                continue;
            };
            match event {
                ContextRecord::Message { message } => pending_messages.push(message),
                ContextRecord::TurnCompleted {
                    turn_id,
                    usage: turn_usage,
                } => {
                    messages.append(&mut pending_messages);
                    completed_turns += 1;
                    next_turn_id = next_turn_id.max(turn_id + 1);
                    usage.prompt_tokens += turn_usage.prompt_tokens;
                    usage.completion_tokens += turn_usage.completion_tokens;
                    usage.total_tokens += turn_usage.total_tokens;
                }
            }
        }
        if completed_turns == 0 {
            usage = state.usage;
        }
        Ok(RestoredSession {
            context: ContextMemory::restore(system_prompt, messages)?,
            next_turn_id: next_turn_id.max(state.next_turn_id),
            usage,
            permissions: state.permissions,
        })
    }

    /// 追加当前会话的一条工具权限变更。
    pub fn append_permission(&self, tool: &str, permission: ToolPermission) -> Result<()> {
        let _guard = self.lock()?;
        let mut state = self.load_state()?;
        state.permissions.retain(|(name, _)| name != tool);
        state.permissions.push((tool.into(), permission));
        self.save_state(&state)
    }

    /// 用当前 Registry 的完整权限快照替换 state，清理已注销工具的过期权限。
    pub fn replace_permissions(&self, permissions: &[(String, ToolPermission)]) -> Result<()> {
        let _guard = self.lock()?;
        let mut state = self.load_state()?;
        state.permissions = permissions.to_vec();
        state
            .permissions
            .sort_by(|left, right| left.0.cmp(&right.0));
        state.permissions.dedup_by(|left, right| left.0 == right.0);
        self.save_state(&state)
    }

    /// 将一次 AgentEvent 追加到 trace.jsonl，供审计和 Transcript 使用。
    pub fn append_agent_event(&self, event: &AgentEvent) -> Result<()> {
        if event.is_volatile() {
            return Ok(());
        }
        let _guard = self.lock()?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.paths.trace)?;
        write_record(
            &mut file,
            &TraceRecord::AgentEvent {
                event: event.clone(),
            },
        )?;
        file.sync_data()?;
        Ok(())
    }

    /// 记录 MCP reload/connect/disconnect；该审计不参与模型上下文。
    pub fn append_mcp_lifecycle(
        &self,
        action: &str,
        server: Option<&str>,
        succeeded: bool,
        detail: Option<&str>,
    ) -> Result<()> {
        let _guard = self.lock()?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.paths.trace)?;
        write_record(
            &mut file,
            &TraceRecord::McpLifecycle {
                action: action.into(),
                server: server.map(str::to_owned),
                succeeded,
                detail: detail.map(str::to_owned),
                occurred_at: chrono::Utc::now().to_rfc3339(),
            },
        )?;
        file.sync_data()?;
        Ok(())
    }

    /// 创建将事件写入该会话文件的 EventSink。
    pub fn event_sink(self: &std::sync::Arc<Self>) -> PersistentEventSink {
        PersistentEventSink {
            store: self.clone(),
        }
    }

    /// 追加一批可重放消息和 Turn 提交记录。
    pub fn append_turn(
        &self,
        messages: &[Message],
        turn_id: u64,
        usage: &TokenUsage,
    ) -> Result<()> {
        self.append_turn_with_sources(messages, turn_id, usage, &[])
    }

    /// 持久化精简后的 Turn，并把 RAG 来源写入独立审计文件。
    pub fn append_turn_with_sources(
        &self,
        messages: &[Message],
        turn_id: u64,
        usage: &TokenUsage,
        sources: &[RagSourceRef],
    ) -> Result<()> {
        self.append_turn_with_audit(messages, turn_id, usage, sources, &[])
    }

    /// 持久化精简 Turn、RAG 来源和脱敏工具 Trace。
    ///
    /// trace.jsonl 先落盘，随后提交 context.jsonl，最后原子更新 state.json。
    /// 即使状态更新中断，恢复逻辑仍可从 context.jsonl 的提交记录重建编号和用量。
    pub fn append_turn_with_audit(
        &self,
        messages: &[Message],
        turn_id: u64,
        usage: &TokenUsage,
        sources: &[RagSourceRef],
        traces: &[ToolTrace],
    ) -> Result<()> {
        let _guard = self.lock()?;
        let mut trace_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.paths.trace)?;
        for trace in traces {
            write_record(
                &mut trace_file,
                &TraceRecord::ToolTrace {
                    trace: trace.clone(),
                },
            )?;
        }
        if !sources.is_empty() {
            write_record(
                &mut trace_file,
                &TraceRecord::RetrievalSources {
                    turn_id,
                    sources: sources.to_vec(),
                },
            )?;
        }
        trace_file.sync_data()?;

        let mut context_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.paths.context)?;
        for message in messages {
            write_record(
                &mut context_file,
                &ContextRecord::Message {
                    message: message.clone(),
                },
            )?;
        }
        write_record(
            &mut context_file,
            &ContextRecord::TurnCompleted {
                turn_id,
                usage: usage.clone(),
            },
        )?;
        context_file.sync_data()?;

        let mut state = self.load_state()?;
        state.next_turn_id = state.next_turn_id.max(turn_id + 1);
        state.usage.prompt_tokens += usage.prompt_tokens;
        state.usage.completion_tokens += usage.completion_tokens;
        state.usage.total_tokens += usage.total_tokens;
        self.save_state(&state)
    }

    /// 从 trace.jsonl 读取全部或指定 Turn 的工具 Trace。
    pub fn tool_traces(&self, turn_id: Option<u64>) -> Result<Vec<ToolTrace>> {
        let _guard = self.lock()?;
        let file = match File::open(&self.paths.trace) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let traces = BufReader::new(file)
            .lines()
            .map_while(Result::ok)
            .filter_map(|line| serde_json::from_str::<TraceRecord>(&line).ok())
            .filter_map(|event| match event {
                TraceRecord::ToolTrace { trace }
                    if turn_id.is_none_or(|turn_id| trace.turn_id == turn_id) =>
                {
                    Some(trace)
                }
                _ => None,
            })
            .collect();
        Ok(traces)
    }

    /// 单独保存失败 Turn 已产生的工具 Trace，不写入可重放消息。
    pub fn append_tool_traces(&self, traces: &[ToolTrace]) -> Result<()> {
        if traces.is_empty() {
            return Ok(());
        }
        let _guard = self.lock()?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.paths.trace)?;
        for trace in traces {
            write_record(
                &mut file,
                &TraceRecord::ToolTrace {
                    trace: trace.clone(),
                },
            )?;
        }
        file.sync_data()?;
        Ok(())
    }

    /// 清空上下文、审计和状态文件，供 /reset 与 /new-session 使用。
    pub fn clear(&self) -> Result<()> {
        let _guard = self.lock()?;
        truncate_file(&self.paths.context)?;
        truncate_file(&self.paths.trace)?;
        self.save_state(&SessionState::default())
    }

    /// 返回当前会话目录，便于诊断显示。
    pub fn path(&self) -> &Path {
        &self.paths.directory
    }

    /// 返回上下文、审计和状态文件路径。
    pub fn paths(&self) -> &SessionPaths {
        &self.paths
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, ()>> {
        self.write_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("会话存储锁已损坏"))
    }

    fn ensure_layout(&self) -> Result<()> {
        let _guard = self.lock()?;
        touch_file(&self.paths.context)?;
        touch_file(&self.paths.trace)?;
        if !self.paths.state.exists() {
            self.save_state(&SessionState::default())?;
        }
        Ok(())
    }

    fn load_state(&self) -> Result<SessionState> {
        match fs::read_to_string(&self.paths.state) {
            Ok(content) if content.trim().is_empty() => Ok(SessionState::default()),
            Ok(content) => serde_json::from_str(&content).context("无法解析 state.json"),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(SessionState::default())
            }
            Err(error) => Err(error.into()),
        }
    }

    fn save_state(&self, state: &SessionState) -> Result<()> {
        let temporary = self.paths.directory.join(".state.json.tmp");
        let mut file = File::create(&temporary)?;
        serde_json::to_writer_pretty(&mut file, state)?;
        file.write_all(b"\n")?;
        file.sync_data()?;
        fs::rename(&temporary, &self.paths.state)?;
        Ok(())
    }

    /// 首次启用三文件布局时，将旧单文件日志按语义拆分；原文件保留作备份。
    fn migrate_legacy(&self, legacy_path: &Path) -> Result<()> {
        if !legacy_path.is_file() {
            return Ok(());
        }
        // 迁移版本最后写入 state.json，兼作完成标记。旧版错误迁移生成的 State
        // 没有该字段（反序列化为 0），升级后会自动重新拆分一次。
        if self.load_state()?.legacy_migration_version >= 1 {
            return Ok(());
        }
        let _guard = self.lock()?;
        let file = File::open(legacy_path)?;
        let mut context_file = File::create(&self.paths.context)?;
        let mut trace_file = File::create(&self.paths.trace)?;
        let mut state = SessionState::default();

        for line in BufReader::new(file).lines() {
            let Ok(line) = line else { continue };
            // 旧版有两个写入线程共用同一文件，极端情况下两个完整 JSON 对象会
            // 紧邻在同一行。流式反序列化可以逐个找回，避免丢失工具调用消息。
            let records =
                serde_json::Deserializer::from_str(&line).into_iter::<serde_json::Value>();
            for record in records {
                let Ok(record) = record else { break };
                migrate_legacy_record(record, &mut context_file, &mut trace_file, &mut state)?;
            }
        }
        context_file.sync_data()?;
        trace_file.sync_data()?;
        state.legacy_migration_version = 1;
        self.save_state(&state)
    }
}

/// 将执行事件写入 SessionStore 的持久化接收器。
pub struct PersistentEventSink {
    store: std::sync::Arc<SessionStore>,
}

impl EventSink for PersistentEventSink {
    /// 事件持久化失败只记录错误，不阻断当前 Turn。
    fn publish(&self, event: AgentEvent) {
        if let Err(error) = self.store.append_agent_event(&event) {
            eprintln!("事件持久化失败: {error:#}");
        }
    }
}

/// 序列化一行 JSONL 记录。
fn write_record(file: &mut File, record: &impl Serialize) -> Result<()> {
    writeln!(file, "{}", serde_json::to_string(record)?)?;
    Ok(())
}

fn touch_file(path: &Path) -> Result<()> {
    OpenOptions::new().create(true).append(true).open(path)?;
    Ok(())
}

fn truncate_file(path: &Path) -> Result<()> {
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?
        .sync_data()?;
    Ok(())
}

fn migrate_legacy_record(
    record: serde_json::Value,
    context_file: &mut File,
    trace_file: &mut File,
    state: &mut SessionState,
) -> Result<()> {
    if record.get("type").and_then(|kind| kind.as_str()) == Some("AgentEvent") {
        // 旧 AgentEvent 的 duration_ms 使用 u128，serde 的内部标签格式无法
        // 反序列化该数字；其外层 JSON 与 TraceRecord 线格式兼容，原样保留。
        return write_record(trace_file, &record);
    }
    let Ok(event) = serde_json::from_value::<LegacySessionEvent>(record) else {
        return Ok(());
    };
    match event {
        LegacySessionEvent::Message { message } => {
            write_record(context_file, &ContextRecord::Message { message })?;
        }
        LegacySessionEvent::TurnCompleted { turn_id, usage } => {
            write_record(
                context_file,
                &ContextRecord::TurnCompleted {
                    turn_id,
                    usage: usage.clone(),
                },
            )?;
            state.next_turn_id = state.next_turn_id.max(turn_id + 1);
            state.usage.prompt_tokens += usage.prompt_tokens;
            state.usage.completion_tokens += usage.completion_tokens;
            state.usage.total_tokens += usage.total_tokens;
        }
        LegacySessionEvent::PermissionChanged { tool, permission } => {
            state.permissions.retain(|(name, _)| name != &tool);
            state.permissions.push((tool, permission));
        }
        LegacySessionEvent::RetrievalSources { turn_id, sources } => {
            write_record(
                trace_file,
                &TraceRecord::RetrievalSources { turn_id, sources },
            )?;
        }
        LegacySessionEvent::ToolTrace { trace } => {
            write_record(trace_file, &TraceRecord::ToolTrace { trace })?;
        }
        LegacySessionEvent::AgentEvent { event } => {
            write_record(trace_file, &TraceRecord::AgentEvent { event })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, ToolCall};
    use std::time::{SystemTime, UNIX_EPOCH};

    /// 创建本测试专用的临时会话目录，避免触碰用户会话。
    fn test_store() -> (SessionStore, PathBuf) {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("asteria-session-test-{id}"));
        let store = SessionStore::at(directory.clone()).unwrap();
        store.ensure_layout().unwrap();
        (store, directory)
    }

    #[test]
    /// 验证消息、Turn ID 和 Token Usage 可以跨 Store 实例恢复。
    fn persists_and_restores_completed_turn() {
        let (store, directory) = test_store();
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
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn replaces_permission_snapshot_and_removes_stale_tools() {
        let (store, directory) = test_store();
        store
            .append_permission("mcp__old__echo", ToolPermission::Allow)
            .unwrap();
        store
            .replace_permissions(&[
                ("Read".into(), ToolPermission::Allow),
                ("mcp__new__echo".into(), ToolPermission::Ask),
            ])
            .unwrap();

        assert_eq!(
            store.restore("system").unwrap().permissions,
            vec![
                ("Read".into(), ToolPermission::Allow),
                ("mcp__new__echo".into(), ToolPermission::Ask),
            ]
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn skips_volatile_assistant_deltas() {
        let (store, directory) = test_store();
        store
            .append_agent_event(&AgentEvent::AssistantDelta {
                turn_id: 1,
                step: 1,
                delta: "streaming".into(),
                offset: 0,
            })
            .unwrap();
        store
            .append_agent_event(&AgentEvent::TurnCompleted {
                turn_id: 1,
                steps: 1,
            })
            .unwrap();
        let trace = fs::read_to_string(&store.paths.trace).unwrap();
        assert!(!trace.contains("streaming"));
        assert!(trace.contains("TurnCompleted"));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn mcp_lifecycle_is_written_only_to_trace() {
        let (store, directory) = test_store();
        store
            .append_mcp_lifecycle("disconnect", Some("fixture"), true, None)
            .unwrap();

        let trace = fs::read_to_string(&store.paths.trace).unwrap();
        assert!(trace.contains("\"type\":\"McpLifecycle\""));
        assert!(trace.contains("\"server\":\"fixture\""));
        assert!(fs::read_to_string(&store.paths.context).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    /// 验证没有 TurnCompleted 的半批次不会进入恢复后的 Context。
    fn ignores_incomplete_turn_batch() {
        let (store, directory) = test_store();
        let mut file = File::create(&store.paths.context).unwrap();
        write_record(
            &mut file,
            &ContextRecord::Message {
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
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    /// 验证损坏的最后一行不会阻止之前已完成 Turn 的恢复。
    fn skips_corrupted_tail_line() {
        let (store, directory) = test_store();
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
            .open(&store.paths.context)
            .unwrap()
            .write_all(b"{broken-json\n")
            .unwrap();
        let restored = store.restore("system").unwrap();
        assert_eq!(restored.context.messages().len(), 1);
        assert_eq!(restored.next_turn_id, 2);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    /// RAG 只持久化短来源，完整正文不能进入可重放消息。
    fn persists_short_retrieval_sources_outside_context() {
        let (store, directory) = test_store();
        let messages = vec![
            Message::User {
                content: "第14章是什么".into(),
            },
            Message::Assistant {
                content: Some("第14章介绍持久化配置。".into()),
                tool_calls: Vec::new(),
            },
        ];
        let sources = vec![RagSourceRef {
            source: PathBuf::from("guide.docx"),
            chunk_indices: vec![14, 15],
        }];
        store
            .append_turn_with_sources(&messages, 3, &TokenUsage::default(), &sources)
            .unwrap();

        let trace_jsonl = fs::read_to_string(&store.paths.trace).unwrap();
        let context_jsonl = fs::read_to_string(&store.paths.context).unwrap();
        assert!(trace_jsonl.contains("\"type\":\"RetrievalSources\""));
        assert!(trace_jsonl.contains("guide.docx"));
        assert!(!context_jsonl.contains("guide.docx"));
        assert!(!trace_jsonl.contains("完整检索正文"));
        let restored = store.restore("system").unwrap();
        assert_eq!(restored.context.messages(), messages);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    /// 工具 Trace 可按 Turn 查询，但参数密钥和完整结果都不能写入日志。
    fn persists_redacted_tool_trace_outside_context() {
        let (store, directory) = test_store();
        let messages = vec![Message::User {
            content: "运行工具".into(),
        }];
        let trace = ToolTrace::completed(
            9,
            2,
            &ToolCall {
                id: "call-9".into(),
                name: "example".into(),
                arguments: r#"{"query":"chapter 14","token":"sensitive-token"}"#.into(),
            },
            "完整工具结果正文",
            false,
            18,
        );
        store
            .append_turn_with_audit(
                &messages,
                9,
                &TokenUsage::default(),
                &[],
                std::slice::from_ref(&trace),
            )
            .unwrap();

        let traces = store.tool_traces(Some(9)).unwrap();
        assert_eq!(traces, vec![trace]);
        assert!(store.tool_traces(Some(8)).unwrap().is_empty());
        let trace_jsonl = fs::read_to_string(&store.paths.trace).unwrap();
        let context_jsonl = fs::read_to_string(&store.paths.context).unwrap();
        assert!(trace_jsonl.contains("\"type\":\"ToolTrace\""));
        assert!(!trace_jsonl.contains("sensitive-token"));
        assert!(!trace_jsonl.contains("完整工具结果正文"));
        assert!(!context_jsonl.contains("\"type\":\"ToolTrace\""));
        assert_eq!(
            store.restore("system").unwrap().context.messages(),
            messages
        );
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    /// 首次启动时将旧 session.jsonl 拆分，并保留原文件作为备份。
    fn migrates_legacy_session_into_three_files() {
        let (initial_store, directory) = test_store();
        drop(initial_store);
        fs::remove_dir_all(&directory).unwrap();
        let legacy_path = directory.with_extension("jsonl");
        let mut legacy = File::create(&legacy_path).unwrap();
        let message = Message::User {
            content: "legacy message".into(),
        };
        let tool_call_message = Message::Assistant {
            content: None,
            tool_calls: vec![ToolCall {
                id: "legacy-event".into(),
                name: "calculate".into(),
                arguments: r#"{"expression":"1+1"}"#.into(),
            }],
        };
        let tool_result_message = Message::Tool {
            call_id: "legacy-event".into(),
            content: "2".into(),
            is_error: false,
        };
        let answer_message = Message::Assistant {
            content: Some("1+1 = 2".into()),
            tool_calls: Vec::new(),
        };
        let trace = ToolTrace::completed(
            4,
            1,
            &ToolCall {
                id: "legacy-call".into(),
                name: "search_docs".into(),
                arguments: r#"{"query":"legacy"}"#.into(),
            },
            "legacy result",
            false,
            7,
        );
        write_record(
            &mut legacy,
            &LegacySessionEvent::Message {
                message: message.clone(),
            },
        )
        .unwrap();
        write_record(
            &mut legacy,
            &LegacySessionEvent::ToolTrace {
                trace: trace.clone(),
            },
        )
        .unwrap();
        write_record(
            &mut legacy,
            &LegacySessionEvent::PermissionChanged {
                tool: "search_docs".into(),
                permission: ToolPermission::Deny,
            },
        )
        .unwrap();
        write!(
            legacy,
            r#"{{"type":"AgentEvent","event":{{"ToolResult":{{"turn_id":4,"call_id":"legacy-event","is_error":false,"duration_ms":7}}}}}}"#
        )
        .unwrap();
        write_record(
            &mut legacy,
            &LegacySessionEvent::Message {
                message: tool_call_message.clone(),
            },
        )
        .unwrap();
        for message in [&tool_result_message, &answer_message] {
            write_record(
                &mut legacy,
                &LegacySessionEvent::Message {
                    message: message.clone(),
                },
            )
            .unwrap();
        }
        write_record(
            &mut legacy,
            &LegacySessionEvent::TurnCompleted {
                turn_id: 4,
                usage: TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 2,
                    total_tokens: 12,
                },
            },
        )
        .unwrap();
        legacy.sync_data().unwrap();
        drop(legacy);

        let store = SessionStore::at(directory.clone()).unwrap();
        store.migrate_legacy(&legacy_path).unwrap();
        store.ensure_layout().unwrap();
        let restored = store.restore("system").unwrap();

        assert_eq!(
            restored.context.messages(),
            &[
                message,
                tool_call_message,
                tool_result_message,
                answer_message
            ]
        );
        assert_eq!(restored.next_turn_id, 5);
        assert_eq!(restored.usage.total_tokens, 12);
        assert_eq!(
            restored.permissions,
            vec![("search_docs".into(), ToolPermission::Deny)]
        );
        assert_eq!(store.tool_traces(Some(4)).unwrap(), vec![trace]);
        assert!(legacy_path.exists());
        assert!(
            !fs::read_to_string(&store.paths.context)
                .unwrap()
                .contains("ToolTrace")
        );
        assert!(
            !fs::read_to_string(&store.paths.trace)
                .unwrap()
                .contains("legacy message")
        );
        assert!(
            fs::read_to_string(&store.paths.trace)
                .unwrap()
                .contains("legacy-event")
        );
        let _ = fs::remove_dir_all(directory);
        let _ = fs::remove_file(legacy_path);
    }
}
