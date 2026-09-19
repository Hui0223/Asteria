use crate::{
    agent_loop::{AgentLoop, CancelToken, LoopConfig, TurnReport},
    context::ContextMemory,
    events::{FanoutEventSink, QueuedEventSink},
    permission::ToolPermission,
    provider::TokenUsage,
    provider::deepseek::DeepSeekProvider,
    rag::{RagStore, SearchDocsTool, source_refs_from_outputs},
    session::SessionStore,
};
use anyhow::{Context, Result};
use std::collections::HashMap;

const SYSTEM: &str = "你是 Asteria，一个可靠、简洁的中文 AI 助手。核心工具包括 Read、Write、Edit、Glob、Grep、Bash、calculate 和 current_time；读取或修改工作区时应调用对应工具，不要声称完成未实际执行的操作。mcp__* 是配置后才会出现的可选扩展工具。用户提到登录或使用 GitHub 时，若已有 mcp__github__* 工具，直接调用它们（身份来自环境变量 GITHUB_TOKEN，例如先用 get_me 确认账号）；不要索要密码、验证码或浏览器登录，也不要说自己无法访问 GitHub。只有这些工具未注册或调用失败时，才说明需要在 .env 中配置 GITHUB_TOKEN。用户提到 Binance、币安、行情或交易时，若已有 mcp__binance-mcp-server__* 工具，直接调用它们；不要索要 Binance API Key。下单、撤单和子账户内转账会先征求批准；Agent 无法出金。";
const RAG_HINT: &str = "本地知识库已启用：查询 docs/rag-docs 中的手册、troubleshooting、章节或 LT 故障时必须调用 search_docs，不要用 Bash/Read 去解压或通读 docx。查看普通工作区源码时使用 Read、Glob 或 Grep。询问知识库第N章/目录/章节内容时，把用户原问题作为 query。以最新 search_docs 结果为准，资料不足时明确说不知道，不要编造。";

/// 对外提供简单问答接口，并组合上下文与 Agent Loop。
pub struct Asteria {
    agent_loop: AgentLoop<DeepSeekProvider>,
    context: ContextMemory,
    session: std::sync::Arc<SessionStore>,
    restored_permissions: Vec<(String, ToolPermission)>,
    mcp_manager: Option<crate::mcp::McpManager>,
}

impl Asteria {
    /// 根据环境变量创建 DeepSeek Agent，并采用默认 Loop 配置。
    pub fn new() -> Result<Self> {
        let session = std::sync::Arc::new(SessionStore::from_env()?);
        let restored = session.restore(SYSTEM)?;
        let mut context = restored.context;
        context.compact_tool_exchanges_since(0, "search_docs")?;
        context.compact_tool_exchanges_with_prefix_since(0, "mcp__")?;
        let mut agent_loop = AgentLoop::new(DeepSeekProvider::from_env()?, LoopConfig::default());
        agent_loop.restore_session_state(restored.next_turn_id, restored.usage);
        let persistent_events = std::sync::Arc::new(session.event_sink());
        agent_loop.set_event_sink(QueuedEventSink::new(vec![persistent_events]));
        for (tool, permission) in &restored.permissions {
            agent_loop.set_tool_permission(tool, *permission).ok();
        }
        Ok(Self {
            agent_loop,
            context,
            session,
            restored_permissions: restored.permissions,
            mcp_manager: None,
        })
    }

    /// 返回当前 Agent 使用的模型名称。
    pub fn model(&self) -> &str {
        self.agent_loop.model()
    }

    /// 返回最近一个 Turn 的真实 Token 使用量；尚未请求模型时返回 None。
    pub fn last_usage(&self) -> Option<&TokenUsage> {
        self.agent_loop.last_turn().map(|turn| &turn.usage)
    }

    /// 返回当前进程内所有 Turn 的累计 Token 使用量。
    pub fn session_usage(&self) -> &TokenUsage {
        self.agent_loop.session_usage()
    }

    /// 更新工具的会话权限，不影响已有 Context 和 Token 统计。
    pub fn set_tool_permission(
        &mut self,
        name: &str,
        permission: crate::permission::ToolPermission,
    ) -> Result<()> {
        self.agent_loop.set_tool_permission(name, permission)?;
        self.session.append_permission(name, permission)
    }

    /// 获取工具权限列表。
    pub fn tool_permissions(&self) -> Vec<(String, crate::permission::ToolPermission)> {
        self.agent_loop.tool_permissions()
    }

    /// 将本次调用的批准请求交给 UI。
    pub fn set_tool_approver(
        &mut self,
        approver: std::sync::Arc<dyn crate::permission::ToolApprover>,
    ) {
        self.agent_loop.set_tool_approver(approver);
    }

    /// 启用本地知识库检索工具，并补充系统提示；只读资料默认允许执行。
    pub fn enable_search_docs(&mut self, store: RagStore) {
        self.agent_loop.register_tool_with_permission(
            SearchDocsTool::new(std::sync::Arc::new(store)),
            ToolPermission::Allow,
        );
        if let Some((_, permission)) = self
            .restored_permissions
            .iter()
            .find(|(name, _)| name == "search_docs")
        {
            let _ = self
                .agent_loop
                .set_tool_permission("search_docs", *permission);
        }
        self.context
            .set_system_prompt(format!("{SYSTEM}\n{RAG_HINT}"));
    }

    /// 加载 MCP 配置、连接 stdio Server，并把发现的工具注册为默认 ask。
    pub async fn enable_mcp(&mut self, trust_project: bool) -> Result<()> {
        let loaded = crate::mcp::load_default(trust_project).await?;
        for tool in loaded.tools {
            let name = crate::tools::AgentTool::name(&tool).to_owned();
            self.agent_loop
                .register_tool_with_permission(tool, ToolPermission::Ask);
            if let Some((_, permission)) = self
                .restored_permissions
                .iter()
                .find(|(restored, _)| restored == &name)
            {
                let _ = self.agent_loop.set_tool_permission(&name, *permission);
            }
        }
        self.mcp_manager = Some(loaded.manager);
        Ok(())
    }

    /// 先完整连接候选配置，全部可用后再原子替换当前 MCP 工具和连接。
    pub async fn reload_mcp(&mut self, trust_project: bool) -> Result<()> {
        let result = async {
            let loaded = crate::mcp::load_default(trust_project).await?;
            if loaded.manager.has_blocking_failures() {
                anyhow::bail!(
                    "MCP reload 未应用，当前连接保持不变：{}",
                    loaded.manager.failure_summary()
                );
            }
            let permissions = self
                .agent_loop
                .tool_permissions()
                .into_iter()
                .collect::<HashMap<_, _>>();
            self.agent_loop.unregister_tools_with_prefix("mcp__");
            for tool in loaded.tools {
                let name = crate::tools::AgentTool::name(&tool).to_owned();
                let permission = permissions
                    .get(&name)
                    .copied()
                    .or_else(|| {
                        self.restored_permissions
                            .iter()
                            .find(|(restored, _)| restored == &name)
                            .map(|(_, permission)| *permission)
                    })
                    .unwrap_or(ToolPermission::Ask);
                self.agent_loop
                    .register_tool_with_permission(tool, permission);
            }
            let old = self.mcp_manager.replace(loaded.manager);
            self.session
                .replace_permissions(&self.agent_loop.tool_permissions())?;
            if let Some(old) = old {
                old.shutdown().await;
            }
            Ok(())
        }
        .await;
        self.record_mcp_lifecycle("reload", None, &result);
        result
    }

    /// 从当前配置连接或重新连接一个 MCP Server。
    pub async fn connect_mcp_server(
        &mut self,
        trust_project: bool,
        server_name: &str,
    ) -> Result<()> {
        self.replace_mcp_server(trust_project, server_name, true)
            .await
    }

    /// 打开浏览器完成 OAuth 后连接指定 MCP Server。
    pub async fn authorize_mcp_server(
        &mut self,
        trust_project: bool,
        server_name: &str,
    ) -> Result<()> {
        crate::mcp::clear_oauth_credentials(server_name)?;
        self.replace_mcp_server(trust_project, server_name, true)
            .await
    }

    /// 删除已保存的 OAuth 凭据，并断开该 Server。
    pub async fn logout_mcp_server(&mut self, server_name: &str) -> Result<()> {
        crate::mcp::clear_oauth_credentials(server_name)?;
        match self.disconnect_mcp_server(server_name).await {
            Ok(()) => Ok(()),
            Err(error) if error.to_string().contains("未知 MCP Server") => Ok(()),
            Err(error) => Err(error),
        }
    }

    async fn replace_mcp_server(
        &mut self,
        trust_project: bool,
        server_name: &str,
        interactive: bool,
    ) -> Result<()> {
        let result = async {
            let loaded = if interactive {
                crate::mcp::load_named_interactive(trust_project, server_name).await?
            } else {
                crate::mcp::load_named(trust_project, server_name).await?
            };
            if loaded.manager.has_blocking_failures() {
                anyhow::bail!(
                    "MCP Server `{server_name}` 连接失败：{}",
                    loaded.manager.failure_summary()
                );
            }
            let previous_permissions = self
                .agent_loop
                .tool_permissions()
                .into_iter()
                .collect::<HashMap<_, _>>();
            let old_tools = self
                .mcp_manager
                .as_ref()
                .and_then(|manager| manager.server_tools(server_name))
                .unwrap_or_default()
                .to_vec();
            let crate::mcp::McpLoadResult {
                manager: candidate,
                tools,
            } = loaded;
            let manager = self
                .mcp_manager
                .as_mut()
                .context("MCP Manager 尚未初始化")?;
            manager.replace_server(server_name, candidate).await?;
            for name in old_tools {
                self.agent_loop.unregister_tool(&name);
            }
            for tool in tools {
                let name = crate::tools::AgentTool::name(&tool).to_owned();
                let permission = previous_permissions
                    .get(&name)
                    .copied()
                    .unwrap_or(ToolPermission::Ask);
                self.agent_loop
                    .register_tool_with_permission(tool, permission);
            }
            self.session
                .replace_permissions(&self.agent_loop.tool_permissions())
        }
        .await;
        self.record_mcp_lifecycle("connect", Some(server_name), &result);
        result
    }

    /// 关闭一个 MCP Server 并立即从模型工具列表删除它的全部工具。
    pub async fn disconnect_mcp_server(&mut self, server_name: &str) -> Result<()> {
        let result = async {
            let manager = self
                .mcp_manager
                .as_mut()
                .context("MCP Manager 尚未初始化")?;
            let tools = manager.disconnect(server_name).await?;
            for name in tools {
                self.agent_loop.unregister_tool(&name);
            }
            self.session
                .replace_permissions(&self.agent_loop.tool_permissions())
        }
        .await;
        self.record_mcp_lifecycle("disconnect", Some(server_name), &result);
        result
    }

    fn record_mcp_lifecycle(&self, action: &str, server: Option<&str>, result: &Result<()>) {
        let detail = result.as_ref().err().map(|error| format!("{error:#}"));
        if let Err(error) =
            self.session
                .append_mcp_lifecycle(action, server, result.is_ok(), detail.as_deref())
        {
            eprintln!("MCP 生命周期审计持久化失败: {error:#}");
        }
    }

    /// 返回当前 MCP Server 状态；尚未加载配置时返回 None。
    pub fn mcp_manager(&self) -> Option<&crate::mcp::McpManager> {
        self.mcp_manager.as_ref()
    }

    /// 注入事件接收器，供 TUI、Transcript 和评估系统订阅执行过程。
    pub fn set_event_sink(&mut self, sink: std::sync::Arc<dyn crate::events::EventSink>) {
        let persistent = std::sync::Arc::new(self.session.event_sink());
        let persistent_queue = QueuedEventSink::new(vec![persistent]);
        self.agent_loop
            .set_event_sink(std::sync::Arc::new(FanoutEventSink {
                sinks: vec![sink, persistent_queue],
            }));
    }

    /// 读取最近一轮报告，包括失败、取消和重试信息。
    pub fn last_turn(&self) -> Option<&TurnReport> {
        self.agent_loop.last_turn()
    }

    /// 只读访问原始记忆，用于本地诊断；不会发送模型请求。
    pub fn context(&self) -> &ContextMemory {
        &self.context
    }

    /// 返回当前 JSONL 会话文件路径。
    pub fn session_path(&self) -> &std::path::Path {
        self.session.path()
    }

    /// 返回会话目录及 context、trace、state 三个文件路径。
    pub fn session_paths(&self) -> &crate::session::SessionPaths {
        self.session.paths()
    }

    /// 读取全部或指定 Turn 的脱敏工具审计记录。
    pub fn tool_traces(&self, turn_id: Option<u64>) -> Result<Vec<crate::events::ToolTrace>> {
        self.session.tool_traces(turn_id)
    }

    /// 创建新的空会话，同时清空内存、持久化记录和累计 Token。
    pub fn new_session(&mut self) -> Result<()> {
        self.context.reset();
        self.session.clear()?;
        self.agent_loop.reset_session_state();
        Ok(())
    }

    /// 清空对话历史，但保留 Agent 的系统设定。
    pub fn reset(&mut self) {
        self.context.reset();
        if let Err(error) = self.session.clear() {
            eprintln!("清空会话持久化失败: {error:#}");
        }
    }

    /// 使用一个新的取消令牌执行完整用户 Turn。
    pub async fn ask(&mut self, input: &str) -> Result<String> {
        self.ask_with_cancel(input, &CancelToken::default()).await
    }

    /// 使用调用方提供的令牌执行 Turn，以支持协作式取消。
    pub async fn ask_with_cancel(&mut self, input: &str, cancel: &CancelToken) -> Result<String> {
        let before = self.context.messages().len();
        let result = self
            .agent_loop
            .run_turn(&mut self.context, input, cancel)
            .await;
        if result.is_ok() {
            let sources = match self
                .context
                .compact_tool_exchanges_since(before, "search_docs")
            {
                Ok(outputs) => source_refs_from_outputs(&outputs),
                Err(error) => {
                    eprintln!("RAG 上下文压缩失败（本轮仍已完成）: {error:#}");
                    Vec::new()
                }
            };
            if let Err(error) = self
                .context
                .compact_tool_exchanges_with_prefix_since(before, "mcp__")
            {
                eprintln!("MCP 上下文压缩失败（本轮仍已完成）: {error:#}");
            }
            if let Some(turn) = self.agent_loop.last_turn()
                && let Err(error) = self.session.append_turn_with_audit(
                    &self.context.messages()[before..],
                    turn.id,
                    &turn.usage,
                    &sources,
                    &turn.tool_traces,
                )
            {
                eprintln!("会话持久化失败（本轮仍已完成）: {error:#}");
            }
        } else if let Some(turn) = self.agent_loop.last_turn()
            && let Err(error) = self.session.append_tool_traces(&turn.tool_traces)
        {
            eprintln!("失败 Turn 的工具审计持久化失败: {error:#}");
        }
        result
    }
}
