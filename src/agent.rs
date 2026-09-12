use crate::{
    agent_loop::{AgentLoop, CancelToken, LoopConfig, TurnReport},
    context::ContextMemory,
    provider::TokenUsage,
    provider::deepseek::DeepSeekProvider,
    session::SessionStore,
};
use anyhow::Result;

const SYSTEM: &str = "你是 Asteria，一个可靠、简洁的中文 AI 助手。需要精确计算或当前时间时调用工具，不要猜测工具结果。";

/// 对外提供简单问答接口，并组合上下文与 Agent Loop。
pub struct Asteria {
    agent_loop: AgentLoop<DeepSeekProvider>,
    context: ContextMemory,
    session: SessionStore,
}

impl Asteria {
    /// 根据环境变量创建 DeepSeek Agent，并采用默认 Loop 配置。
    pub fn new() -> Result<Self> {
        let session = SessionStore::from_env()?;
        let restored = session.restore(SYSTEM)?;
        let mut agent_loop = AgentLoop::new(DeepSeekProvider::from_env()?, LoopConfig::default());
        agent_loop.restore_session_state(restored.next_turn_id, restored.usage);
        for (tool, permission) in restored.permissions {
            agent_loop.set_tool_permission(&tool, permission).ok();
        }
        Ok(Self {
            agent_loop,
            context: restored.context,
            session,
        })
    }

    /// 返回当前 Agent 使用的模型名称。
    pub fn model(&self) -> &str {
        self.agent_loop.model()
    }

    /// 将请求重试提示交给终端统一绘制。
    pub fn set_retry_output(&mut self, output: impl Fn(String) + Send + Sync + 'static) {
        self.agent_loop.set_retry_output(output);
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

    /// 注入事件接收器，供 TUI、Transcript 和评估系统订阅执行过程。
    pub fn set_event_sink(&mut self, sink: std::sync::Arc<dyn crate::events::EventSink>) {
        self.agent_loop.set_event_sink(sink);
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
        if result.is_ok()
            && let Some(turn) = self.agent_loop.last_turn()
            && let Err(error) =
                self.session
                    .append_turn(&self.context.messages()[before..], turn.id, &turn.usage)
        {
            eprintln!("会话持久化失败（本轮仍已完成）: {error:#}");
        }
        result
    }
}
