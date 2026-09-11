use crate::{
    context::ContextMemory,
    context_builder::{ContextBuilder, ContextPolicy, HeuristicTokenEstimator},
    provider::{ModelProvider, TokenUsage},
    tools,
};
use anyhow::{Result, bail};
/// 可克隆的异步取消信号，取消后唤醒所有 cancelled() 等待者。
pub use tokio_util::sync::CancellationToken as CancelToken;

/// 一个用户 Turn 在 Agent Loop 中可能处于的状态。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnState {
    Running,
    WaitingForTools,
    Completed,
    Failed,
    Cancelled,
}

/// 保存一个 Turn 的身份、执行步数、状态轨迹和最终答案。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnReport {
    pub id: u64,
    pub steps: usize,
    pub state: TurnState,
    pub state_history: Vec<TurnState>,
    pub answer: Option<String>,
    pub latest_context_tokens: usize,
    pub context_truncated: bool,
    pub usage: TokenUsage,
    pub retries: usize,
}

/// 控制单个 Turn 最多允许多少次“模型思考 → 工具处理”的 Step。
#[derive(Clone, Copy, Debug)]
pub struct LoopConfig {
    pub max_steps_per_turn: usize,
}

impl Default for LoopConfig {
    /// 提供安全的默认 Step 上限，防止模型无限调用工具。
    fn default() -> Self {
        Self {
            max_steps_per_turn: 8,
        }
    }
}

/// 驱动一个可替换模型供应商，并显式管理 Turn 与 Step。
pub struct AgentLoop<P> {
    provider: P,
    config: LoopConfig,
    next_turn_id: u64,
    last_turn: Option<TurnReport>,
    context_builder: ContextBuilder<HeuristicTokenEstimator>,
    session_usage: TokenUsage,
    retry_policy: crate::retry::RetryPolicy,
    retry_output: Option<Box<dyn Fn(String) + Send + Sync>>,
    tool_registry: tools::ToolRegistry,
}

impl<P: ModelProvider> AgentLoop<P> {
    /// 使用给定供应商和循环配置创建 Agent Loop。
    pub fn new(provider: P, config: LoopConfig) -> Self {
        Self::with_context_policy(provider, config, ContextPolicy::default())
    }

    /// 使用自定义上下文预算创建 Agent Loop，便于按模型能力调整窗口。
    pub fn with_context_policy(
        provider: P,
        config: LoopConfig,
        context_policy: ContextPolicy,
    ) -> Self {
        Self {
            provider,
            config,
            next_turn_id: 1,
            last_turn: None,
            context_builder: ContextBuilder::new(context_policy, HeuristicTokenEstimator),
            session_usage: TokenUsage::default(),
            retry_policy: crate::retry::RetryPolicy::default(),
            tool_registry: tools::ToolRegistry::default(),
            retry_output: None,
        }
    }

    /// 返回底层供应商实际使用的模型名称。
    pub fn model(&self) -> &str {
        self.provider.model()
    }

    /// 注入重试信息的输出入口，使 CLI 能在重绘输入行时安全显示日志。
    pub fn set_retry_output(&mut self, output: impl Fn(String) + Send + Sync + 'static) {
        self.retry_output = Some(Box::new(output));
    }

    /// 设置请求重试策略，拒绝没有首次尝试的配置。
    pub fn set_retry_policy(&mut self, policy: crate::retry::RetryPolicy) -> Result<()> {
        anyhow::ensure!(policy.max_attempts > 0, "max_attempts 必须大于零");
        self.retry_policy = policy;
        Ok(())
    }

    /// 返回最近一个 Turn 的执行报告，便于观测和测试状态变化。
    pub fn last_turn(&self) -> Option<&TurnReport> {
        self.last_turn.as_ref()
    }

    /// 返回当前进程内所有已完成或部分执行 Turn 的累计 Token 使用量。
    pub fn session_usage(&self) -> &TokenUsage {
        &self.session_usage
    }

    /// 异步执行 Turn；取消请发信号并等待本方法结束，以完成回滚，勿直接丢弃此 Future。
    pub async fn run_turn(
        &mut self,
        context: &mut ContextMemory,
        input: &str,
        cancel: &CancelToken,
    ) -> Result<String> {
        let checkpoint = context.checkpoint();
        self.begin_turn();
        let result = self.run_steps(context, input, cancel).await;
        if result.is_err() {
            context.rollback(checkpoint);
            if self.current_state() != TurnState::Cancelled {
                self.transition(TurnState::Failed);
            }
        }
        result
    }

    /// 分配唯一 Turn ID，并初始化该 Turn 的状态轨迹。
    fn begin_turn(&mut self) {
        let id = self.next_turn_id;
        self.next_turn_id += 1;
        self.last_turn = Some(TurnReport {
            id,
            steps: 0,
            state: TurnState::Running,
            state_history: vec![TurnState::Running],
            answer: None,
            latest_context_tokens: 0,
            context_truncated: false,
            usage: TokenUsage::default(),
            retries: 0,
        });
    }

    /// 逐 Step 请求模型、执行工具，直到得到最终答案或达到上限。
    async fn run_steps(
        &mut self,
        context: &mut ContextMemory,
        input: &str,
        cancel: &CancelToken,
    ) -> Result<String> {
        self.ensure_not_cancelled(cancel)?;
        context.append_user(input)?;

        for _ in 0..self.config.max_steps_per_turn {
            self.ensure_not_cancelled(cancel)?;
            self.increment_steps();
            let tool_schema = self.tool_registry.schema();
            let prepared = self.context_builder.prepare(context, &tool_schema)?;
            self.record_prepared_context(prepared.estimated_tokens(), prepared.truncated());
            let mut attempt = 1;
            let message = loop {
                self.ensure_not_cancelled(cancel)?;
                // 优先接收已就绪的响应以记录 usage，然后再检查取消。
                let response = tokio::select! {
                    biased;
                    result = self.provider.complete(&prepared, tool_schema.clone()) => Some(result),
                    _ = cancel.cancelled() => None,
                };
                let Some(response) = response else {
                    self.transition(TurnState::Cancelled);
                    bail!("当前 Turn 已取消");
                };
                match response {
                    Ok(message) => break message,
                    Err(error) => {
                        self.ensure_not_cancelled(cancel)?;
                        if attempt >= self.retry_policy.max_attempts
                            || !crate::retry::is_retryable(&error)
                        {
                            return Err(error);
                        }
                        let delay = self.retry_policy.delay(attempt);
                        if let Some(turn) = &mut self.last_turn {
                            turn.retries += 1;
                        }
                        let notice = format!(
                            "[Retry] attempt={}/{} delay={}ms",
                            attempt + 1,
                            self.retry_policy.max_attempts,
                            delay.as_millis()
                        );
                        if let Some(output) = &self.retry_output {
                            output(notice);
                        } else {
                            eprintln!("{notice}");
                        }
                        if !crate::retry::wait(delay, cancel).await {
                            self.ensure_not_cancelled(cancel)?;
                        }
                        attempt += 1;
                    }
                }
            };
            if let Some(usage) = &message.usage {
                self.record_usage(usage);
            }
            self.ensure_not_cancelled(cancel)?;
            let calls = message.tool_calls;
            context.append_assistant(message.content.clone(), calls.clone())?;

            if calls.is_empty() {
                let answer = message.content.unwrap_or_default();
                self.complete(answer.clone());
                return Ok(answer);
            }

            self.transition(TurnState::WaitingForTools);
            for call in calls {
                self.ensure_not_cancelled(cancel)?;
                let output = tokio::select! {
                    output = self.tool_registry.execute(&call.name, &call.arguments) => output,
                    _ = cancel.cancelled() => {
                        self.transition(TurnState::Cancelled);
                        bail!("当前 Turn 已取消");
                    }
                };
                context.append_tool_result(call.id, output.content, output.is_error)?;
            }
            self.transition(TurnState::Running);
        }
        bail!("工具调用 Step 过多，已停止")
    }

    /// 在取消信号出现时记录 Cancelled 状态并中止当前 Turn。
    fn ensure_not_cancelled(&mut self, cancel: &CancelToken) -> Result<()> {
        if cancel.is_cancelled() {
            self.transition(TurnState::Cancelled);
            bail!("当前 Turn 已取消");
        }
        Ok(())
    }

    /// 将当前 Turn 的 Step 计数加一。
    fn increment_steps(&mut self) {
        if let Some(turn) = &mut self.last_turn {
            turn.steps += 1;
        }
    }

    /// 记录当前 Step 发送给模型的估算 Token 数和裁剪状态。
    fn record_prepared_context(&mut self, estimated_tokens: usize, truncated: bool) {
        if let Some(turn) = &mut self.last_turn {
            turn.latest_context_tokens = estimated_tokens;
            turn.context_truncated |= truncated;
        }
    }

    /// 累加当前 Step 的真实 Token 使用量，供整个 Turn 统计。
    fn record_usage(&mut self, usage: &TokenUsage) {
        if let Some(turn) = &mut self.last_turn {
            turn.usage.prompt_tokens += usage.prompt_tokens;
            turn.usage.completion_tokens += usage.completion_tokens;
            turn.usage.total_tokens += usage.total_tokens;
        }
        self.session_usage.prompt_tokens += usage.prompt_tokens;
        self.session_usage.completion_tokens += usage.completion_tokens;
        self.session_usage.total_tokens += usage.total_tokens;
    }

    /// 更新当前 Turn 状态，并保留去重后的状态变化轨迹。
    fn transition(&mut self, state: TurnState) {
        if let Some(turn) = &mut self.last_turn {
            turn.state = state;
            if turn.state_history.last() != Some(&state) {
                turn.state_history.push(state);
            }
        }
    }

    /// 记录最终答案并把当前 Turn 标记为完成。
    fn complete(&mut self, answer: String) {
        if let Some(turn) = &mut self.last_turn {
            turn.answer = Some(answer);
        }
        self.transition(TurnState::Completed);
    }

    /// 返回当前 Turn 状态；仅用于已经开始执行的内部流程。
    fn current_state(&self) -> TurnState {
        self.last_turn
            .as_ref()
            .map_or(TurnState::Failed, |turn| turn.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context_builder::PreparedContext, message::ToolCall, provider::AssistantTurn};
    use anyhow::{Context, Result};
    use serde_json::Value;
    use std::{cell::RefCell, collections::VecDeque};

    /// 按预设顺序返回响应，让 Loop 测试不依赖真实网络。
    struct FakeProvider {
        responses: RefCell<VecDeque<AssistantTurn>>,
    }

    impl FakeProvider {
        /// 用一组预设响应创建假模型。
        fn new(responses: Vec<AssistantTurn>) -> Self {
            Self {
                responses: RefCell::new(responses.into()),
            }
        }
    }

    impl ModelProvider for FakeProvider {
        /// 返回固定模型名，满足 Agent Loop 接口。
        fn model(&self) -> &str {
            "fake-model"
        }

        /// 弹出下一个预设响应，模拟模型完成一个 Step。
        async fn complete(
            &self,
            _context: &PreparedContext,
            _tools: Value,
        ) -> Result<AssistantTurn> {
            self.responses
                .borrow_mut()
                .pop_front()
                .context("假模型没有更多响应")
        }
    }

    /// 构造一个不调用工具、直接回答的模型响应。
    fn answer(content: &str) -> AssistantTurn {
        AssistantTurn {
            content: Some(content.into()),
            tool_calls: Vec::new(),
            usage: None,
        }
    }

    /// 构造一个调用计算器的模型响应。
    fn calculate_call(id: &str) -> AssistantTurn {
        AssistantTurn {
            content: None,
            tool_calls: vec![ToolCall {
                id: id.into(),
                name: "calculate".into(),
                arguments: "{\"expression\":\"1+1\"}".into(),
            }],
            usage: None,
        }
    }

    #[tokio::test]
    /// 验证无需工具的 Turn 只包含一个 Step 并正常完成。
    async fn completes_plain_turn_in_one_step() {
        let provider = FakeProvider::new(vec![answer("hello")]);
        let mut agent_loop = AgentLoop::new(provider, LoopConfig::default());
        let mut context = ContextMemory::new("system");

        assert_eq!(
            agent_loop
                .run_turn(&mut context, "hi", &CancelToken::default())
                .await
                .unwrap(),
            "hello"
        );
        let report = agent_loop.last_turn().unwrap();
        assert_eq!(report.steps, 1);
        assert_eq!(report.state, TurnState::Completed);
    }

    #[tokio::test]
    /// 验证一次工具调用会产生两个 Step 和完整状态轨迹。
    async fn executes_tool_between_two_steps() {
        let provider = FakeProvider::new(vec![calculate_call("a"), answer("2")]);
        let mut agent_loop = AgentLoop::new(provider, LoopConfig::default());
        let mut context = ContextMemory::new("system");

        agent_loop
            .run_turn(&mut context, "1+1", &CancelToken::default())
            .await
            .unwrap();
        let report = agent_loop.last_turn().unwrap();
        assert_eq!(report.steps, 2);
        assert_eq!(
            report.state_history,
            vec![
                TurnState::Running,
                TurnState::WaitingForTools,
                TurnState::Running,
                TurnState::Completed
            ]
        );
    }

    #[tokio::test]
    /// 验证超过 Step 上限会失败，并回滚该 Turn 写入的全部消息。
    async fn rolls_back_when_step_limit_is_reached() {
        let provider = FakeProvider::new(vec![calculate_call("a")]);
        let mut agent_loop = AgentLoop::new(
            provider,
            LoopConfig {
                max_steps_per_turn: 1,
            },
        );
        let mut context = ContextMemory::new("system");

        assert!(
            agent_loop
                .run_turn(&mut context, "keep calling", &CancelToken::default())
                .await
                .is_err()
        );
        assert!(context.messages().is_empty());
        assert_eq!(agent_loop.last_turn().unwrap().state, TurnState::Failed);
    }

    #[tokio::test]
    /// 验证取消的 Turn 不污染上下文，并留下 Cancelled 状态。
    async fn cancellation_rolls_back_turn() {
        let provider = FakeProvider::new(vec![answer("unused")]);
        let mut agent_loop = AgentLoop::new(provider, LoopConfig::default());
        let mut context = ContextMemory::new("system");
        let cancel = CancelToken::default();
        cancel.cancel();

        assert!(
            agent_loop
                .run_turn(&mut context, "hi", &cancel)
                .await
                .is_err()
        );
        assert!(context.messages().is_empty());
        assert_eq!(agent_loop.last_turn().unwrap().state, TurnState::Cancelled);
    }

    #[tokio::test]
    /// 验证每个新 Turn 都会获得不同且递增的 ID。
    async fn assigns_unique_turn_ids() {
        let provider = FakeProvider::new(vec![answer("one"), answer("two")]);
        let mut agent_loop = AgentLoop::new(provider, LoopConfig::default());
        let mut context = ContextMemory::new("system");

        agent_loop
            .run_turn(&mut context, "first", &CancelToken::default())
            .await
            .unwrap();
        let first_id = agent_loop.last_turn().unwrap().id;
        agent_loop
            .run_turn(&mut context, "second", &CancelToken::default())
            .await
            .unwrap();
        assert_eq!(agent_loop.last_turn().unwrap().id, first_id + 1);
    }

    #[tokio::test]
    /// 验证每个 Step 的 usage 同时累计到 Turn 和整个 Session。
    async fn accumulates_session_usage_across_turns() {
        let mut first = answer("one");
        first.usage = Some(TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 2,
            total_tokens: 12,
        });
        let mut second = answer("two");
        second.usage = Some(TokenUsage {
            prompt_tokens: 20,
            completion_tokens: 3,
            total_tokens: 23,
        });
        let provider = FakeProvider::new(vec![first, second]);
        let mut agent_loop = AgentLoop::new(provider, LoopConfig::default());
        let mut context = ContextMemory::new("system");

        agent_loop
            .run_turn(&mut context, "first", &CancelToken::default())
            .await
            .unwrap();
        assert_eq!(agent_loop.last_turn().unwrap().usage.total_tokens, 12);
        agent_loop
            .run_turn(&mut context, "second", &CancelToken::default())
            .await
            .unwrap();
        assert_eq!(agent_loop.last_turn().unwrap().usage.total_tokens, 23);
        assert_eq!(agent_loop.session_usage().total_tokens, 35);
    }

    /// 模拟前两次连接失败，第三次成功或继续失败的模型。
    struct FlakyProvider {
        calls: std::cell::Cell<usize>,
        succeed: bool,
    }
    impl ModelProvider for FlakyProvider {
        /// 返回测试模型名称。
        fn model(&self) -> &str {
            "flaky"
        }
        /// 返回可重试连接错误或带用量的答案。
        async fn complete(&self, _: &PreparedContext, _: Value) -> Result<AssistantTurn> {
            self.calls.set(self.calls.get() + 1);
            if self.calls.get() < 3 || !self.succeed {
                return Err(std::io::Error::from(std::io::ErrorKind::ConnectionReset).into());
            }
            let mut result = answer("done");
            result.usage = Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 2,
                total_tokens: 12,
            });
            Ok(result)
        }
    }

    #[tokio::test]
    /// 重试不增加 Step，不重复写消息，用量只累计一次；耗尽后回滚。
    async fn retries_preserve_step_context_and_usage() {
        for succeed in [true, false] {
            let mut engine = AgentLoop::new(
                FlakyProvider {
                    calls: std::cell::Cell::new(0),
                    succeed,
                },
                LoopConfig::default(),
            );
            engine
                .set_retry_policy(crate::retry::RetryPolicy {
                    base_delay: std::time::Duration::ZERO,
                    ..Default::default()
                })
                .unwrap();
            let mut memory = ContextMemory::new("system");
            let result = engine
                .run_turn(&mut memory, "hi", &CancelToken::default())
                .await;
            assert_eq!(result.is_ok(), succeed);
            assert_eq!(engine.provider.calls.get(), 3);
            assert_eq!(engine.last_turn().unwrap().steps, 1);
            assert_eq!(engine.last_turn().unwrap().retries, 2);
            assert_eq!(memory.messages().len(), if succeed { 2 } else { 0 });
            assert_eq!(
                engine.session_usage().total_tokens,
                if succeed { 12 } else { 0 }
            );
        }
    }

    /// 第一 Step 完成工具调用；第二 Step 挂起或失败，后续新 Turn 正常回答。
    struct InterruptibleProvider {
        calls: std::cell::Cell<usize>,
        waiting: std::sync::Arc<tokio::sync::Notify>,
        retry: bool,
    }

    impl ModelProvider for InterruptibleProvider {
        /// 返回取消测试使用的模型名。
        fn model(&self) -> &str {
            "interruptible"
        }

        /// 用通知精确标记进入请求或退避的位置，避免测试依赖网络速度。
        async fn complete(&self, _: &PreparedContext, _: Value) -> Result<AssistantTurn> {
            let call = self.calls.get() + 1;
            self.calls.set(call);
            if call == 1 {
                let mut response = calculate_call("completed-tool");
                response.usage = Some(TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 2,
                    total_tokens: 12,
                });
                return Ok(response);
            }
            if call == 2 {
                self.waiting.notify_one();
                if self.retry {
                    return Err(std::io::Error::from(std::io::ErrorKind::ConnectionReset).into());
                }
                return std::future::pending().await;
            }
            Ok(answer("下一轮正常"))
        }
    }

    /// 验证取消后回滚完整本轮、保留已有用量、不会重试被取消的请求且下一轮正常。
    async fn check_interrupt(retry: bool) {
        let waiting = std::sync::Arc::new(tokio::sync::Notify::new());
        let provider = InterruptibleProvider {
            calls: std::cell::Cell::new(0),
            waiting: waiting.clone(),
            retry,
        };
        let mut engine = AgentLoop::new(provider, LoopConfig::default());
        engine
            .set_retry_policy(crate::retry::RetryPolicy {
                base_delay: std::time::Duration::from_secs(8),
                ..Default::default()
            })
            .unwrap();
        let mut memory = ContextMemory::new("system");
        memory.append_user("旧问题").unwrap();
        memory
            .append_assistant(Some("旧回答".into()), vec![])
            .unwrap();
        let original = memory.messages().to_vec();
        let cancel = CancelToken::new();
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            let (result, ()) = tokio::join!(engine.run_turn(&mut memory, "计算", &cancel), async {
                waiting.notified().await;
                cancel.cancel();
            });
            result
        })
        .await
        .expect("取消必须立即唤醒请求/退避");
        assert!(result.is_err());
        let report = engine.last_turn().unwrap();
        assert_eq!(report.state, TurnState::Cancelled);
        assert_eq!(report.steps, 2);
        assert_eq!(report.retries, usize::from(retry));
        assert_eq!(report.usage.total_tokens, 12);
        assert_eq!(engine.session_usage().total_tokens, 12);
        assert_eq!(engine.provider.calls.get(), 2);
        assert_eq!(memory.messages(), original);
        let next = engine
            .run_turn(&mut memory, "继续", &CancelToken::new())
            .await
            .unwrap();
        assert_eq!(next, "下一轮正常");
        assert_eq!(engine.last_turn().unwrap().state, TurnState::Completed);
        assert_eq!(engine.last_turn().unwrap().id, 2);
        assert_eq!(engine.session_usage().total_tokens, 12);
        assert_eq!(memory.messages().len(), original.len() + 2);
    }

    #[tokio::test]
    /// 正在等待模型时取消，完成回滚后可继续下一轮。
    async fn cancels_pending_request_and_resumes() {
        check_interrupt(false).await;
    }

    #[tokio::test]
    /// 在 8 秒退避中取消，不等待完整退避，也不会再发一次请求。
    async fn cancels_backoff_and_resumes() {
        check_interrupt(true).await;
    }

    /// 在响应就绪的同一时刻发送取消信号。
    struct ReadyCancelledProvider(CancelToken);
    impl ModelProvider for ReadyCancelledProvider {
        /// 返回竞态测试模型名称。
        fn model(&self) -> &str {
            "ready-cancelled"
        }
        /// 返回带 usage 的响应，同时触发取消。
        async fn complete(&self, _: &PreparedContext, _: Value) -> Result<AssistantTurn> {
            self.0.cancel();
            let mut response = answer("不应写入历史");
            response.usage = Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 2,
                total_tokens: 12,
            });
            Ok(response)
        }
    }

    #[tokio::test]
    /// 已收到响应的 usage 在取消时仍保留，但正文不写入上下文。
    async fn cancellation_keeps_ready_response_usage() {
        let cancel = CancelToken::new();
        let mut engine = AgentLoop::new(
            ReadyCancelledProvider(cancel.clone()),
            LoopConfig::default(),
        );
        let mut memory = ContextMemory::new("system");
        assert!(engine.run_turn(&mut memory, "hi", &cancel).await.is_err());
        assert_eq!(engine.last_turn().unwrap().state, TurnState::Cancelled);
        assert_eq!(engine.session_usage().total_tokens, 12);
        assert!(memory.messages().is_empty());
    }
}
