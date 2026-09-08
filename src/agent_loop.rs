use crate::{
    context::ContextMemory,
    context_builder::{ContextBuilder, ContextPolicy, HeuristicTokenEstimator},
    provider::ModelProvider,
    tools,
};
use anyhow::{Result, bail};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

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

/// 可在线程之间共享的协作式取消信号。
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// 发出取消请求；Loop 会在下一个模型或工具边界停止。
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// 检查调用方是否已经发出取消请求。
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// 驱动一个可替换模型供应商，并显式管理 Turn 与 Step。
pub struct AgentLoop<P> {
    provider: P,
    config: LoopConfig,
    next_turn_id: u64,
    last_turn: Option<TurnReport>,
    context_builder: ContextBuilder<HeuristicTokenEstimator>,
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
        }
    }

    /// 返回底层供应商实际使用的模型名称。
    pub fn model(&self) -> &str {
        self.provider.model()
    }

    /// 返回最近一个 Turn 的执行报告，便于观测和测试状态变化。
    pub fn last_turn(&self) -> Option<&TurnReport> {
        self.last_turn.as_ref()
    }

    /// 执行一个完整用户 Turn；失败或取消时自动回滚本 Turn 的上下文。
    pub fn run_turn(
        &mut self,
        context: &mut ContextMemory,
        input: &str,
        cancel: &CancelToken,
    ) -> Result<String> {
        let checkpoint = context.checkpoint();
        self.begin_turn();
        let result = self.run_steps(context, input, cancel);
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
        });
    }

    /// 逐 Step 请求模型、执行工具，直到得到最终答案或达到上限。
    fn run_steps(
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
            let tool_schema = tools::schema();
            let prepared = self.context_builder.prepare(context, &tool_schema)?;
            self.record_prepared_context(prepared.estimated_tokens(), prepared.truncated());
            let message = self.provider.complete(&prepared, tool_schema)?;
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
                let output = tools::execute(&call.name, &call.arguments);
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
        fn complete(&self, _context: &PreparedContext, _tools: Value) -> Result<AssistantTurn> {
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
        }
    }

    #[test]
    /// 验证无需工具的 Turn 只包含一个 Step 并正常完成。
    fn completes_plain_turn_in_one_step() {
        let provider = FakeProvider::new(vec![answer("hello")]);
        let mut agent_loop = AgentLoop::new(provider, LoopConfig::default());
        let mut context = ContextMemory::new("system");

        assert_eq!(
            agent_loop
                .run_turn(&mut context, "hi", &CancelToken::default())
                .unwrap(),
            "hello"
        );
        let report = agent_loop.last_turn().unwrap();
        assert_eq!(report.steps, 1);
        assert_eq!(report.state, TurnState::Completed);
    }

    #[test]
    /// 验证一次工具调用会产生两个 Step 和完整状态轨迹。
    fn executes_tool_between_two_steps() {
        let provider = FakeProvider::new(vec![calculate_call("a"), answer("2")]);
        let mut agent_loop = AgentLoop::new(provider, LoopConfig::default());
        let mut context = ContextMemory::new("system");

        agent_loop
            .run_turn(&mut context, "1+1", &CancelToken::default())
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

    #[test]
    /// 验证超过 Step 上限会失败，并回滚该 Turn 写入的全部消息。
    fn rolls_back_when_step_limit_is_reached() {
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
                .is_err()
        );
        assert!(context.messages().is_empty());
        assert_eq!(agent_loop.last_turn().unwrap().state, TurnState::Failed);
    }

    #[test]
    /// 验证取消的 Turn 不污染上下文，并留下 Cancelled 状态。
    fn cancellation_rolls_back_turn() {
        let provider = FakeProvider::new(vec![answer("unused")]);
        let mut agent_loop = AgentLoop::new(provider, LoopConfig::default());
        let mut context = ContextMemory::new("system");
        let cancel = CancelToken::default();
        cancel.cancel();

        assert!(agent_loop.run_turn(&mut context, "hi", &cancel).is_err());
        assert!(context.messages().is_empty());
        assert_eq!(agent_loop.last_turn().unwrap().state, TurnState::Cancelled);
    }

    #[test]
    /// 验证每个新 Turn 都会获得不同且递增的 ID。
    fn assigns_unique_turn_ids() {
        let provider = FakeProvider::new(vec![answer("one"), answer("two")]);
        let mut agent_loop = AgentLoop::new(provider, LoopConfig::default());
        let mut context = ContextMemory::new("system");

        agent_loop
            .run_turn(&mut context, "first", &CancelToken::default())
            .unwrap();
        let first_id = agent_loop.last_turn().unwrap().id;
        agent_loop
            .run_turn(&mut context, "second", &CancelToken::default())
            .unwrap();
        assert_eq!(agent_loop.last_turn().unwrap().id, first_id + 1);
    }
}
