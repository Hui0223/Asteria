use anyhow::Result;
use asteria_agent::{
    agent_loop::{AgentLoop, CancelToken, LoopConfig, TurnState},
    context::ContextMemory,
    context_builder::PreparedContext,
    message::{Message, ToolCall},
    provider::{AssistantTurn, ModelProvider, TokenUsage},
    retry::RetryPolicy,
};
use serde_json::Value;
use std::{cell::RefCell, collections::VecDeque, time::Duration};

/// 按脚本返回响应，并捕获每一次实际传入模型接口的上下文。
struct Scripted {
    responses: RefCell<VecDeque<Result<AssistantTurn>>>,
    seen: RefCell<Vec<Vec<Message>>>,
}
impl ModelProvider for Scripted {
    /// 返回离线测试模型名。
    fn model(&self) -> &str {
        "scripted"
    }
    /// 记录模型可见消息，再返回一个预设成功响应或连接错误。
    async fn complete(&self, context: &PreparedContext, _: Value) -> Result<AssistantTurn> {
        self.seen.borrow_mut().push(context.messages().to_vec());
        self.responses
            .borrow_mut()
            .pop_front()
            .expect("unexpected attempt")
    }
}
/// 构造可重试的连接重置错误。
fn disconnected() -> Result<AssistantTurn> {
    Err(std::io::Error::from(std::io::ErrorKind::ConnectionReset).into())
}
/// 构造缺少 expression 参数的真实工具调用，响应携带模拟用量。
fn bad_tool() -> Result<AssistantTurn> {
    Ok(AssistantTurn {
        content: None,
        tool_calls: vec![ToolCall {
            id: "bad-1".into(),
            name: "calculate".into(),
            arguments: "{}".into(),
        }],
        usage: Some(TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 2,
            total_tokens: 12,
        }),
    })
}
/// 执行场景并校验模型可见上下文、整轮回滚、重试次数及已报告用量。
async fn run_case(
    case: &str,
    responses: Vec<Result<AssistantTurn>>,
    success: bool,
    tool_error_visible: bool,
    tokens: usize,
) {
    let expected_attempts = responses.len();
    let provider = Scripted {
        responses: RefCell::new(responses.into()),
        seen: RefCell::new(Vec::new()),
    };
    // 通过共享引用观察 provider 的输入；适配器只负责转发。
    let mut engine = AgentLoop::new(Borrowed(&provider), LoopConfig::default());
    engine
        .set_retry_policy(RetryPolicy {
            base_delay: Duration::ZERO,
            ..Default::default()
        })
        .unwrap();
    let mut memory = ContextMemory::new("system");
    memory.append_user("旧问题").unwrap();
    memory
        .append_assistant(Some("旧回答".into()), vec![])
        .unwrap();
    let before = memory.messages().to_vec();
    let result = engine
        .run_turn(&mut memory, "计算测试", &CancelToken::default())
        .await;
    assert_eq!(result.is_ok(), success);
    let seen = provider.seen.borrow();
    assert_eq!(seen.len(), expected_attempts);
    for (index, messages) in seen.iter().enumerate() {
        assert_eq!(&messages[..2], before.as_slice());
        let errors: Vec<_> = messages
            .iter()
            .filter(|m| matches!(m, Message::Tool { is_error: true, .. }))
            .collect();
        assert_eq!(errors.len(), usize::from(tool_error_visible && index > 0));
        if let Some(Message::Tool {
            call_id, content, ..
        }) = errors.first().copied()
        {
            assert_eq!(call_id, "bad-1");
            assert_eq!(content, "工具执行失败: 缺少 expression 参数");
        }
        // 网络错误不会增加任何消息：每次重试看到的消息完全一致。
        if index > usize::from(tool_error_visible) {
            assert_eq!(messages, &seen[index - 1]);
        }
    }
    if !success {
        assert_eq!(memory.messages(), before);
    } else {
        assert_eq!(memory.messages().len(), 6);
        memory.validate().unwrap();
    }
    let report = engine.last_turn().unwrap();
    assert_eq!(
        report.state,
        if success {
            TurnState::Completed
        } else {
            TurnState::Failed
        }
    );
    assert_eq!(report.retries, if success { 0 } else { 2 });
    assert_eq!(report.usage.total_tokens, tokens);
    assert_eq!(engine.session_usage().total_tokens, tokens);
    println!(
        "{case}: attempts={} context_per_attempt={:?} final_messages={} state={:?} turn_tokens={} session_tokens={}",
        seen.len(),
        seen.iter().map(Vec::len).collect::<Vec<_>>(),
        memory.messages().len(),
        report.state,
        report.usage.total_tokens,
        engine.session_usage().total_tokens
    );
}
/// 保持对预设模型的只读借用，便于执行后检查捕获的请求。
struct Borrowed<'a>(&'a Scripted);
impl ModelProvider for Borrowed<'_> {
    /// 代理模型名称。
    fn model(&self) -> &str {
        self.0.model()
    }
    /// 代理模型请求。
    async fn complete(&self, context: &PreparedContext, tools: Value) -> Result<AssistantTurn> {
        self.0.complete(context, tools).await
    }
}
#[tokio::test]
/// 三次连接失败：错误不入上下文，本轮回滚，旧历史保留。
async fn connection_errors_are_not_messages() {
    run_case(
        "Case1",
        vec![disconnected(), disconnected(), disconnected()],
        false,
        false,
        0,
    )
    .await;
}
#[tokio::test]
/// 工具错误可被下一 Step 读取，整轮成功后留在历史中。
async fn tool_error_is_visible_to_model() {
    run_case(
        "Case2",
        vec![
            bad_tool(),
            Ok(AssistantTurn {
                content: Some("缺少 expression 参数，请补充".into()),
                tool_calls: vec![],
                usage: None,
            }),
        ],
        true,
        true,
        12,
    )
    .await;
}
#[tokio::test]
/// 工具失败后请求耗尽：本轮全部回滚，但已返回的用量不会撤销。
async fn rollback_removes_tool_error_but_keeps_usage() {
    run_case(
        "Case3",
        vec![bad_tool(), disconnected(), disconnected(), disconnected()],
        false,
        true,
        12,
    )
    .await;
}
