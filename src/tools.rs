use crate::agent_loop::CancelToken;
use crate::permission::{ToolApprover, ToolPermission};
use chrono::Local;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// 工具执行后的统一文本结果及错误标记。
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

/// 一个并发工具批次的结果，带回原始调用位置以维持消息顺序。
pub struct ToolExecutionResult {
    pub index: usize,
    pub call_id: String,
    pub output: ToolOutput,
}

/// 工具执行时携带的运行上下文。
#[derive(Clone)]
pub struct ToolExecutionContext {
    pub turn_id: u64,
    pub call_id: String,
    pub cancel: CancelToken,
}

/// 定义一个可被模型发现和调用的工具。
#[async_trait::async_trait]
pub trait AgentTool: Send + Sync {
    /// 返回稳定的工具名称。
    fn name(&self) -> &str;
    /// 返回发送给模型的 OpenAI/DeepSeek 工具 Schema。
    fn schema(&self) -> Value;
    /// 校验原始 JSON 参数并执行工具。
    async fn execute(&self, raw_args: &str) -> ToolOutput;

    /// 带上下文的执行入口；默认兼容旧工具实现。
    async fn execute_with_context(
        &self,
        raw_args: &str,
        _context: ToolExecutionContext,
    ) -> ToolOutput {
        self.execute(raw_args).await
    }
}

/// 保存工具实例，并负责 Schema 汇总和按名称分发。
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn AgentTool>>,
    max_output_chars: usize,
    max_execution_time: Duration,
    permissions: HashMap<String, ToolPermission>,
    approver: Option<Arc<dyn ToolApprover>>,
}

impl ToolRegistry {
    /// 创建空注册中心。
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            max_output_chars: 8_000,
            max_execution_time: Duration::from_secs(30),
            permissions: HashMap::new(),
            approver: None,
        }
    }

    /// 注册或替换工具；未显式指定权限的新实例需要人工批准。
    pub fn register<T: AgentTool + 'static>(&mut self, tool: T) {
        self.register_with_permission(tool, ToolPermission::Ask);
    }

    /// 注册工具时由调用方明确设置权限，替换实例时不继承旧的授权。
    pub fn register_with_permission<T: AgentTool + 'static>(
        &mut self,
        tool: T,
        permission: ToolPermission,
    ) {
        let name = tool.name().to_owned();
        self.permissions.insert(name.clone(), permission);
        self.tools.insert(name, Box::new(tool));
    }

    /// 修改已存在工具的会话权限，未知名称返回错误。
    pub fn set_permission(&mut self, name: &str, permission: ToolPermission) -> anyhow::Result<()> {
        anyhow::ensure!(self.tools.contains_key(name), "未知工具: {name}");
        self.permissions.insert(name.to_owned(), permission);
        Ok(())
    }

    /// 按名称排序返回权限列表，供 CLI 展示。
    pub fn permissions(&self) -> Vec<(String, ToolPermission)> {
        let mut entries: Vec<_> = self
            .permissions
            .iter()
            .map(|(name, permission)| (name.clone(), *permission))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }

    /// 返回工具当前权限，供 Loop 在等待审批前发布事件。
    pub fn permission(&self, name: &str) -> Option<ToolPermission> {
        self.permissions.get(name).copied()
    }

    /// 注入异步审批处理器；没有处理器时 ask 自动拒绝。
    pub fn set_approver(&mut self, approver: Arc<dyn ToolApprover>) {
        self.approver = Some(approver);
    }

    /// 汇总所有已注册工具的 Schema。
    pub fn schema(&self) -> Value {
        Value::Array(self.tools.values().map(|tool| tool.schema()).collect())
    }

    /// 按模型给出的名称执行工具；未知名称返回结构化错误。
    pub async fn execute(&self, name: &str, raw_args: &str) -> ToolOutput {
        self.execute_call(
            name,
            raw_args,
            None,
            ToolExecutionContext {
                turn_id: 0,
                call_id: String::new(),
                cancel: CancelToken::new(),
            },
        )
        .await
    }

    /// 权限通过后才创建工具执行 Future、开始执行计时；拒绝也返回配对工具结果。
    async fn execute_call(
        &self,
        name: &str,
        raw_args: &str,
        call_id: Option<&str>,
        context: ToolExecutionContext,
    ) -> ToolOutput {
        let output = match self.tools.get(name) {
            Some(tool) => {
                if let Err(error) = validate_arguments(tool.schema(), raw_args) {
                    return truncate_output(
                        ToolOutput {
                            content: format!("工具执行失败: 参数校验失败: {error}"),
                            is_error: true,
                        },
                        self.max_output_chars,
                    );
                }
                let allowed = match self.permissions.get(name).copied().unwrap_or_default() {
                    ToolPermission::Allow => true,
                    ToolPermission::Deny => false,
                    ToolPermission::Ask => match &self.approver {
                        Some(approver) => approver.approve(name, raw_args, call_id).await,
                        None => false,
                    },
                };
                if !allowed {
                    return truncate_output(
                        ToolOutput {
                            content: format!("工具执行失败: 权限拒绝或未获批准: {name}"),
                            is_error: true,
                        },
                        self.max_output_chars,
                    );
                }
                match tokio::time::timeout(
                    self.max_execution_time,
                    tool.execute_with_context(raw_args, context),
                )
                .await
                {
                    Ok(output) => output,
                    Err(_) => ToolOutput {
                        content: format!(
                            "工具执行失败: 执行超时（超过 {} 秒）",
                            self.max_execution_time.as_secs()
                        ),
                        is_error: true,
                    },
                }
            }
            None => ToolOutput {
                content: format!("工具执行失败: 未知工具: {name}"),
                is_error: true,
            },
        };
        truncate_output(output, self.max_output_chars)
    }

    /// 并发执行一批工具，结果按模型返回的调用顺序排序。
    pub async fn execute_batch(
        &self,
        calls: &[crate::message::ToolCall],
        turn_id: u64,
        cancel: &CancelToken,
    ) -> Vec<ToolExecutionResult> {
        use futures::{StreamExt, stream::FuturesUnordered};
        let pending = FuturesUnordered::new();
        for (index, call) in calls.iter().enumerate() {
            let call_id = call.id.clone();
            pending.push(async move {
                ToolExecutionResult {
                    index,
                    call_id,
                    output: self
                        .execute_call(
                            &call.name,
                            &call.arguments,
                            Some(&call.id),
                            ToolExecutionContext {
                                turn_id,
                                call_id: call.id.clone(),
                                cancel: cancel.clone(),
                            },
                        )
                        .await,
                }
            });
        }
        let mut results = pending.collect::<Vec<_>>().await;
        results.sort_by_key(|result| result.index);
        results
    }

    /// 创建带自定义工具结果字符上限的注册中心。
    pub fn with_max_output_chars(max_output_chars: usize) -> Self {
        Self {
            max_output_chars: max_output_chars.max(1),
            ..Self::default()
        }
    }

    /// 创建带自定义工具执行时限的默认注册中心。
    pub fn with_execution_timeout(max_execution_time: Duration) -> Self {
        Self {
            max_execution_time: max_execution_time.max(Duration::from_millis(1)),
            ..Self::default()
        }
    }
}

/// 按工具 Schema 做最小通用校验，确保错误参数不会触发审批或执行副作用。
fn validate_arguments(schema: Value, raw_args: &str) -> Result<(), String> {
    let args: Value = serde_json::from_str(raw_args).map_err(|error| error.to_string())?;
    if !args.is_object() {
        return Err("参数必须是 JSON 对象".into());
    }
    if let Some(required) = schema["function"]["parameters"]["required"].as_array() {
        for field in required.iter().filter_map(Value::as_str) {
            if args.get(field).is_none() || args[field].is_null() {
                return Err(format!("缺少 {field} 参数"));
            }
        }
    }
    Ok(())
}

impl Default for ToolRegistry {
    /// 创建包含 Asteria 内置工具的默认注册中心。
    fn default() -> Self {
        let mut registry = Self::new();
        registry.register_with_permission(CalculateTool, ToolPermission::Allow);
        registry.register_with_permission(CurrentTimeTool, ToolPermission::Allow);
        registry.register_with_permission(WaitForTool, ToolPermission::Allow);
        registry
    }
}

/// 计算数学表达式的内置工具。
pub struct CalculateTool;
#[async_trait::async_trait]
impl AgentTool for CalculateTool {
    /// 返回工具名 calculate。
    fn name(&self) -> &str {
        "calculate"
    }
    /// 返回计算工具 Schema。
    fn schema(&self) -> Value {
        json!({"type":"function","function":{"name":"calculate","description":"计算一个数学表达式","parameters":{"type":"object","properties":{"expression":{"type":"string","description":"例如 (12+3)*4"}},"required":["expression"]}}})
    }
    /// 校验 expression 长度并执行表达式。
    async fn execute(&self, raw_args: &str) -> ToolOutput {
        execute_result(raw_args, |args| {
            let expression = args["expression"].as_str().ok_or("缺少 expression 参数")?;
            if expression.len() > 200 {
                return Err("表达式过长".into());
            }
            meval::eval_str(expression)
                .map(|number| number.to_string())
                .map_err(|error| error.to_string())
        })
    }
}

/// 获取本机当前时间的内置工具。
pub struct CurrentTimeTool;
#[async_trait::async_trait]
impl AgentTool for CurrentTimeTool {
    /// 返回工具名 current_time。
    fn name(&self) -> &str {
        "current_time"
    }
    /// 返回时间工具 Schema。
    fn schema(&self) -> Value {
        json!({"type":"function","function":{"name":"current_time","description":"获取运行机器的当前本地时间","parameters":{"type":"object","properties":{}}}})
    }
    /// 忽略空对象以外的字段并返回 RFC3339 本地时间。
    async fn execute(&self, raw_args: &str) -> ToolOutput {
        execute_result(raw_args, |_| Ok(Local::now().to_rfc3339()))
    }
}

/// 等待指定秒数的测试工具，用于在 TUI 中验证超时和取消。
pub struct WaitForTool;

#[async_trait::async_trait]
impl AgentTool for WaitForTool {
    /// 返回工具名 wait_for。
    fn name(&self) -> &str {
        "wait_for"
    }
    /// 返回等待工具的参数 Schema。
    fn schema(&self) -> Value {
        json!({"type":"function","function":{"name":"wait_for","description":"等待指定秒数，用于测试工具执行中的取消和超时；不要用于普通任务","parameters":{"type":"object","properties":{"seconds":{"type":"number","description":"等待秒数，范围 1 到 120"}},"required":["seconds"]}}})
    }
    /// 在等待期间保持异步挂起，并在完成后返回确认文本。
    async fn execute(&self, raw_args: &str) -> ToolOutput {
        let seconds = match serde_json::from_str::<Value>(raw_args)
            .ok()
            .and_then(|args| args["seconds"].as_f64())
        {
            Some(seconds) if (1.0..=120.0).contains(&seconds) => seconds,
            _ => {
                return ToolOutput {
                    content: "工具执行失败: seconds 必须是 1 到 120 之间的数字".into(),
                    is_error: true,
                };
            }
        };
        tokio::time::sleep(Duration::from_secs_f64(seconds)).await;
        ToolOutput {
            content: format!("已等待 {seconds:.1} 秒"),
            is_error: false,
        }
    }
}

/// 解析 JSON 后运行具体工具逻辑，并统一转换成功或失败格式。
fn execute_result<F>(raw_args: &str, execute: F) -> ToolOutput
where
    F: FnOnce(Value) -> Result<String, String>,
{
    match serde_json::from_str(raw_args)
        .map_err(|error| error.to_string())
        .and_then(execute)
    {
        Ok(content) => ToolOutput {
            content,
            is_error: false,
        },
        Err(error) => ToolOutput {
            content: format!("工具执行失败: {error}"),
            is_error: true,
        },
    }
}

/// 保留 UTF-8 字符边界并给模型明确的截断提示，避免超长结果污染上下文。
fn truncate_output(mut output: ToolOutput, max_chars: usize) -> ToolOutput {
    let length = output.content.chars().count();
    if length > max_chars {
        output.content = output.content.chars().take(max_chars).collect::<String>();
        output.content.push_str(&format!(
            "\n[工具结果已截断：原始 {length} 字符，最多保留 {max_chars} 字符]"
        ));
    }
    output
}

/// 返回默认注册中心的工具 Schema，保留旧调用入口。
pub fn schema() -> Value {
    ToolRegistry::default().schema()
}

/// 执行默认注册中心中的工具，保留旧调用入口。
pub async fn execute(name: &str, raw_args: &str) -> ToolOutput {
    ToolRegistry::default().execute(name, raw_args).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// 注册中心 Schema 与默认工具集合保持一致。
    fn default_registry_exposes_builtins() {
        let schema = ToolRegistry::default().schema();
        assert_eq!(schema.as_array().unwrap().len(), 3);
        assert!(schema.to_string().contains("calculate"));
        assert!(schema.to_string().contains("current_time"));
        assert!(schema.to_string().contains("wait_for"));
    }

    #[tokio::test]
    /// 未知工具必须返回错误结果而不是 panic。
    async fn unknown_tool_is_structured_error() {
        let output = ToolRegistry::default().execute("missing", "{}").await;
        assert!(output.is_error);
        assert!(output.content.contains("未知工具"));
    }

    #[tokio::test]
    /// 参数错误会进入工具结果，便于模型在下一 Step 修正。
    async fn invalid_arguments_are_tool_errors() {
        let output = ToolRegistry::default().execute("calculate", "{}").await;
        assert!(output.is_error);
        assert!(output.content.contains("缺少 expression"));
    }

    #[test]
    /// 验证超长结果按字符截断，不破坏中文 UTF-8，并标明原始长度。
    fn long_results_are_truncated() {
        let output = truncate_output(
            ToolOutput {
                content: "你好世界".repeat(10),
                is_error: false,
            },
            5,
        );
        assert!(output.content.starts_with("你好世界你"));
        assert!(output.content.contains("工具结果已截断"));
        assert!(!output.is_error);
    }

    /// 只用于测试注册中心：返回 10000 个中文字符的超长结果。
    struct LargeOutputTool;

    #[async_trait::async_trait]
    impl AgentTool for LargeOutputTool {
        /// 返回测试工具名称。
        fn name(&self) -> &str {
            "large_output"
        }
        /// 返回测试工具的最小 JSON Schema。
        fn schema(&self) -> Value {
            json!({"type":"function","function":{"name":"large_output","description":"返回超长测试内容","parameters":{"type":"object","properties":{}}}})
        }
        /// 生成用于触发截断逻辑的 10000 个字符。
        async fn execute(&self, _: &str) -> ToolOutput {
            ToolOutput {
                content: "你好".repeat(5000),
                is_error: false,
            }
        }
    }

    #[tokio::test]
    /// 验证注册工具的超长结果被截断到默认 8000 字符并保留中文边界。
    async fn registry_truncates_large_tool_output() {
        let mut registry = ToolRegistry::new();
        registry.register_with_permission(LargeOutputTool, ToolPermission::Allow);
        let output = registry.execute("large_output", "{}").await;

        assert!(!output.is_error);
        assert!(output.content.contains("工具结果已截断"));
        assert!(output.content.contains("原始 10000 字符"));
        assert!(output.content.contains("最多保留 8000 字符"));
        assert!(output.content.starts_with(&"你好".repeat(4000)));
    }

    /// 只用于测试超时：故意等待很久而不返回结果。
    struct SleepingTool;

    #[async_trait::async_trait]
    impl AgentTool for SleepingTool {
        /// 返回测试工具名称。
        fn name(&self) -> &str {
            "sleeping"
        }
        /// 返回测试工具 Schema。
        fn schema(&self) -> Value {
            json!({"type":"function","function":{"name":"sleeping","parameters":{"type":"object"}}})
        }
        /// 等待一段超过测试上限的时间。
        async fn execute(&self, _: &str) -> ToolOutput {
            tokio::time::sleep(Duration::from_secs(5)).await;
            ToolOutput {
                content: "不应到达".into(),
                is_error: false,
            }
        }
    }

    #[tokio::test]
    /// 验证工具超时会返回错误结果，而不是拖住 Agent Loop。
    async fn registry_times_out_slow_tool() {
        let mut registry = ToolRegistry::with_execution_timeout(Duration::from_millis(5));
        registry.register_with_permission(SleepingTool, ToolPermission::Allow);
        let output = registry.execute("sleeping", "{}").await;
        assert!(output.is_error);
        assert!(output.content.contains("执行超时"));
    }

    #[tokio::test]
    /// 验证等待工具参数范围和完成结果。
    async fn wait_tool_validates_and_completes() {
        let registry = ToolRegistry::default();
        let invalid = registry.execute("wait_for", "{\"seconds\":0}").await;
        assert!(invalid.is_error);
        let result = registry.execute("wait_for", "{\"seconds\":0.001}").await;
        assert!(result.is_error);
        let result = registry.execute("wait_for", "{\"seconds\":1}").await;
        assert!(!result.is_error);
        assert!(result.content.contains("已等待"));
    }

    /// 记录执行次数，确保被拒绝的调用连工具函数都没有进入。
    struct CountedTool(Arc<std::sync::atomic::AtomicUsize>);

    #[async_trait::async_trait]
    impl AgentTool for CountedTool {
        /// 返回计数测试工具名。
        fn name(&self) -> &str {
            "counted"
        }
        /// 返回最小测试 Schema。
        fn schema(&self) -> Value {
            json!({"type":"function","function":{"name":"counted","parameters":{"type":"object","required":["value"]}}})
        }
        /// 计数增加即表示工具真实启动。
        async fn execute(&self, args: &str) -> ToolOutput {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            ToolOutput {
                content: args.to_owned(),
                is_error: false,
            }
        }
    }

    #[tokio::test]
    /// 默认 ask 无审批器、显式 deny 均不执行；显式 allow 才执行。
    async fn permissions_gate_execution() {
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry.register(CountedTool(count.clone()));
        assert!(registry.execute("counted", r#"{"value":1}"#).await.is_error);
        registry
            .set_permission("counted", ToolPermission::Deny)
            .unwrap();
        assert!(registry.execute("counted", r#"{"value":1}"#).await.is_error);
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 0);
        registry
            .set_permission("counted", ToolPermission::Allow)
            .unwrap();
        assert!(!registry.execute("counted", r#"{"value":1}"#).await.is_error);
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(
            registry
                .set_permission("missing", ToolPermission::Allow)
                .is_err()
        );
    }

    #[tokio::test]
    /// 参数不完整时在权限检查前失败，工具执行次数保持为零。
    async fn schema_validation_precedes_permission_and_execution() {
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry.register_with_permission(CountedTool(count.clone()), ToolPermission::Ask);
        let output = registry.execute("counted", "{}").await;
        assert!(output.is_error);
        assert!(output.content.contains("缺少 value 参数"));
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    /// 批准不缓存：下一次同名调用仍需批准；审批时间不计入执行超时。
    async fn ask_is_per_call_and_outside_execution_timeout() {
        use crate::permission::ChannelApprover;
        use std::sync::atomic::Ordering;
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut registry = ToolRegistry::with_execution_timeout(Duration::from_millis(5));
        registry.register(CountedTool(count.clone()));
        let (approver, mut requests) = ChannelApprover::channel();
        registry.set_approver(Arc::new(approver));
        for (id, allowed) in [(1, true), (2, false)] {
            let (output, ()) = tokio::join!(registry.execute("counted", r#"{"value":1}"#), async {
                let request = requests.recv().await.unwrap();
                assert_eq!(request.id, id);
                assert_eq!(request.tool_name, "counted");
                assert_eq!(request.arguments, r#"{"value":1}"#);
                assert_eq!(count.load(Ordering::SeqCst), usize::from(id == 2));
                tokio::time::sleep(Duration::from_millis(20)).await;
                request.reply.send(allowed).unwrap();
            });
            assert_eq!(output.is_error, !allowed);
            assert_eq!(count.load(Ordering::SeqCst), 1);
        }
        drop(requests);
        assert!(registry.execute("counted", r#"{"value":1}"#).await.is_error);
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    /// 并发工具使用不同审批编号，乱序批准不混淆调用参数和 call_id。
    async fn parallel_approvals_are_independent() {
        use crate::{message::ToolCall, permission::ChannelApprover};
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry.register(CountedTool(count.clone()));
        let (approver, mut requests) = ChannelApprover::channel();
        registry.set_approver(Arc::new(approver));
        let calls = vec![
            ToolCall {
                id: "a".into(),
                name: "counted".into(),
                arguments: "{\"value\":1}".into(),
            },
            ToolCall {
                id: "b".into(),
                name: "counted".into(),
                arguments: "{\"value\":2}".into(),
            },
        ];
        let batch_cancel = CancelToken::new();
        let (results, ()) = tokio::join!(registry.execute_batch(&calls, 1, &batch_cancel), async {
            let a = requests.recv().await.unwrap();
            let b = requests.recv().await.unwrap();
            assert_ne!(a.id, b.id);
            assert_ne!(a.call_id, b.call_id);
            assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 0);
            let b_allowed = b.call_id.as_deref() == Some("b");
            b.reply.send(b_allowed).unwrap();
            let a_allowed = a.call_id.as_deref() == Some("b");
            a.reply.send(a_allowed).unwrap();
        });
        assert_eq!(results[0].call_id, "a");
        assert!(results[0].output.is_error);
        assert_eq!(results[1].call_id, "b");
        assert!(!results[1].output.is_error);
        assert_eq!(results[1].output.content, calls[1].arguments);
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
