use chrono::Local;
use serde_json::{Value, json};
use std::collections::HashMap;

/// 工具执行后的统一文本结果及错误标记。
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

/// 定义一个可被模型发现和调用的工具。
pub trait AgentTool: Send + Sync {
    /// 返回稳定的工具名称。
    fn name(&self) -> &str;
    /// 返回发送给模型的 OpenAI/DeepSeek 工具 Schema。
    fn schema(&self) -> Value;
    /// 校验原始 JSON 参数并执行工具。
    fn execute(&self, raw_args: &str) -> ToolOutput;
}

/// 保存工具实例，并负责 Schema 汇总和按名称分发。
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn AgentTool>>,
    max_output_chars: usize,
}

impl ToolRegistry {
    /// 创建空注册中心。
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            max_output_chars: 8_000,
        }
    }

    /// 注册或替换同名工具。
    pub fn register<T: AgentTool + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name().into(), Box::new(tool));
    }

    /// 汇总所有已注册工具的 Schema。
    pub fn schema(&self) -> Value {
        Value::Array(self.tools.values().map(|tool| tool.schema()).collect())
    }

    /// 按模型给出的名称执行工具；未知名称返回结构化错误。
    pub fn execute(&self, name: &str, raw_args: &str) -> ToolOutput {
        let output = match self.tools.get(name) {
            Some(tool) => tool.execute(raw_args),
            None => ToolOutput {
                content: format!("工具执行失败: 未知工具: {name}"),
                is_error: true,
            },
        };
        truncate_output(output, self.max_output_chars)
    }

    /// 创建带自定义工具结果字符上限的注册中心。
    pub fn with_max_output_chars(max_output_chars: usize) -> Self {
        Self {
            max_output_chars: max_output_chars.max(1),
            ..Self::default()
        }
    }
}

impl Default for ToolRegistry {
    /// 创建包含 Asteria 内置工具的默认注册中心。
    fn default() -> Self {
        let mut registry = Self::new();
        registry.register(CalculateTool);
        registry.register(CurrentTimeTool);
        registry
    }
}

/// 计算数学表达式的内置工具。
pub struct CalculateTool;
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
    fn execute(&self, raw_args: &str) -> ToolOutput {
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
    fn execute(&self, raw_args: &str) -> ToolOutput {
        execute_result(raw_args, |_| Ok(Local::now().to_rfc3339()))
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
pub fn execute(name: &str, raw_args: &str) -> ToolOutput {
    ToolRegistry::default().execute(name, raw_args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// 注册中心 Schema 与默认工具集合保持一致。
    fn default_registry_exposes_builtins() {
        let schema = ToolRegistry::default().schema();
        assert_eq!(schema.as_array().unwrap().len(), 2);
        assert!(schema.to_string().contains("calculate"));
        assert!(schema.to_string().contains("current_time"));
    }

    #[test]
    /// 未知工具必须返回错误结果而不是 panic。
    fn unknown_tool_is_structured_error() {
        let output = ToolRegistry::default().execute("missing", "{}");
        assert!(output.is_error);
        assert!(output.content.contains("未知工具"));
    }

    #[test]
    /// 参数错误会进入工具结果，便于模型在下一 Step 修正。
    fn invalid_arguments_are_tool_errors() {
        let output = ToolRegistry::default().execute("calculate", "{}");
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
        fn execute(&self, _: &str) -> ToolOutput {
            ToolOutput {
                content: "你好".repeat(5000),
                is_error: false,
            }
        }
    }

    #[test]
    /// 验证注册工具的超长结果被截断到默认 8000 字符并保留中文边界。
    fn registry_truncates_large_tool_output() {
        let mut registry = ToolRegistry::new();
        registry.register(LargeOutputTool);
        let output = registry.execute("large_output", "{}");

        assert!(!output.is_error);
        assert!(output.content.contains("工具结果已截断"));
        assert!(output.content.contains("原始 10000 字符"));
        assert!(output.content.contains("最多保留 8000 字符"));
        assert!(output.content.starts_with(&"你好".repeat(4000)));
    }
}
