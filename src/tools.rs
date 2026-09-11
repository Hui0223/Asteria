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
}

impl ToolRegistry {
    /// 创建空注册中心。
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
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
        match self.tools.get(name) {
            Some(tool) => tool.execute(raw_args),
            None => ToolOutput {
                content: format!("工具执行失败: 未知工具: {name}"),
                is_error: true,
            },
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
}
