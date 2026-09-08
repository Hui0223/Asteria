use chrono::Local;
use serde_json::{Value, json};

pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

pub fn schema() -> Value {
    json!([
        {"type":"function","function":{
            "name":"calculate","description":"计算一个数学表达式",
            "parameters":{"type":"object","properties":{
                "expression":{"type":"string","description":"例如 (12+3)*4"}
            },"required":["expression"]}
        }},
        {"type":"function","function":{
            "name":"current_time","description":"获取运行机器的当前本地时间",
            "parameters":{"type":"object","properties":{}}
        }}
    ])
}

pub fn execute(name: &str, raw_args: &str) -> ToolOutput {
    let result = (|| -> Result<String, String> {
        let args: Value = serde_json::from_str(raw_args).map_err(|error| error.to_string())?;
        match name {
            "calculate" => {
                let expression = args["expression"].as_str().ok_or("缺少 expression 参数")?;
                if expression.len() > 200 {
                    return Err("表达式过长".into());
                }
                meval::eval_str(expression)
                    .map(|number| number.to_string())
                    .map_err(|error| error.to_string())
            }
            "current_time" => Ok(Local::now().to_rfc3339()),
            _ => Err(format!("未知工具: {name}")),
        }
    })();
    match result {
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
