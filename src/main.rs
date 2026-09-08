use anyhow::{Context, Result, bail};
use chrono::Local;
use reqwest::blocking::Client;
use serde_json::{Value, json};
use std::{
    env,
    io::{self, Write},
};

const API_URL: &str = "https://api.deepseek.com/chat/completions";
const SYSTEM: &str = "你是 Asteria，一个可靠、简洁的中文 AI 助手。需要精确计算或当前时间时调用工具，不要猜测工具结果。";

fn tools() -> Value {
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

fn run_tool(name: &str, raw_args: &str) -> String {
    let args: Value = match serde_json::from_str(raw_args) {
        Ok(value) => value,
        Err(error) => return format!("参数解析失败: {error}"),
    };
    match name {
        "calculate" => args["expression"]
            .as_str()
            .ok_or_else(|| "缺少 expression 参数".to_string())
            .and_then(|expr| {
                if expr.len() > 200 {
                    return Err("表达式过长".into());
                }
                meval::eval_str(expr)
                    .map(|n| n.to_string())
                    .map_err(|e| e.to_string())
            })
            .unwrap_or_else(|error| format!("计算失败: {error}")),
        "current_time" => Local::now().to_rfc3339(),
        _ => format!("未知工具: {name}"),
    }
}

struct Asteria {
    client: Client,
    api_key: String,
    model: String,
    messages: Vec<Value>,
}

impl Asteria {
    fn new() -> Result<Self> {
        let api_key =
            env::var("DEEPSEEK_API_KEY").context("请先在 .env 中设置 DEEPSEEK_API_KEY")?;
        Ok(Self {
            client: Client::new(),
            api_key,
            model: env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".into()),
            messages: vec![json!({"role":"system","content":SYSTEM})],
        })
    }

    fn reset(&mut self) {
        self.messages.truncate(1);
    }

    fn ask(&mut self, input: &str) -> Result<String> {
        self.messages.push(json!({"role":"user","content":input}));
        for _ in 0..8 {
            let response: Value = self
                .client
                .post(API_URL)
                .bearer_auth(&self.api_key)
                .json(&json!({
                    "model": self.model, "messages": self.messages,
                    "tools": tools(), "tool_choice": "auto",
                    "thinking": {"type":"enabled"}
                }))
                .send()
                .context("无法连接 DeepSeek API")?
                .error_for_status()
                .context("DeepSeek API 返回错误")?
                .json()
                .context("无法解析 DeepSeek 响应")?;

            let message = response["choices"][0]["message"].clone();
            if message.is_null() {
                bail!("DeepSeek 响应中没有 message");
            }
            self.messages.push(message.clone());
            let Some(calls) = message["tool_calls"]
                .as_array()
                .filter(|calls| !calls.is_empty())
            else {
                return Ok(message["content"].as_str().unwrap_or("").to_string());
            };
            for call in calls {
                let id = call["id"].as_str().unwrap_or("");
                let name = call["function"]["name"].as_str().unwrap_or("");
                let args = call["function"]["arguments"].as_str().unwrap_or("{}");
                let output = run_tool(name, args);
                self.messages.push(json!({
                    "role":"tool", "tool_call_id":id, "content":output
                }));
            }
        }
        bail!("工具调用轮次过多，已停止")
    }
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let mut agent = Asteria::new()?;
    println!("Asteria · {}  (/reset 清空记忆，/exit 退出)", agent.model);
    loop {
        print!("\n你: ");
        io::stdout().flush()?;
        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }
        match input.trim() {
            "/exit" => break,
            "/reset" => {
                agent.reset();
                println!("Asteria: 记忆已清空。");
            }
            "" => {}
            text => match agent.ask(text) {
                Ok(answer) => println!("Asteria: {answer}"),
                Err(error) => eprintln!("错误: {error:#}"),
            },
        }
    }
    Ok(())
}
