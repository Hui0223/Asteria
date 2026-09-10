use anyhow::Result;
use asteria_agent::agent::Asteria;
use std::io::{self, Write};

/// 启动命令行 Agent，并持续读取用户输入直到退出。
fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let mut agent = Asteria::new()?;
    println!(
        "Asteria · {}  (/context 查看记忆，/usage 查看统计，/reset 清空记忆，/exit 退出)",
        agent.model()
    );
    loop {
        print!("\n你: ");
        io::stdout().flush()?;
        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }
        match input.trim() {
            "/exit" => break,
            "/context" => print_context(&agent),
            "/usage" => print_usage(&agent),
            "/reset" => {
                agent.reset();
                println!("Asteria: 记忆已清空。");
            }
            "" => {}
            text => {
                match agent.ask(text) {
                    Ok(answer) => println!("Asteria: {answer}"),
                    Err(error) => println!("错误: {error:#}"),
                }
                print_usage(&agent);
            }
        }
    }
    Ok(())
}

/// 显示原始内存消息；Debug 转义控制字符，明确区分它与预算后的请求视图。
fn print_context(agent: &Asteria) {
    let context = agent.context();
    println!(
        "[ContextMemory] messages={}（不含 System；原始历史，不是本次模型请求）",
        context.messages().len()
    );
    println!("System: {:?}", context.system_prompt());
    for (index, message) in context.messages().iter().enumerate() {
        println!("{}: {:?}", index + 1, message);
    }
}

/// 成功或失败都显示报告；用量只代表已收到的服务端 usage。
fn print_usage(agent: &Asteria) {
    if let Some(turn) = agent.last_turn() {
        println!(
            "[Turn {}] state={:?} steps={} retries={} context_messages={}",
            turn.id,
            turn.state,
            turn.steps,
            turn.retries,
            agent.context().messages().len()
        );
        println!(
            "[Turn Token Usage] input={} output={} total={}",
            turn.usage.prompt_tokens, turn.usage.completion_tokens, turn.usage.total_tokens
        );
    } else {
        println!("尚未执行 Turn。");
    }
    let usage = agent.session_usage();
    println!(
        "[Session Token Usage] input={} output={} total={}",
        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
    );
    println!("用量仅累计已收到的 usage；未返回的用量未知。/reset 不清空累计。");
}
