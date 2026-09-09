use anyhow::Result;
use asteria_agent::agent::Asteria;
use std::io::{self, Write};

/// 启动命令行 Agent，并持续读取用户输入直到退出。
fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let mut agent = Asteria::new()?;
    println!("Asteria · {}  (/reset 清空记忆，/exit 退出)", agent.model());
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
                Ok(answer) => {
                    println!("Asteria: {answer}");
                    if let Some(usage) = agent.last_usage() {
                        println!(
                            "[Turn Token Usage] input={} output={} total={}",
                            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                        );
                        let session = agent.session_usage();
                        println!(
                            "[Session Token Usage] input={} output={} total={}",
                            session.prompt_tokens, session.completion_tokens, session.total_tokens
                        );
                    }
                }
                Err(error) => eprintln!("错误: {error:#}"),
            },
        }
    }
    Ok(())
}
