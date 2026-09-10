use anyhow::{Context, Result};
use asteria_agent::{
    agent::Asteria,
    agent_loop::{CancelToken, TurnState},
};
use std::collections::VecDeque;
use std::io::{self, Write};
use tokio::sync::mpsc;

/// 启动命令行 Agent，并持续读取用户输入直到退出。
#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let mut agent = Asteria::new()?;
    println!(
        "Asteria · {}  (/context 查看记忆，/usage 查看统计，/reset 清空记忆，/exit 退出；运行中 Ctrl+C 或 /cancel 回车取消当前轮)",
        agent.model()
    );
    let mut input_lines = read_input()?;
    let mut queued = VecDeque::new();
    loop {
        print!("\n你: ");
        io::stdout().flush()?;
        let input = if let Some(line) = queued.pop_front() {
            line
        } else {
            tokio::select! {
                line = input_lines.recv() => match line {
                    Some(line) => line?,
                    None => break,
                },
                signal = tokio::signal::ctrl_c() => {
                    signal.context("无法监听 Ctrl+C")?;
                    println!("\n当前没有运行中的 Turn，输入 /exit 退出。");
                    continue;
                }
            }
        };
        match input.trim() {
            "/exit" => break,
            "/cancel" => println!("当前没有运行中的 Turn。"),
            "/context" => print_context(&agent),
            "/usage" => print_usage(&agent),
            "/reset" => {
                agent.reset();
                println!("Asteria: 记忆已清空。");
            }
            "" => {}
            text => {
                match ask_interruptible(&mut agent, text, &mut input_lines, &mut queued).await {
                    Ok(answer) => println!("Asteria: {answer}"),
                    Err(_)
                        if agent
                            .last_turn()
                            .is_some_and(|turn| turn.state == TurnState::Cancelled) =>
                    {
                        println!("Asteria: 当前 Turn 已取消，可继续提问。");
                    }
                    Err(error) => println!("错误: {error:#}"),
                }
                print_usage(&agent);
            }
        }
    }
    Ok(())
}

/// 用独立输入线程读取终端，通过有界通道交给异步主循环。
/// 不使用 Tokio 的阻塞 stdin 任务，避免 /exit 时运行时等待 stdin 关闭。
fn read_input() -> Result<mpsc::Receiver<io::Result<String>>> {
    let (sender, receiver) = mpsc::channel(16);
    std::thread::Builder::new()
        .name("asteria-stdin".into())
        .spawn(move || {
            loop {
                let mut line = String::new();
                match io::stdin().read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if sender.blocking_send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => {
                        let _ = sender.blocking_send(Err(error));
                        break;
                    }
                }
            }
        })
        .context("无法创建终端输入线程")?;
    Ok(receiver)
}

/// 同时等待回答和 Ctrl+C；只取消请求信号，继续等待 Turn 完成状态更新与回滚。
async fn ask_interruptible(
    agent: &mut Asteria,
    input: &str,
    input_lines: &mut mpsc::Receiver<io::Result<String>>,
    queued: &mut VecDeque<String>,
) -> Result<String> {
    let cancel = CancelToken::new();
    let request = agent.ask_with_cancel(input, &cancel);
    tokio::pin!(request);
    let mut input_open = true;
    loop {
        tokio::select! {
            biased;
            signal = tokio::signal::ctrl_c() => {
                println!("\n[Cancel] 收到 Ctrl+C，正在取消当前 Turn。");
                cancel.cancel();
                let result = request.await;
                signal.context("无法监听 Ctrl+C")?;
                return result;
            }
            result = &mut request => return result,
            line = input_lines.recv(), if input_open => {
                match line {
                    Some(Ok(line)) if line.trim() == "/cancel" => {
                        println!("[Cancel] 收到 /cancel，正在取消当前 Turn。");
                        cancel.cancel();
                        return request.await;
                    }
                    // 正常输入排队到下一轮，保持管道批量输入的顺序。
                    Some(Ok(line)) => queued.push_back(line),
                    Some(Err(error)) => {
                        cancel.cancel();
                        let _ = request.await;
                        return Err(error.into());
                    }
                    // 输入 EOF 不是取消：等待当前回答后处理已排队的行。
                    None => input_open = false,
                }
            }
        }
    }
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
