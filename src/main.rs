mod terminal;

use anyhow::{Context, Result, bail};
use asteria_agent::{
    agent::Asteria,
    agent_loop::{CancelToken, TurnState},
};
use std::collections::VecDeque;
use terminal::{InputEvent, Output, Terminal};

/// 启动命令行 Agent，由行编辑器负责输入与屏幕重绘。
#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let mut agent = Asteria::new()?;
    let mut terminal = Terminal::start()?;
    let retry_output = terminal.output.clone();
    agent.set_retry_output(move |message| retry_output.print(message));
    terminal.output.print(format!(
        "Asteria · {}  (/context 查看记忆，/usage 查看统计，/reset 清空记忆，/exit 退出；Ctrl+C 或 /cancel 取消当前轮)",
        agent.model()
    ));
    let mut queued: VecDeque<String> = VecDeque::new();
    loop {
        let input = if let Some(line) = queued.pop_front() {
            terminal
                .output
                .print(format!("[开始处理排队输入] {:?}", line.trim()));
            line
        } else {
            tokio::select! {
                event = terminal.input.recv() => match event {
                    Some(InputEvent::Line(line)) => line,
                    Some(InputEvent::Cancel) => {
                        terminal.output.print("当前没有运行中的 Turn，已清空输入行。");
                        continue;
                    }
                    Some(InputEvent::Error(error)) => { terminal.output.print(format!("输入错误: {error}")); break; }
                    None => break,
                },
                // Ctrl+C 按键由编辑器投递；也接收宿主直接发送的 SIGINT。
                signal = tokio::signal::ctrl_c() => {
                    signal.context("无法监听 Ctrl+C")?;
                    terminal.output.print("当前没有运行中的 Turn，输入 /exit 退出。");
                    continue;
                }
            }
        };
        match input.trim() {
            "/exit" => break,
            "/cancel" => terminal.output.print("当前没有运行中的 Turn。"),
            "/context" => print_context(&agent, &terminal.output),
            "/usage" => print_usage(&agent, &terminal.output),
            "/reset" => {
                agent.reset();
                terminal.output.print("Asteria: 记忆已清空。");
            }
            "" => {}
            text => {
                match ask_interruptible(&mut agent, text, &mut terminal, &mut queued).await {
                    Ok(answer) => terminal.output.print(format!("Asteria: {answer}")),
                    Err(_)
                        if agent
                            .last_turn()
                            .is_some_and(|turn| turn.state == TurnState::Cancelled) =>
                    {
                        terminal
                            .output
                            .print("Asteria: 当前 Turn 已取消，可继续提问。");
                    }
                    Err(error) => terminal.output.print(format!("错误: {error:#}")),
                }
                print_usage(&agent, &terminal.output);
            }
        }
    }
    terminal.finish();
    Ok(())
}

/// 同时等待回答、输入与取消；普通输入排队，取消后等待回滚完成。
async fn ask_interruptible(
    agent: &mut Asteria,
    input: &str,
    terminal: &mut Terminal,
    queued: &mut VecDeque<String>,
) -> Result<String> {
    terminal
        .output
        .print("[处理中] 可继续输入并回车排队，或输入 /cancel 取消当前轮。");
    let cancel = CancelToken::new();
    let request = agent.ask_with_cancel(input, &cancel);
    tokio::pin!(request);
    let mut input_open = true;
    loop {
        tokio::select! {
            biased;
            signal = tokio::signal::ctrl_c() => {
                terminal.output.print("[Cancel] 收到 Ctrl+C，正在取消当前 Turn。");
                cancel.cancel();
                let result = request.await;
                signal.context("无法监听 Ctrl+C")?;
                return result;
            }
            result = &mut request => return result,
            event = terminal.input.recv(), if input_open => {
                match event {
                    Some(InputEvent::Cancel) => {
                        terminal.output.print("[Cancel] 收到 Ctrl+C，正在取消当前 Turn。");
                        cancel.cancel();
                        return request.await;
                    }
                    Some(InputEvent::Line(line)) if line.trim() == "/cancel" => {
                        terminal.output.print("[Cancel] 收到 /cancel，正在取消当前 Turn。");
                        cancel.cancel();
                        return request.await;
                    }
                    Some(InputEvent::Line(line)) if line.trim().is_empty() => {},
                    Some(InputEvent::Line(line)) => {
                        queued.push_back(line);
                        terminal.output.print(format!("[已排队] 当前有 {} 条待处理输入，将按顺序执行。", queued.len()));
                    },
                    Some(InputEvent::Error(error)) => {
                        cancel.cancel();
                        let _ = request.await;
                        bail!("输入错误: {error}");
                    }
                    // EOF 不取消已提交的问题，仍等待答案并处理已排队的消息。
                    None => input_open = false,
                }
            }
        }
    }
}

/// 显示原始记忆；Debug 转义控制字符，整块输出后由编辑器恢复输入行。
fn print_context(agent: &Asteria, output: &Output) {
    let context = agent.context();
    let mut lines = vec![
        format!(
            "[ContextMemory] messages={}（不含 System；原始历史，不是本次模型请求）",
            context.messages().len()
        ),
        format!("System: {:?}", context.system_prompt()),
    ];
    for (index, message) in context.messages().iter().enumerate() {
        lines.push(format!("{}: {:?}", index + 1, message));
    }
    output.print(lines.join("\n"));
}

/// 成功或失败都显示报告；合并输出，避免逐行打印打断正在编辑的文字。
fn print_usage(agent: &Asteria, output: &Output) {
    let mut lines = Vec::new();
    if let Some(turn) = agent.last_turn() {
        lines.push(format!(
            "[Turn {}] state={:?} steps={} retries={} context_messages={}",
            turn.id,
            turn.state,
            turn.steps,
            turn.retries,
            agent.context().messages().len()
        ));
        lines.push(format!(
            "[Turn Token Usage] input={} output={} total={}",
            turn.usage.prompt_tokens, turn.usage.completion_tokens, turn.usage.total_tokens
        ));
    } else {
        lines.push("尚未执行 Turn。".into());
    }
    let usage = agent.session_usage();
    lines.push(format!(
        "[Session Token Usage] input={} output={} total={}",
        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
    ));
    lines.push("用量仅累计已收到的 usage；未返回的用量未知。/reset 不清空累计。".into());
    output.print(lines.join("\n"));
}
