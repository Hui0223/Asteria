use anyhow::{Context, Result};
use reedline::{ExternalPrinter, Prompt, PromptEditMode, PromptHistorySearch, Reedline, Signal};
use std::{
    borrow::Cow,
    io::{self, IsTerminal, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};
use tokio::sync::mpsc as async_channel;

/// 编辑器投递完整输入或取消操作；半成品输入不会进入 Agent。
pub enum InputEvent {
    Line(String),
    Cancel,
    Error(String),
}

enum OutputEvent {
    Text(String),
    Finish(mpsc::Sender<()>),
}

/// 所有回答、诊断和重试信息共享的输出入口。
#[derive(Clone, Default)]
pub struct Output {
    sender: Option<mpsc::Sender<OutputEvent>>,
}

impl Output {
    /// 交互模式通过编辑器打印并恢复输入行；管道模式只输出普通文本。
    pub fn print(&self, message: impl AsRef<str>) {
        if let Some(sender) = &self.sender {
            let _ = sender.send(OutputEvent::Text(message.as_ref().into()));
        } else {
            write_plain(message.as_ref());
        }
    }

    /// 等待最终输出刷新完毕，避免 /exit 时遗漏尚未绘制的回答。
    fn finish(&self) {
        if let Some(sender) = &self.sender {
            let (sent, received) = mpsc::channel();
            if sender.send(OutputEvent::Finish(sent)).is_ok() {
                let _ = received.recv();
            }
        }
    }
}

/// 统一输入编辑与后台输出，只有编辑器绘制“你:”提示符。
pub struct Terminal {
    pub input: async_channel::UnboundedReceiver<InputEvent>,
    pub output: Output,
}

impl Terminal {
    /// 真终端使用 Reedline；重定向或 TERM=dumb 使用纯文本输入输出。
    pub fn start() -> Result<Self> {
        let (sender, input) = async_channel::unbounded_channel();
        let interactive = io::stdin().is_terminal()
            && io::stdout().is_terminal()
            && std::env::var("TERM").is_ok_and(|term| term != "dumb");
        if interactive {
            let printer = ExternalPrinter::<String>::new(64);
            let active = Arc::new(AtomicBool::new(true));
            let (out_sender, out_receiver) = mpsc::channel();
            let output = Output {
                sender: Some(out_sender),
            };
            let output_active = active.clone();
            let output_printer = printer.clone();
            std::thread::Builder::new()
                .name("asteria-output".into())
                .spawn(move || print_messages(out_receiver, output_printer, output_active))
                .context("无法启动终端输出线程")?;
            if let Err(error) = std::thread::Builder::new()
                .name("asteria-editor".into())
                .spawn(move || read_edited(printer, active, sender))
            {
                output.finish();
                return Err(error).context("无法启动终端编辑线程");
            }
            Ok(Self { input, output })
        } else {
            std::thread::Builder::new()
                .name("asteria-stdin".into())
                .spawn(move || read_plain(sender))
                .context("无法启动输入线程")?;
            Ok(Self {
                input,
                output: Output::default(),
            })
        }
    }

    /// 输入编辑已结束后刷新最终输出。
    pub fn finish(&self) {
        self.output.finish();
    }
}

/// 后台输出串行送入编辑器；有限等待避免编辑器退出时阻塞在已满队列上。
fn print_messages(
    receiver: mpsc::Receiver<OutputEvent>,
    printer: ExternalPrinter<String>,
    active: Arc<AtomicBool>,
) {
    while let Ok(event) = receiver.recv() {
        match event {
            OutputEvent::Text(mut text) => loop {
                if !active.load(Ordering::Acquire) {
                    while let Some(queued) = printer.get_line() {
                        write_plain(&queued);
                    }
                    write_plain(&text);
                    break;
                }
                match printer
                    .sender()
                    .send_timeout(text, Duration::from_millis(20))
                {
                    Ok(()) => break,
                    Err(error) => text = error.into_inner(),
                }
            },
            OutputEvent::Finish(done) => {
                while let Some(queued) = printer.get_line() {
                    write_plain(&queued);
                }
                let _ = done.send(());
                return;
            }
        }
    }
}

/// 行编辑线程拥有 raw mode，退出前先恢复终端，再关闭输入通道。
fn read_edited(
    printer: ExternalPrinter<String>,
    active: Arc<AtomicBool>,
    sender: async_channel::UnboundedSender<InputEvent>,
) {
    let mut editor = Reedline::create()
        .with_ansi_colors(false)
        .with_external_printer(printer);
    let final_event = loop {
        match editor.read_line(&UserPrompt) {
            Ok(Signal::Success(line)) => {
                if line.trim() == "/exit" {
                    break Some(InputEvent::Line(line));
                }
                if sender.send(InputEvent::Line(line)).is_err() {
                    break None;
                }
            }
            Ok(Signal::CtrlC) => {
                if sender.send(InputEvent::Cancel).is_err() {
                    break None;
                }
            }
            Ok(Signal::CtrlD) => break None,
            Ok(_) => continue,
            Err(error) => break Some(InputEvent::Error(error.to_string())),
        }
    };
    drop(editor);
    active.store(false, Ordering::Release);
    if let Some(event) = final_event {
        let _ = sender.send(event);
    }
}

/// 纯文本读取兼容 here-doc 和脚本，不输出 ANSI 编辑指令。
fn read_plain(sender: async_channel::UnboundedSender<InputEvent>) {
    loop {
        let mut line = String::new();
        match io::stdin().read_line(&mut line) {
            Ok(0) => return,
            Ok(_) => {
                let exit = line.trim() == "/exit";
                if sender.send(InputEvent::Line(line)).is_err() || exit {
                    return;
                }
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => {
                let _ = sender.send(InputEvent::Error(error.to_string()));
                return;
            }
        }
    }
}

/// 终端不处于编辑状态时打印完整一行。
fn write_plain(text: &str) {
    let _ = writeln!(io::stdout().lock(), "{text}");
}

/// 使用简单中文提示符，避免编辑状态、时间或路径重复占据输入行。
struct UserPrompt;
impl Prompt for UserPrompt {
    /// 左提示由编辑器计算中文显示宽度。
    fn render_prompt_left(&self) -> Cow<'_, str> {
        Cow::Borrowed("你")
    }
    /// 不展示右侧提示。
    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }
    /// 所有编辑模式使用相同的提示符。
    fn render_prompt_indicator(&self, _: PromptEditMode) -> Cow<'_, str> {
        Cow::Borrowed(": ")
    }
    /// 多行粘贴使用续行提示。
    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Borrowed("… ")
    }
    /// 历史搜索提示，仅使用内存历史。
    fn render_prompt_history_search_indicator(&self, _: PromptHistorySearch) -> Cow<'_, str> {
        Cow::Borrowed("搜索: ")
    }
}
