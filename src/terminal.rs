use anyhow::{Context, Result};
use nu_ansi_term::{Color, Style};
use reedline::{
    ExternalPrinter, Highlighter, Prompt, PromptEditMode, PromptHistorySearch, Reedline, Signal,
    StyledText,
};
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

/// 统一输入编辑与后台输出；交互模式用加粗绿色显示用户问题。
pub struct Terminal {
    pub input: async_channel::UnboundedReceiver<InputEvent>,
    pub output: Output,
    busy: Option<Arc<AtomicBool>>,
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
            let busy = Arc::new(AtomicBool::new(true));
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
            let editor_busy = busy.clone();
            if let Err(error) = std::thread::Builder::new()
                .name("asteria-editor".into())
                .spawn(move || read_edited(printer, active, editor_busy, sender))
            {
                output.finish();
                return Err(error).context("无法启动终端编辑线程");
            }
            Ok(Self {
                input,
                output,
                busy: Some(busy),
            })
        } else {
            std::thread::Builder::new()
                .name("asteria-stdin".into())
                .spawn(move || read_plain(sender))
                .context("无法启动输入线程")?;
            Ok(Self {
                input,
                output: Output::default(),
                busy: None,
            })
        }
    }

    /// 输入编辑已结束后刷新最终输出。
    pub fn finish(&self) {
        self.output.finish();
    }

    /// 处理中改用省略号提示，避免流式输出打断正在编辑的输入行。
    pub fn set_busy(&self, busy: bool) {
        if let Some(flag) = &self.busy {
            flag.store(busy, Ordering::Release);
        }
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
    busy: Arc<AtomicBool>,
    sender: async_channel::UnboundedSender<InputEvent>,
) {
    let mut editor = Reedline::create()
        .with_ansi_colors(true)
        .with_highlighter(Box::new(UserInputHighlighter))
        .with_external_printer(printer);
    let final_event = loop {
        let prompt = UserPrompt::for_terminal(busy.clone());
        match editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    echo_submitted(&line, busy.load(Ordering::Acquire));
                    if !trimmed.starts_with('/') {
                        busy.store(true, Ordering::Release);
                    }
                }
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

const USER_PROMPT: &str = "User：";
const BUSY_PROMPT: &str = "… ";

fn user_question_style() -> Style {
    Style::new().bold().fg(Color::LightGreen)
}

/// 把正在输入的问题画成加粗亮绿色；斜杠命令保持默认颜色。
struct UserInputHighlighter;

impl Highlighter for UserInputHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
        let mut styled = StyledText::new();
        let style = if line.trim_start().starts_with('/') {
            Style::new()
        } else {
            user_question_style()
        };
        styled.push((style, line.to_string()));
        styled
    }
}

/// 空闲时单行 `User` 提示；处理中收成省略号。
struct UserPrompt {
    busy: Arc<AtomicBool>,
}

impl UserPrompt {
    fn for_terminal(busy: Arc<AtomicBool>) -> Self {
        Self { busy }
    }

    fn left_prompt(&self) -> String {
        if self.busy.load(Ordering::Acquire) {
            BUSY_PROMPT.into()
        } else {
            USER_PROMPT.into()
        }
    }
}

impl Prompt for UserPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        Cow::Owned(self.left_prompt())
    }

    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _: PromptEditMode) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Borrowed("  ")
    }

    /// 历史搜索提示，仅使用内存历史。
    fn render_prompt_history_search_indicator(&self, _: PromptHistorySearch) -> Cow<'_, str> {
        Cow::Borrowed("搜索: ")
    }

    fn get_prompt_color(&self) -> Color {
        Color::LightGreen
    }

    fn get_indicator_color(&self) -> Color {
        Color::LightGreen
    }
}

/// 问题回显为加粗绿色单行；斜杠命令仍用 `›`，避免批准过程再套一层标签。
fn echo_submitted(line: &str, busy: bool) {
    let (columns, rows) = terminal_size::terminal_size()
        .map(
            |(terminal_size::Width(columns), terminal_size::Height(rows))| {
                (usize::from(columns), usize::from(rows))
            },
        )
        .unwrap_or((80, 24));
    let (erase_rows, message) = submitted_echo(line, columns, busy);
    let erase_rows = erase_rows.min(rows.saturating_sub(1).max(1));

    let mut stdout = io::stdout().lock();
    for _ in 0..erase_rows {
        let _ = write!(stdout, "\x1b[1A\r\x1b[2K");
    }
    let _ = writeln!(stdout, "{message}");
    let _ = stdout.flush();
}

fn submitted_echo(line: &str, columns: usize, busy: bool) -> (usize, String) {
    let trimmed = line.trim();
    let prefix_width = if busy {
        unicode_width::UnicodeWidthStr::width(BUSY_PROMPT)
    } else {
        unicode_width::UnicodeWidthStr::width(USER_PROMPT)
    };
    let erase_rows = submitted_screen_rows(line, prefix_width, columns);
    if trimmed.starts_with('/') {
        (erase_rows, format!("› {trimmed}"))
    } else {
        (erase_rows, format!("{}\n", render_user_question(trimmed)))
    }
}

fn render_user_question(line: &str) -> String {
    user_question_style()
        .paint(format!("{USER_PROMPT}{}", sanitize_user_text(line)))
        .to_string()
}

fn sanitize_user_text(line: &str) -> String {
    line.chars()
        .filter(|character| *character == '\n' || !character.is_control())
        .map(|character| if character == '\t' { ' ' } else { character })
        .collect()
}

fn submitted_screen_rows(line: &str, prefix_width: usize, columns: usize) -> usize {
    use unicode_width::UnicodeWidthStr;

    let columns = columns.max(1);
    line.split('\n')
        .map(|part| {
            let width = prefix_width + UnicodeWidthStr::width(part);
            width.saturating_sub(1) / columns + 1
        })
        .sum::<usize>()
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submitted_row_count_handles_chinese_and_wrapping() {
        let prefix = unicode_width::UnicodeWidthStr::width(USER_PROMPT);
        assert_eq!(submitted_screen_rows("中文", prefix, 20), 1);
        assert_eq!(submitted_screen_rows(&"中".repeat(8), prefix, 20), 2);
        assert_eq!(submitted_screen_rows("first\nsecond", prefix, 20), 2);
    }

    #[test]
    fn live_prompt_is_a_single_user_line() {
        let prompt = UserPrompt {
            busy: Arc::new(AtomicBool::new(false)),
        };
        assert_eq!(prompt.left_prompt(), "User：");
        assert_eq!(prompt.get_prompt_color(), Color::LightGreen);
    }

    #[test]
    fn busy_prompt_collapses_to_a_single_line() {
        let prompt = UserPrompt {
            busy: Arc::new(AtomicBool::new(true)),
        };
        assert_eq!(prompt.left_prompt(), "… ");
    }

    #[test]
    fn user_question_is_bold_green_and_drops_control_sequences() {
        let painted = render_user_question("中文问题中文问题\x1b[31m");
        assert!(!painted.contains('╭'));
        assert!(!painted.contains('│'));
        assert_eq!(
            painted,
            user_question_style()
                .paint("User：中文问题中文问题[31m")
                .to_string()
        );
    }

    #[test]
    fn slash_commands_are_not_styled_as_user_questions() {
        let command = submitted_echo("/approve 1", 80, true).1;
        assert_eq!(command, "› /approve 1");
        assert!(!command.contains("User"));
        let question = submitted_echo("LT 反复重启", 80, false).1;
        assert!(question.contains("User"));
        assert!(question.contains("LT 反复重启"));
        assert!(!question.contains('╭'));
        assert!(question.contains("\x1b["));
        assert!(question.ends_with('\n'), "问题与后续输出之间应空一行");
    }

    #[test]
    fn highlighter_paints_questions_green_and_leaves_slash_commands_plain() {
        let question = UserInputHighlighter.highlight("LT 反复重启", 0);
        assert_eq!(question.buffer[0].0, user_question_style());
        let command = UserInputHighlighter.highlight("/approve 1", 0);
        assert_eq!(command.buffer[0].0, Style::new());
    }
}
