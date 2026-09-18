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

/// 统一输入编辑与后台输出；交互模式把用户问题画在 User 消息框里。
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
            OutputEvent::Text(text) => {
                // Reedline 只擦输入行；多行 User 框的顶边和标签要先清掉，避免残影。
                let mut text = format!(
                    "{}{text}",
                    "\x1b[1A\r\x1b[2K".repeat(LIVE_PROMPT_DECORATION_ROWS)
                );
                loop {
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
                }
            }
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
        let prompt = UserPrompt::for_terminal();
        match editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => {
                replace_submitted_prompt(&line);
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

/// 顶边和 `User` 标签各占一行，输入落在第三行。
const LIVE_PROMPT_DECORATION_ROWS: usize = 2;
const USER_LABEL: &str = "User";

/// 编辑态即显示完整 User 消息框；提交后再补底边。
struct UserPrompt {
    box_width: usize,
}

impl UserPrompt {
    fn for_terminal() -> Self {
        Self {
            box_width: prompt_box_width(terminal_columns()),
        }
    }

    fn left_prompt(&self) -> String {
        format!(
            "{}\n{}\n│ ",
            box_top(self.box_width),
            box_row(USER_LABEL, self.box_width)
        )
    }
}

impl Prompt for UserPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        Cow::Owned(self.left_prompt())
    }

    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("│ ")
    }

    fn render_prompt_indicator(&self, _: PromptEditMode) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Borrowed("│ ")
    }

    /// 历史搜索提示，仅使用内存历史。
    fn render_prompt_history_search_indicator(&self, _: PromptHistorySearch) -> Cow<'_, str> {
        Cow::Borrowed("│ 搜索: ")
    }

    fn right_prompt_on_last_line(&self) -> bool {
        true
    }
}

/// 清除 Reedline 刚提交的普通提示行，并在同一位置绘制完整问题框。
fn replace_submitted_prompt(line: &str) {
    let (columns, rows) = terminal_size::terminal_size()
        .map(
            |(terminal_size::Width(columns), terminal_size::Height(rows))| {
                (usize::from(columns), usize::from(rows))
            },
        )
        .unwrap_or((80, 24));
    let submitted_rows = submitted_screen_rows(line, columns).min(rows.saturating_sub(1).max(1));
    let message = render_user_box(line, columns);

    let mut stdout = io::stdout().lock();
    for _ in 0..submitted_rows {
        let _ = write!(stdout, "\x1b[1A\r\x1b[2K");
    }
    let _ = writeln!(stdout, "{message}");
    let _ = stdout.flush();
}

fn render_user_box(line: &str, columns: usize) -> String {
    let box_width = prompt_box_width(columns);
    let inner_width = box_inner_width(box_width);
    let mut message = format!(
        "{}\n{}\n",
        box_top(box_width),
        box_row(USER_LABEL, box_width)
    );
    for content in wrap_for_box(line, inner_width) {
        message.push_str(&box_row(&content, box_width));
        message.push('\n');
    }
    message.push_str(&box_bottom(box_width));
    message
}

fn terminal_columns() -> usize {
    terminal_size::terminal_size()
        .map(|(terminal_size::Width(columns), _)| usize::from(columns))
        .unwrap_or(80)
}

fn prompt_box_width(columns: usize) -> usize {
    columns.saturating_sub(1).max(10)
}

fn box_inner_width(box_width: usize) -> usize {
    box_width.saturating_sub(4).max(1)
}

fn box_top(box_width: usize) -> String {
    format!("╭{}╮", "─".repeat(box_width.saturating_sub(2)))
}

fn box_bottom(box_width: usize) -> String {
    format!("╰{}╯", "─".repeat(box_width.saturating_sub(2)))
}

fn box_row(content: &str, box_width: usize) -> String {
    use unicode_width::UnicodeWidthStr;

    let inner_width = box_inner_width(box_width);
    let padding = inner_width.saturating_sub(UnicodeWidthStr::width(content));
    format!("│ {content}{} │", " ".repeat(padding))
}

fn wrap_for_box(line: &str, width: usize) -> Vec<String> {
    use unicode_width::UnicodeWidthChar;

    let mut lines = vec![String::new()];
    let mut current_width = 0usize;
    for character in line.chars() {
        if character == '\n' {
            lines.push(String::new());
            current_width = 0;
            continue;
        }
        let replacement = if character == '\t' { ' ' } else { character };
        if replacement.is_control() {
            continue;
        }
        let character_width = UnicodeWidthChar::width(replacement).unwrap_or(0);
        if current_width + character_width > width && !lines.last().unwrap().is_empty() {
            lines.push(String::new());
            current_width = 0;
        }
        lines.last_mut().unwrap().push(replacement);
        current_width += character_width;
    }
    lines
}

fn submitted_screen_rows(line: &str, columns: usize) -> usize {
    use unicode_width::UnicodeWidthStr;

    let columns = columns.max(1);
    LIVE_PROMPT_DECORATION_ROWS
        + line
            .split('\n')
            .map(|part| {
                let width = 2 + UnicodeWidthStr::width(part);
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
        assert_eq!(submitted_screen_rows("中文", 20), 3);
        assert_eq!(submitted_screen_rows(&"中".repeat(8), 20), 3);
        assert_eq!(submitted_screen_rows("first\nsecond", 20), 4);
    }

    #[test]
    fn live_prompt_puts_user_inside_the_box() {
        use unicode_width::UnicodeWidthStr;

        let prompt = UserPrompt { box_width: 16 };
        let left = prompt.left_prompt();
        let lines: Vec<_> = left.lines().collect();
        assert_eq!(lines[0], "╭──────────────╮");
        assert_eq!(lines[1], "│ User         │");
        assert_eq!(lines[2], "│ ");
        assert!(lines.iter().all(|line| UnicodeWidthStr::width(*line) <= 16));
    }

    #[test]
    fn user_box_wraps_chinese_and_drops_control_sequences() {
        use unicode_width::UnicodeWidthStr;

        let box_text = render_user_box("中文问题中文问题\x1b[31m", 24);
        assert!(!box_text.contains('\x1b'));
        assert!(
            box_text
                .lines()
                .all(|line| UnicodeWidthStr::width(line) <= 23)
        );
        let lines: Vec<_> = box_text.lines().collect();
        assert!(lines[0].starts_with('╭'));
        assert!(lines[1].starts_with("│ User"));
        assert!(lines[1].ends_with('│'));
        assert!(lines[2].contains("中文问题"));
        assert!(box_text.ends_with('╯'));
    }
}
