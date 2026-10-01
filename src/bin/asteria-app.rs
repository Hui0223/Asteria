use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::{
    env, io,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    thread,
};
use tao::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder},
    window::WindowBuilder,
};
use wry::WebViewBuilder;

enum AppEvent {
    Line(String),
    Exited(Option<i32>),
}

fn main() -> Result<()> {
    dotenvy::from_path(workspace_dir().join(".env")).ok();
    dotenvy::dotenv().ok();

    let workspace = workspace_dir();
    let agent = find_agent_binary()?;
    let mut child = Command::new(&agent)
        .arg("--rpc")
        .current_dir(&workspace)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("无法启动 {}", agent.display()))?;
    let stdin = Arc::new(Mutex::new(
        child.stdin.take().context("asteria-agent 未提供 stdin")?,
    ));
    let stdout = child.stdout.take().context("asteria-agent 未提供 stdout")?;
    let stderr = child.stderr.take().context("asteria-agent 未提供 stderr")?;
    let child = Arc::new(Mutex::new(child));

    let event_loop = EventLoopBuilder::<AppEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();
    thread::spawn({
        let proxy = proxy.clone();
        move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if proxy.send_event(AppEvent::Line(line)).is_err() {
                    break;
                }
            }
        }
    });
    thread::spawn({
        let proxy = proxy.clone();
        move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if line.trim().is_empty() {
                    continue;
                }
                let payload = json!({
                    "event": "log",
                    "text": line,
                })
                .to_string();
                if proxy.send_event(AppEvent::Line(payload)).is_err() {
                    break;
                }
            }
        }
    });
    thread::spawn({
        let proxy = proxy.clone();
        let child = child.clone();
        move || {
            let status = child.lock().ok().and_then(|mut child| child.wait().ok());
            let _ = proxy.send_event(AppEvent::Exited(status.and_then(|status| status.code())));
        }
    });

    let window = WindowBuilder::new()
        .with_title("Asteria")
        .with_inner_size(LogicalSize::new(1100.0, 720.0))
        .build(&event_loop)
        .context("无法创建窗口")?;
    let ipc_stdin = stdin.clone();
    let webview = WebViewBuilder::new()
        .with_html(page_html())
        .with_ipc_handler(move |request| {
            if let Err(error) = send_command(&ipc_stdin, request.body()) {
                eprintln!("发送 RPC 命令失败: {error:#}");
            }
        })
        .build(&window)
        .context("无法创建 WebView")?;

    event_loop.run(move |event, _target, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::UserEvent(AppEvent::Line(line)) => {
                if line.contains("\"event\":\"exit\"") {
                    if let Ok(mut child) = child.lock() {
                        let _ = child.kill();
                    }
                    *control_flow = ControlFlow::Exit;
                }
                if let Some(script) = push_script(&line) {
                    let _ = webview.evaluate_script(&script);
                }
            }
            Event::UserEvent(AppEvent::Exited(code)) => {
                let payload = json!({
                    "type": "offline",
                    "text": format!("Asteria 进程已退出（{}）", code.map_or("null".into(), |code| code.to_string())),
                });
                let _ = webview.evaluate_script(&format!("window.__asteriaPush({payload})"));
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                if let Ok(mut child) = child.lock() {
                    let _ = child.kill();
                }
                *control_flow = ControlFlow::Exit;
            }
            _ => {}
        }
    });
}

fn send_command(stdin: &Arc<Mutex<impl Write>>, body: &str) -> Result<()> {
    let message: Value = serde_json::from_str(body).context("窗口消息必须是 JSON")?;
    let Some(command) = to_rpc(&message) else {
        return Ok(());
    };
    let mut stdin = stdin
        .lock()
        .map_err(|_| io::Error::other("stdin 锁已损坏"))?;
    writeln!(stdin, "{command}")?;
    stdin.flush()?;
    Ok(())
}

fn to_rpc(message: &Value) -> Option<Value> {
    Some(match message.get("type")?.as_str()? {
        "ask" => json!({"method":"ask","params":{"text": message.get("text")?}}),
        "approve" => json!({"method":"approve","params":{"approval_id": message.get("id")?}}),
        "deny" => json!({"method":"deny","params":{"approval_id": message.get("id")?}}),
        "cancel" => json!({"method":"cancel"}),
        "new_session" => json!({"method":"new_session"}),
        "switch_session" => json!({"method":"switch_session","params":{"id": message.get("id")?}}),
        _ => return None,
    })
}

fn push_script(line: &str) -> Option<String> {
    let event: Value = serde_json::from_str(line).ok()?;
    let mapped = map_event(event)?;
    Some(format!("window.__asteriaPush({mapped})"))
}

fn map_event(event: Value) -> Option<Value> {
    Some(match event.get("event")?.as_str()? {
        "ready" => json!({
            "type": "ready",
            "model": event.get("model")?,
            "rag": event.get("rag")?,
            "mcp": event.get("mcp")?,
            "currentSession": event.get("current_session")?,
            "sessions": event.get("sessions")?,
        }),
        "user" => json!({"type":"user","text": event.get("text")?}),
        "command" => json!({"type":"command","text": event.get("text")?}),
        "command_result" => json!({"type":"commandResult","text": event.get("text")?}),
        "exit" => json!({"type":"exit"}),
        "status" => json!({"type":"status","text": event.get("text")?}),
        "assistant_begin" => json!({"type":"assistantBegin"}),
        "assistant_delta" => json!({"type":"assistantDelta","text": event.get("text")?}),
        "tool_started" => json!({"type":"toolStarted","name": event.get("name")?}),
        "tool_result" => json!({
            "type": "toolResult",
            "name": event.get("name")?,
            "ok": event.get("ok")?,
            "ms": event.get("ms")?,
        }),
        "approval" => json!({
            "type": "approval",
            "id": event.get("id")?,
            "tool": event.get("tool")?,
            "arguments": event.get("arguments")?,
        }),
        "turn_completed" => json!({
            "type": "turnCompleted",
            "steps": event.get("steps")?,
            "tokens": event.get("tokens")?,
        }),
        "turn_cancelled" => json!({"type":"turnCancelled"}),
        "session_cleared" => json!({"type":"sessionCleared"}),
        "session_changed" => json!({
            "type": "sessionChanged",
            "currentId": event.get("current_id")?,
            "title": event.get("title")?,
            "messageCount": event.get("message_count")?,
            "resetView": event.get("reset_view")?,
            "sessions": event.get("sessions")?,
        }),
        "error" => json!({"type":"error","message": event.get("message")?}),
        "log" => json!({"type":"log","text": event.get("text")?}),
        _ => return None,
    })
}

fn page_html() -> String {
    include_str!("../../app/web/index.html")
        .replace("/*APP_CSS*/", include_str!("../../app/web/app.css"))
        .replace("/*APP_JS*/", include_str!("../../app/web/app.js"))
}

fn workspace_dir() -> PathBuf {
    let mut dir = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if let Ok(canonical) = dir.canonicalize() {
        dir = canonical;
    }
    let mut cursor = dir.clone();
    loop {
        if cursor.join("Cargo.toml").is_file() {
            return cursor;
        }
        if !cursor.pop() {
            break;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn find_agent_binary() -> Result<PathBuf> {
    if let Some(path) = env::var_os("ASTERIA_BIN") {
        return Ok(PathBuf::from(path));
    }
    let mut candidates = Vec::new();
    if let Ok(exe) = env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("asteria-agent"));
    }
    if let Some(target) = env::var_os("CARGO_TARGET_DIR") {
        let target = PathBuf::from(target);
        candidates.push(target.join("debug/asteria-agent"));
        candidates.push(target.join("release/asteria-agent"));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(root.join("target/debug/asteria-agent"));
    candidates.push(root.join("target/release/asteria-agent"));
    if let Some(path) = candidates.into_iter().find(|path| path.is_file()) {
        return Ok(path);
    }
    bail!(
        "找不到 asteria-agent。请先在仓库根目录运行 cargo build --bin asteria-agent，或设置 ASTERIA_BIN。"
    )
}
