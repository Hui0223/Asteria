const vscode = require("vscode");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const readline = require("readline");

class AsteriaViewProvider {
  constructor(context) {
    this.context = context;
    this.view = undefined;
    this.child = undefined;
  }

  resolveWebviewView(view) {
    this.view = view;
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, "media")],
    };
    view.webview.html = this.html(view.webview);
    view.webview.onDidReceiveMessage((message) => this.onMessage(message));
    view.onDidDispose(() => this.stop());
    this.start();
  }

  html(webview) {
    const style = webview.asWebviewUri(
      vscode.Uri.joinPath(this.context.extensionUri, "media", "sidebar.css")
    );
    const script = webview.asWebviewUri(
      vscode.Uri.joinPath(this.context.extensionUri, "media", "sidebar.js")
    );
    const nonce = String(Date.now());
    return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';" />
  <link rel="stylesheet" href="${style}" />
</head>
<body>
  <div id="app">
    <div id="header">
      <span id="title">Asteria</span>
      <span id="model-wrap"><span class="dot"></span><span id="model">connecting</span></span>
    </div>
    <div id="meta">正在启动本机 Agent…</div>
    <div id="log"></div>
    <div id="composer">
      <input id="input" placeholder="Ask Asteria..." />
      <button id="send" aria-label="发送">➤</button>
    </div>
  </div>
  <script nonce="${nonce}" src="${script}"></script>
</body>
</html>`;
  }

  onMessage(message) {
    if (!this.child || !this.child.stdin.writable) {
      this.post({ type: "error", message: "Asteria 进程未运行。请先 cargo build --release。" });
      return;
    }
    const payload = (() => {
      switch (message.type) {
        case "ask":
          return { method: "ask", params: { text: message.text } };
        case "approve":
          return { method: "approve", params: { approval_id: message.id } };
        case "deny":
          return { method: "deny", params: { approval_id: message.id } };
        case "cancel":
          return { method: "cancel" };
        default:
          return null;
      }
    })();
    if (payload) {
      this.child.stdin.write(`${JSON.stringify(payload)}\n`);
    }
  }

  start() {
    this.stop();
    const folder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (!folder) {
      this.post({ type: "error", message: "请先打开 Asteria 仓库作为工作区。" });
      return;
    }
    const binary = findBinary(folder);
    if (!binary) {
      this.post({
        type: "error",
        message: "找不到 asteria-agent。请在仓库根目录运行 cargo build --release。",
      });
      return;
    }
    this.child = spawn(binary, ["--rpc"], {
      cwd: folder,
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child.on("error", (error) => {
      this.post({ type: "error", message: String(error) });
    });
    this.child.on("exit", (code) => {
      this.post({ type: "offline", text: `Asteria 进程已退出（${code ?? "null"}）` });
    });
    readline.createInterface({ input: this.child.stdout }).on("line", (line) => {
      this.onEvent(line);
    });
    readline.createInterface({ input: this.child.stderr }).on("line", (line) => {
      if (line.trim()) {
        this.post({ type: "log", text: line });
      }
    });
  }

  onEvent(line) {
    try {
      const event = JSON.parse(line);
      const mapped = mapEvent(event);
      if (mapped) {
        this.post(mapped);
      }
    } catch {
      if (line.trim()) {
        this.post({ type: "log", text: line });
      }
    }
  }

  post(message) {
    this.view?.webview.postMessage(message);
  }

  stop() {
    if (this.child) {
      this.child.kill();
      this.child = undefined;
    }
  }
}

function findBinary(workspace) {
  const candidates = [
    process.env.ASTERIA_BIN,
    path.join(workspace, "target/release/asteria-agent"),
    path.join(workspace, "target/debug/asteria-agent"),
  ].filter(Boolean);
  return candidates.find((candidate) => fs.existsSync(candidate));
}

function mapEvent(event) {
  switch (event.event) {
    case "ready":
      return {
        type: "ready",
        model: event.model,
        rag: event.rag,
        mcp: event.mcp,
        currentSession: event.current_session,
        sessions: event.sessions,
      };
    case "user":
      return { type: "user", text: event.text };
    case "command":
      return { type: "command", text: event.text };
    case "command_result":
      return { type: "commandResult", text: event.text };
    case "status":
      return { type: "status", text: event.text };
    case "assistant_begin":
      return { type: "assistantBegin" };
    case "assistant_delta":
      return { type: "assistantDelta", text: event.text };
    case "tool_started":
      return { type: "toolStarted", name: event.name };
    case "tool_result":
      return { type: "toolResult", name: event.name, ok: event.ok, ms: event.ms };
    case "approval":
      return { type: "approval", id: event.id, tool: event.tool, arguments: event.arguments };
    case "turn_completed":
      return {
        type: "turnCompleted",
        steps: event.steps,
        tokens: event.tokens,
      };
    case "turn_cancelled":
      return { type: "turnCancelled" };
    case "session_cleared":
      return { type: "sessionCleared" };
    case "session_changed":
      return {
        type: "sessionChanged",
        currentId: event.current_id,
        title: event.title,
        messageCount: event.message_count,
        resetView: event.reset_view,
        sessions: event.sessions,
      };
    case "error":
      return { type: "error", message: event.message };
    case "log":
      return { type: "log", text: event.text };
    default:
      return null;
  }
}

async function openAsteria() {
  await vscode.commands.executeCommand("workbench.view.explorer");
  await vscode.commands.executeCommand("asteria.chat.focus");
}

function activate(context) {
  const provider = new AsteriaViewProvider(context);
  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 1000);
  status.text = "$(comment-discussion) Asteria";
  status.tooltip = "打开 Asteria";
  status.command = "asteria.focus";
  status.show();
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("asteria.chat", provider, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
    vscode.commands.registerCommand("asteria.focus", () => openAsteria()),
    status,
    { dispose: () => provider.stop() }
  );
  void openAsteria();
  void vscode.window
    .showInformationMessage("Asteria 在左边文件列表最下面，也可点窗口底部的 Asteria", "打开")
    .then((choice) => {
      if (choice === "打开") {
        void openAsteria();
      }
    });
}

function deactivate() {}

module.exports = { activate, deactivate };
