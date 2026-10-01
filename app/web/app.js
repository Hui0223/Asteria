const log = document.getElementById("log");
const input = document.getElementById("input");
const send = document.getElementById("send");
const sessionList = document.getElementById("session-list");
let assistant;
let currentSessionId = "";

function post(message) {
  if (window.ipc && typeof window.ipc.postMessage === "function") {
    window.ipc.postMessage(JSON.stringify(message));
    return;
  }
  console.error("ipc unavailable", message);
}

function append(html) {
  const node = document.createElement("div");
  node.innerHTML = html;
  while (node.firstChild) {
    log.appendChild(node.firstChild);
  }
  log.scrollTop = log.scrollHeight;
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function beginAnswer() {
  const card = document.createElement("div");
  card.className = "answer";
  const label = document.createElement("div");
  label.className = "asteria-label";
  label.textContent = "Asteria";
  assistant = document.createElement("div");
  assistant.className = "assistant";
  card.appendChild(label);
  card.appendChild(assistant);
  log.appendChild(card);
}

function renderSessions(sessions, currentId) {
  if (!sessionList) {
    return;
  }
  currentSessionId = currentId || "";
  sessionList.innerHTML = "";
  (sessions || []).forEach((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "session" + (session.id === currentSessionId ? " active" : "");
    button.dataset.id = session.id;
    button.textContent = session.title || session.id;
    button.title = session.id;
    sessionList.appendChild(button);
  });
}

function ask() {
  const text = input.value.trim();
  if (!text) {
    return;
  }
  post({ type: "ask", text });
  input.value = "";
}

function handle(message) {
  switch (message.type) {
    case "ready":
      document.getElementById("model").textContent = message.model;
      document.getElementById("model-wrap").classList.remove("offline");
      document.getElementById("rag-pill").textContent = message.rag;
      document.getElementById("mcp-pill").textContent = message.mcp;
      renderSessions(message.sessions, message.currentSession);
      break;
    case "user":
      assistant = null;
      append(`<div class="user">User：${escapeHtml(message.text)}</div>`);
      break;
    case "command":
      assistant = null;
      append(`<div class="slash">› ${escapeHtml(message.text)}</div>`);
      break;
    case "commandResult":
      append(`<pre class="command-result">${escapeHtml(message.text)}</pre>`);
      break;
    case "status":
      append(`<div class="status">… ${escapeHtml(message.text)}</div>`);
      break;
    case "assistantBegin":
      beginAnswer();
      break;
    case "assistantDelta":
      if (!assistant) {
        beginAnswer();
      }
      assistant.textContent += message.text;
      log.scrollTop = log.scrollHeight;
      break;
    case "toolStarted":
      append(
        `<div class="tool"><span class="pending">…</span>${escapeHtml(message.name)}</div>`
      );
      break;
    case "toolResult":
      append(
        `<div class="tool"><span class="${message.ok ? "ok" : "fail"}">${message.ok ? "✓" : "×"}</span>${escapeHtml(message.name)} · ${message.ms}ms</div>`
      );
      break;
    case "approval":
      append(`
        <div class="approval" data-id="${message.id}">
          <h3>需要批准 #${message.id}</h3>
          <div class="tool-name">${escapeHtml(message.tool)}</div>
          <pre>${escapeHtml(message.arguments)}</pre>
          <div class="row">
            <button class="allow" data-act="approve" data-id="${message.id}">Allow</button>
            <button class="deny" data-act="deny" data-id="${message.id}">Deny</button>
          </div>
        </div>`);
      break;
    case "turnCompleted":
      assistant = null;
      append(`<div class="log">${message.steps} steps · ${message.tokens} tokens</div>`);
      break;
    case "turnCancelled":
      assistant = null;
      append(`<div class="log">Turn 已取消</div>`);
      break;
    case "sessionChanged":
      renderSessions(message.sessions, message.currentId);
      if (message.resetView) {
        assistant = null;
        log.innerHTML = "";
        append(
          `<div class="log">已切换到 ${escapeHtml(message.title || "新对话")}（${message.messageCount || 0} 条消息）。</div>`
        );
      }
      break;
    case "sessionCleared":
      break;
    case "error":
      append(`<div class="error">${escapeHtml(message.message)}</div>`);
      break;
    case "offline":
      document.getElementById("model-wrap").classList.add("offline");
      document.getElementById("model").textContent = "offline";
      append(`<div class="log">${escapeHtml(message.text)}</div>`);
      break;
    case "log":
      append(`<div class="log">${escapeHtml(message.text)}</div>`);
      break;
    default:
      break;
  }
}

window.__asteriaPush = handle;

send.addEventListener("click", ask);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    ask();
  }
});
document.getElementById("new-chat").addEventListener("click", () => {
  post({ type: "new_session" });
});
sessionList.addEventListener("click", (event) => {
  const button = event.target.closest("button.session[data-id]");
  if (!button || button.dataset.id === currentSessionId) {
    return;
  }
  post({ type: "switch_session", id: button.dataset.id });
});
log.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-act]");
  if (!button) {
    return;
  }
  post({ type: button.dataset.act, id: Number(button.dataset.id) });
  button.parentElement.querySelectorAll("button").forEach((item) => {
    item.disabled = true;
  });
});
