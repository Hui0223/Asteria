# Asteria

一个 Rust 命令行 Agent，使用 DeepSeek，支持多轮对话、内置文件与 Shell 工具、可选 MCP、RAG、上下文预算与摘要、请求重试和 Token 统计。

## 本地 RAG 示例

RAG 检索器位于 `src/rag.rs`，读取本地 UTF-8 或 DOCX 文档、按结构切分，再用 BM25 风格评分排序相关片段。明确文件名时会在该文档内部继续按问题检索；简称可匹配较长的真实文件名，错误文件名会回退到全库检索。常见中文故障问法会扩展为英文手册术语，例如“LT 反复重启”可匹配 `LT Continuous Rebooting`。检索输出按 Token 预算限制，并按规范化内容哈希去重。它不依赖向量数据库或 API Key，可以直接本地运行：

```bash
cargo run --bin asteria-rag -- docs/rag-demo "Context Kernel"
# 启动持续对话式 RAG（需要 .env 中的 API Key）
cargo run --bin asteria-rag -- --generate docs/rag-demo
# 远程知识源只需提供一次，之后连续输入问题
cargo run --bin asteria-rag -- --generate --url https://raw.githubusercontent.com/Hui0223/Asteria/master/README.md
```

默认命令会打印知识库片段数量、来源、匹配分数和增强 Prompt；它展示 RAG 的 Retrieval 和 Augmentation。主 TUI 和 `--generate` 会把检索注册为 `search_docs` 工具：用户消息保持原问题，完整 Tool Result 只在当前 Turn 生成答案时可见，Turn 完成后即从上下文移除。输入 `/exit` 退出。

## 运行

```bash
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY
cargo run --release
```

输入 `/context` 查看原始记忆，`/usage` 查看最近 Turn 与会话累计用量，`/trace` 查看最近 Turn 的脱敏工具审计，`/trace <turn_id>` 查询指定 Turn，`/mcp status` 查看 MCP Server 状态，`/reset` 清空记忆（保留用量统计），`/exit` 退出。

独立聊天窗口不依赖 Cursor，会弹出自己的桌面窗口并拉起 `asteria-agent --rpc`：

```bash
cargo build --bin asteria-agent
cargo run --bin asteria-app
```

终端 TUI 仍是 `cargo run`。Cursor 侧栏是可选入口，见 `extensions/asteria/README.md`。

## 核心文件系统与 Shell 工具

文件系统能力属于 Asteria 内置工具，不依赖 MCP Server：

- `Read`：按 UTF-8 读取文件，支持 1-based `offset` 和 `limit`，输出稳定行号。
- `Write`：新建或完整覆盖文件；父目录必须已经存在。
- `Edit`：用 `old_string`/`new_string` 精确修改文件。默认要求唯一匹配，多处替换必须显式设置 `replace_all=true`。
- `Glob`：使用 `**/*.rs` 一类 Glob pattern 按文件名查找。
- `Grep`：使用 Rust 正则表达式按文件内容搜索，可用 Glob 限定文件。
- `Bash`：在工作区目录启动用户 Shell，可用于列目录、移动文件和其他系统操作。

`Read`、`Glob`、`Grep` 默认 `allow`；`Write`、`Edit`、`Bash` 默认 `ask`。五个文件工具会规范化路径并拒绝工作区外访问，且不会跟随目录遍历中的符号链接。`Bash` 是明确的高风险任意命令边界：虽然默认工作目录位于工作区，但获批命令仍可访问系统其他位置，因此执行前必须检查完整参数。

文件最大读取 10 MiB，单次写入及修改后的文件最大 5 MiB；搜索结果和工具输出还有独立数量、字符上限。`Write`/`Edit` 的正文不会进入 ToolTrace 参数预览，只保存参数哈希。

## 可选 MCP（stdio 与 Streamable HTTP）

MCP 只用于额外接入第三方工具，不承担 Asteria 核心文件能力。Asteria 使用官方 Rust SDK `rmcp`，用户级配置位于 `~/.asteria/mcp.json`，项目级配置位于 `.asteria/mcp.json`：

```json
{
  "mcpServers": {
    "local": {
      "command": "your-mcp-server",
      "args": ["--workspace", "${workspaceFolder}"],
      "enabledTools": ["tool_name"],
      "startupTimeoutMs": 30000,
      "toolTimeoutMs": 60000
    },
    "cloudflare-docs": {
      "url": "https://docs.mcp.cloudflare.com/mcp",
      "toolTimeoutMs": 30000
    },
    "binance-mcp-server": {
      "url": "https://agent.binance.com/mcp/agentic",
      "oauthClientId": "grok",
      "toolTimeoutMs": 30000
    }
  }
}
```

每个 Server 必须且只能配置 `command` 或 `url`。`${workspaceFolder}` 会在 stdio 的 `args` 和 `cwd` 中解析为 Asteria 启动目录；`env` 和远程 `headers` 支持 `${NAME}` 环境变量引用。远程 URL 必须使用 HTTPS，只有 localhost 允许 HTTP。`Authorization` 必须写成 `"Bearer ${TOKEN_NAME}"`，明文 Token 会被拒绝，解析后的敏感值不会写入 Schema、TUI 或 ToolTrace。需要浏览器 OAuth 的远程 Server 可设置 `oauthClientId`；凭据保存在用户目录，不进入 mcp.json。`enabledTools` 省略时允许发现到的全部工具；`disabledTools` 始终优先。

项目配置中的同名 Server 会覆盖用户配置。TUI 默认加载当前工作区的 `.asteria/mcp.json`；工具仍默认 `ask`，调用前会征求批准。若当前仓库不可信，可跳过项目配置：

```bash
cargo run --release -- --no-project-mcp
```

空闲时也可用 `/mcp trust` 加载项目配置，或 `/mcp untrust` 重新跳过。`--trust-project-mcp` 仍被接受，但已不再需要。测试可设置 `ASTERIA_NO_PROJECT_MCP=1`，避免连上本机项目里的 Server。

远端工具以 `mcp__<server>__<tool>` 注册，避免和核心工具重名，默认权限为 `ask`。可以先用 `/mcp` 和 `/permissions` 检查，再批准单次调用或用 `/permission mcp__external__tool_name allow` 修改会话权限。权限保存在 `state.json`，调用的脱敏审计保存在 `trace.jsonl`。

MCP 配置和连接支持在 TUI 空闲状态动态管理：

```text
/mcp status
/mcp tools
/mcp trust
/mcp untrust
/mcp reload
/mcp connect external
/mcp disconnect external
/mcp auth binance-mcp-server
/mcp logout binance-mcp-server
```

`reload` 会从磁盘重新读取用户级和受信任的项目级配置，先建立完整候选连接；所有启用的 Server 均可用后才整体替换当前连接，失败时旧连接和工具保持不变。`connect` 只验证并替换指定 Server，`disconnect` 会关闭子进程并立即从模型 Schema 中注销其工具。同名工具在重载时继承当前权限，已删除工具的过期权限会从 `state.json` 清理；每次生命周期操作的成功或失败均写入 `trace.jsonl`，不会进入模型上下文。

MCP 原始 Tool Result 只在当前 Turn 生成答案时可见，随后从 `ContextMemory` 和 `context.jsonl` 中移除；后续上下文保留用户问题与最终回答，审计仍保留工具名、脱敏参数、结果哈希、状态和耗时。所有工具结果还受 Registry 的 8000 字符上限约束，图片、音频和二进制资源不会把 Base64 数据注入模型上下文。

当前支持 stdio 和 Streamable HTTP Tools，包括无会话的远程 Server、HTTP Header、Bearer Token、OAuth 2.1 PKCE 登录、独立启动/调用超时和会话过期重建。需要浏览器登录的 Server 会显示 `auth_required`；输入 `/mcp auth <server>` 或 `/mcp connect <server>` 会打开系统浏览器完成授权。OAuth 凭据保存在 `~/.asteria/oauth/<server>.json`，权限为 `0600`，不会写入会话或 ToolTrace。`oauthClientId` 用于 Binance 这类只接受预注册客户端的服务；未设置时依次尝试 Client ID Metadata Document 和 Dynamic Client Registration。Resources 和 Prompts 留待后续阶段。

项目示例配置还包含四组常用只读远程工具：

- Context7：`resolve-library-id`、`query-docs`，查询 Rust、JavaScript 等第三方库的最新文档。
- DeepWiki：`ask_question`、`read_wiki_structure`、`read_wiki_contents`，读取公开 GitHub 仓库的结构化知识。
- Microsoft Learn：`microsoft_docs_search`、`microsoft_code_sample_search`、`microsoft_docs_fetch`，检索微软官方文档与代码示例。
- GitHub：`get_me`、`get_file_contents`、`search_code`、`issue_read`、`pull_request_read`，通过官方远程 MCP 只读访问账号与仓库信息。启动前在 `.env` 或进程环境中设置最小权限的 `GITHUB_TOKEN`；配置不会接受明文 Token。
- Binance MCP Server：`https://agent.binance.com/mcp/agentic`，通过 OAuth 连接官方 Agentic 子账户，可读行情、查余额，并在你确认后交易 Spot / Margin / Convert / 合约或在子账户钱包之间划转。没有出金权限。首次使用输入 `/mcp auth binance-mcp-server`，在桌面浏览器登录 Binance 并授权；资金需你在 Binance 网页手动转入 Agentic 子账户。交易与划转工具默认 `ask`。

这些 Server 均使用 Streamable HTTP 且无需本地 Node.js。示例配置通过 `enabledTools` 固定允许列表，避免远端新增工具后未经审查自动进入模型 Schema。`fixture` 只用于协议回归测试，不属于生产工具。

模型请求经过动态 Tool Router：工作区只读发现工具、`calculate` 和 `current_time` 保持可见，写入/Shell 工具按代码修改意图加入；MCP 工具根据 Server 名、工具名、描述和用户问题评分，每个 Step 默认最多注入 8 个候选。用户明确点名的 MCP 工具和当前 Turn 已调用的工具始终保留，因此路由不会破坏强制调用或工具结果续步。Registry 仍保存全部工具并负责权限与执行，Router 只缩小发送给模型的 Schema，不会绕过审批。

会话命令：`/session` 显示当前会话目录及三个持久化文件；`/new-session` 清空当前会话、重置 Turn 编号和 Session Token 统计，开始一个全新的会话。

会话默认使用 `.asteria/session/`，也可以通过 `ASTERIA_SESSION_PATH` 指定目录：

```text
.asteria/session/
├── context.jsonl  # 可恢复为 ContextMemory 的已完成 Turn 消息
├── trace.jsonl    # AgentEvent、ToolTrace 和 RAG 来源审计
└── state.json     # 格式版本、下一个 Turn ID、累计用量和工具权限
```

`context.jsonl` 是唯一用于恢复模型记忆的文件。普通工具 Turn 会保存完整消息；RAG Turn 只保存 User 与最终 Assistant，`search_docs` 正文不会进入后续上下文。未出现 `TurnCompleted` 的半批消息不会恢复。

`trace.jsonl` 从物理上与上下文隔离，不会发送给模型。每次已完成的工具调用保存脱敏 `ToolTrace`，包含 Turn、Step、工具名、参数预览、参数/结果哈希、状态、耗时和完成时间，但不保存完整结果。失败或取消的 Turn 不写入可重放消息，已经完成的工具 Trace 仍可保留。

旧版 `.asteria/session.jsonl` 会在首次启动时自动拆分到上述三个文件，原文件保留作为迁移备份。旧的 `ASTERIA_SESSION_PATH=/path/name.jsonl` 配置也兼容，新目录将使用 `/path/name/`。

用于测试工具取消和超时：输入“请调用 wait_for 工具等待 60 秒”，等待期间输入 `/cancel` 并回车。`wait_for` 是专用测试工具，只等待，不进行系统操作；参数范围为 1～120 秒，工具默认超时为 30 秒。

## 工具权限

当前会话支持 `allow`（直接执行）、`deny`（拒绝执行）和 `ask`（本次调用需批准）。只读内置工具默认 allow，写入、Shell 和动态 MCP 工具默认 ask；通过 Rust API `register()` 注册的新工具也默认 ask，无审批处理器时拒绝执行。明确允许可使用 `register_with_permission(tool, ToolPermission::Allow)`。

```text
/permissions
/permission wait_for ask
请调用 wait_for 等待 5 秒。
```

出现 `需要批准 #1` 面板后查看工具名称和格式化参数预览，再输入 `/approve 1` 或 `/deny 1`。编号以实际显示为准，每个请求不同；批准只针对当前一次调用，不会把工具永久改成 allow。`/cancel` 取消整轮并使旧审批编号失效。没有当前审批时，`/approve` 不会预先授权下一次调用。

权限只作用于工具执行，不会阻止模型自行输出文本。deny 或人工拒绝产生 `is_error=true` 的 Tool Result，供模型解释；不要以模型说“已执行”为证据，应查看 `/trace`。同批次中 allow 工具可以继续运行，ask 只阻止尚未批准的对应工具。

等待批准不占用工具执行的 30 秒超时；没有输入通道时默认拒绝，避免脚本永久挂起。`/reset` 保留会话权限，重启恢复默认设置。本阶段没有磁盘规则、路径匹配或沙箱。权限配置在空闲时生效；运行中的 `/permission` 会排队到本轮结束，想立即停止请用 `/cancel`。

运行真实 DeepSeek 与 TUI 权限回归（产生 API 用量）：

```bash
cargo build
python3 tests/permission_tui.py
```

脚本需要 `tests/requirements-terminal.txt` 中的依赖；验证批准、拒绝、取消后过期编号无效，以及 reset 不清除权限。

交互终端由 Reedline 使用单行 `User：` 提示接收输入：问题和标签都以加粗亮绿色显示，提交后只保留一行，并在 `Asteria` 回答前空一行。斜杠命令回显为 `› /command`，不套用问题样式。DeepSeek 以 SSE 推送 `assistant.delta`；TUI 合并换行或约 80 字后按行刷新 `Asteria` 回答，Reedline 无法在同一行逐 token 重绘。增量事件不写入 `trace.jsonl`。输入行支持中文宽度、退格、左右移动、跨行和粘贴。回答、排队提示、重试日志出现时，会输出在输入行上方并恢复尚未提交的草稿。默认使用紧凑事件视图：隐藏 Step、call ID 和重复权限日志，工具结束时合并为一行，每轮回答后只显示一行 Step、Token、重试和耗时摘要；完整统计仍由 `/usage` 提供。输入 `/verbose on` 可临时恢复详细 AgentEvent，`/verbose off` 返回紧凑模式。输入历史只保存在内存，不自动写历史文件。

等待回答期间仍可输入，收到普通输入后显示 `[已排队]` 和待处理条数。当前轮结束后显示 `[开始处理排队输入]` 及对应内容，再按顺序执行；空行不排队。`/cancel` 立即取消当前轮，其余输入（包括 `/context`、`/usage`、`/trace`、`/reset`、`/exit`）按队列顺序处理。取消当前轮不会清空已排队的输入。管道、重定向或 `TERM=dumb` 使用纯文本模式，不启用光标编辑。

如果内嵌终端没有将 Ctrl+C 传给进程，在等待模型回答期间输入 `/cancel` 并按回车，同样可以取消当前 Turn。程序会打印取消提示，随后完成回滚并显示 `Turn 已取消`。这提供了不依赖快捷键的取消入口，并不表示终端的按键传递问题已修复。

模型请求或重试等待期间按 **Ctrl+C**，取消当前 Turn、回滚本轮新增消息，并回到输入提示。空闲时 Ctrl+C 只提示退出方法，不结束程序。每轮使用新的取消令牌，上一轮的取消不会影响下一轮。

HTTP 连接超时为 10 秒，单次请求超时为 120 秒。取消会停止本地等待，不保证服务端停止生成；只有已返回的 usage 能计入统计。当前计算与时间工具仍同步执行，在工具边界检查取消；这不提供任意长运行工具的强制中断。

API Key 只保存在本地 `.env` 中，请勿提交。

默认模型为 `deepseek-v4-flash`，也可通过 `DEEPSEEK_MODEL` 修改。

## 代码结构

- `src/agent.rs`：Agent 门面和诊断接口
- `src/agent_loop.rs`：异步 Turn/Step 调度、取消、回滚与用量
- `src/provider/`：异步模型接口与 DeepSeek HTTP 请求
- `src/retry.rs`：错误分类、退避及可取消的异步等待
- `src/context_builder.rs`：请求预算与历史摘要
- `src/context.rs`：强类型上下文、校验、检查点与失败回滚
- `src/message.rs`：内部消息模型
- `src/tools.rs`：统一工具 Registry、权限门和执行边界
- `src/core_tools.rs`：内置 Read/Write/Edit/Glob/Grep/Bash
- `src/mcp/config.rs`：用户级/项目级 MCP 配置加载、覆盖与信任边界
- `src/mcp/manager.rs`：stdio Server 生命周期、工具发现和连接状态
- `src/mcp/tool.rs`：MCP Tool 到 AgentTool 的 Schema、调用与结果适配
- `src/permission.rs`：权限值与异步审批通道
- `src/tui/`：事件状态归约、流式 LiveView、紧凑/详细渲染、审批面板与 TUI EventSink
- `src/main.rs`：命令分发和 Turn 编排
- `src/terminal.rs`：中文行编辑、统一后台输出、终端退出恢复

库调用已改为异步：`agent.ask(input).await`、`agent.ask_with_cancel(input, &cancel).await`。
取消时调用 `cancel.cancel()`，并等待 Turn Future 返回以完成回滚；不要直接丢弃或 abort 整个 Turn Future。

## 取消测试

```bash
cargo test cancels_
cargo test cancellation_keeps_ready_response_usage
cargo test --test failure_context
```

`cancels_http_*` 使用本地真实 TCP 连接，分别覆盖等待响应头和 JSON 正文读取中取消；不会调用 DeepSeek。
Loop 测试另行验证退避取消、已返回用量保留和下一轮继续执行。

真实服务手动验证：运行 `cargo run`，先让 Agent 记住一个代号；下一轮要求长回答，等待期间按 Ctrl+C；执行 `/context` 确认仅保留取消前的历史；再询问代号，确认仍能正常回答。

本次实际验证：Turn 1 记住 `ORION-731`，报告总用量 350；Turn 2 在真实请求等待时取消，原始历史仍为 2 条，已报告的会话用量保持 350；Turn 3 正确回复 `ORION-731`，会话用量变为 719。另一次进程使用不可用的本地代理，在 500ms 退避中按 Ctrl+C，得到 `Cancelled` 且消息数为 0。

## 终端编辑回归测试

在 Python 测试环境中安装 `tests/requirements-terminal.txt`，然后执行：

```bash
cargo build
python3 tests/terminal_editing.py
# 额外验证真实 DeepSeek 回答和取消，会产生 API 用量
python3 tests/terminal_editing.py --live
```

脚本使用真实 PTY 启动当前二进制，并通过 pyte 解析 ANSI 屏幕，断言输入行没有中文残影。覆盖 80/24 列显示、中文标点删除、组合字符、光标移动、跨行删除、粘贴、重试输出期间编辑、Ctrl+C 及 `/exit` 后恢复终端模式。

## 回滚

初始可运行版本标记为 `asteria-v0.1-working`。需要临时查看旧版本时：

```bash
git switch --detach asteria-v0.1-working
```

返回最新版本：

```bash
git switch master
```
