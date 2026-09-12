# Asteria

一个 Rust 命令行 Agent，使用 DeepSeek，支持多轮对话、工具调用、上下文预算与摘要、请求重试和 Token 统计。

## 运行

```bash
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY
cargo run --release
```

输入 `/context` 查看原始记忆，`/usage` 查看最近 Turn 与会话累计用量，`/reset` 清空记忆（保留用量统计），`/exit` 退出。

会话默认持久化到 `.asteria/session.jsonl`，也可以通过 `ASTERIA_SESSION_PATH` 指定路径。成功完成的 Turn 会写入 User、Assistant、Tool 消息和 Token Usage；启动时按完成标记重放。失败或取消的 Turn 不会写入，半写入的未完成批次会被忽略。

用于测试工具取消和超时：输入“请调用 wait_for 工具等待 60 秒”，等待期间输入 `/cancel` 并回车。`wait_for` 是专用测试工具，只等待，不进行系统操作；参数范围为 1～120 秒，工具默认超时为 30 秒。

## 工具权限

当前会话支持 `allow`（直接执行）、`deny`（拒绝执行）和 `ask`（本次调用需批准）。三个内置工具默认 allow；通过 Rust API `register()` 注册的新工具默认 ask，无审批处理器时拒绝执行。明确允许可使用 `register_with_permission(tool, ToolPermission::Allow)`。

```text
/permissions
/permission wait_for ask
请调用 wait_for 等待 5 秒。
```

出现 `[待批准 #1]` 后查看工具名称和完整参数，再输入 `/approve 1` 或 `/deny 1`。编号以实际显示为准，每个请求不同；批准只针对当前一次调用，不会把工具永久改成 allow。`/cancel` 取消整轮并使旧审批编号失效。没有当前审批时，`/approve` 不会预先授权下一次调用。

权限只作用于工具执行，不会阻止模型自行输出文本。deny 或人工拒绝产生 `is_error=true` 的 Tool Result，供模型解释；不要以模型说“已执行”为证据，应查看 `/context`。同批次中 allow 工具可以继续运行，ask 只阻止尚未批准的对应工具。

等待批准不占用工具执行的 30 秒超时；没有输入通道时默认拒绝，避免脚本永久挂起。`/reset` 保留会话权限，重启恢复默认设置。本阶段没有磁盘规则、路径匹配或沙箱。权限配置在空闲时生效；运行中的 `/permission` 会排队到本轮结束，想立即停止请用 `/cancel`。

运行真实 DeepSeek 与 TUI 权限回归（产生 API 用量）：

```bash
cargo build
python3 tests/permission_tui.py
```

脚本需要 `tests/requirements-terminal.txt` 中的依赖；验证批准、拒绝、取消后过期编号无效，以及 reset 不清除权限。

交互终端由 Reedline 统一绘制 `你: ` 输入行，支持中文宽度、退格、左右移动、跨行和粘贴。回答、排队提示、重试日志出现时，会输出在输入行上方并恢复尚未提交的草稿。输入历史只保存在内存，不自动写历史文件。

等待回答期间仍可输入，收到普通输入后显示 `[已排队]` 和待处理条数。当前轮结束后显示 `[开始处理排队输入]` 及对应内容，再按顺序执行；空行不排队。`/cancel` 立即取消当前轮，其余输入（包括 `/context`、`/usage`、`/reset`、`/exit`）按队列顺序处理。取消当前轮不会清空已排队的输入。管道、重定向或 `TERM=dumb` 使用纯文本模式，不启用光标编辑。

如果内嵌终端没有将 Ctrl+C 传给进程，在等待模型回答期间输入 `/cancel` 并按回车，同样可以取消当前 Turn。程序会打印 `[Cancel] 收到 ...`，随后完成回滚并显示 `Cancelled`。这提供了不依赖快捷键的取消入口，并不表示终端的按键传递问题已修复。

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
- `src/tools.rs`：工具定义与执行
- `src/permission.rs`：权限值与异步审批通道
- `src/approval_ui.rs`：TUI 待审批请求及单次批准/拒绝命令
- `src/main.rs`：命令行交互
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
