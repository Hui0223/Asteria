# Asteria

一个 Rust 命令行 Agent，使用 DeepSeek，支持多轮对话、工具调用、上下文预算与摘要、请求重试和 Token 统计。

## 运行

```bash
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY
cargo run --release
```

输入 `/context` 查看原始记忆，`/usage` 查看最近 Turn 与会话累计用量，`/reset` 清空记忆（保留用量统计），`/exit` 退出。

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
- `src/main.rs`：命令行交互

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

## 回滚

初始可运行版本标记为 `asteria-v0.1-working`。需要临时查看旧版本时：

```bash
git switch --detach asteria-v0.1-working
```

返回最新版本：

```bash
git switch master
```
