# Asteria

一个百余行的 Rust 命令行 Agent，使用 DeepSeek，支持多轮对话、函数工具调用、数学计算和本地时间查询。

## 运行

```bash
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY
cargo run --release
```

输入 `/reset` 清空对话记忆，输入 `/exit` 退出。API Key 只保存在本地 `.env` 中，请勿提交。

默认模型为 `deepseek-v4-flash`，也可通过 `DEEPSEEK_MODEL` 修改。

## 代码结构

- `src/agent.rs`：Agent Loop 与 DeepSeek 请求
- `src/context.rs`：强类型上下文、校验、检查点与失败回滚
- `src/message.rs`：内部消息模型
- `src/tools.rs`：工具定义与执行
- `src/main.rs`：命令行交互

## 回滚

初始可运行版本标记为 `asteria-v0.1-working`。需要临时查看旧版本时：

```bash
git switch --detach asteria-v0.1-working
```

返回最新版本：

```bash
git switch master
```
