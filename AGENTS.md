# Asteria 项目协作约定

## GitHub 同步

- 远程仓库：`https://github.com/Hui0223/Asteria.git`
- 默认分支：`master`
- 每次代码修改完成后，先运行 `cargo fmt`、`cargo test` 和 `cargo clippy --all-targets -- -D warnings`。
- 验证通过后创建清晰的 Conventional Commit，再推送到 `origin/master`。
- 不提交 `.env`、API Key、`.asteria/` 会话数据、`.DS_Store` 或临时测试输出。
- Git 凭据只从用户本机配置读取，不写入项目文件、提交或日志。

## 当前基准

- 用户确认的稳定基准标签：`asteria-baseline-2026-09-10`
- 当前架构包含 Context Kernel、预算化上下文、异步 DeepSeek、Turn/Step Loop、重试、Tool Registry、权限、事件流和 JSONL Session Persistence。
