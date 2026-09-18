use crate::{
    agent_loop::CancelToken,
    permission::ToolPermission,
    tools::{AgentTool, ToolExecutionContext, ToolOutput, ToolRegistry},
};
use anyhow::{Context, Result, bail};
use globset::Glob;
use regex::RegexBuilder;
use serde_json::{Map, Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use walkdir::WalkDir;

const MAX_FILE_BYTES: u64 = 10 * 1024 * 1024;
const MAX_WRITE_BYTES: usize = 5 * 1024 * 1024;
const DEFAULT_READ_LINES: usize = 2_000;
const MAX_READ_LINES: usize = 5_000;
const MAX_SEARCH_RESULTS: usize = 2_000;

#[derive(Clone)]
struct Workspace {
    root: Arc<PathBuf>,
}

impl Workspace {
    fn current() -> Self {
        let root = std::env::current_dir()
            .ok()
            .and_then(|path| fs::canonicalize(path).ok())
            .unwrap_or_else(|| PathBuf::from("."));
        Self {
            root: Arc::new(root),
        }
    }

    fn resolve_existing(&self, raw_path: &str) -> Result<PathBuf> {
        let candidate = self.candidate(raw_path)?;
        let resolved = fs::canonicalize(&candidate)
            .with_context(|| format!("路径不存在或不可访问: {}", candidate.display()))?;
        self.ensure_inside(&resolved)?;
        Ok(resolved)
    }

    fn resolve_for_write(&self, raw_path: &str) -> Result<PathBuf> {
        let candidate = self.candidate(raw_path)?;
        if candidate.exists() {
            return self.resolve_existing(raw_path);
        }
        let parent = candidate.parent().context("写入路径没有父目录")?;
        let parent = fs::canonicalize(parent)
            .with_context(|| format!("父目录不存在或不可访问: {}", parent.display()))?;
        self.ensure_inside(&parent)?;
        let file_name = candidate.file_name().context("写入路径缺少文件名")?;
        Ok(parent.join(file_name))
    }

    fn resolve_directory(&self, raw_path: &str) -> Result<PathBuf> {
        let path = self.resolve_existing(raw_path)?;
        anyhow::ensure!(path.is_dir(), "目标不是目录: {}", path.display());
        Ok(path)
    }

    fn candidate(&self, raw_path: &str) -> Result<PathBuf> {
        anyhow::ensure!(!raw_path.trim().is_empty(), "path 不能为空");
        let path = Path::new(raw_path);
        Ok(if path.is_absolute() {
            path.to_owned()
        } else {
            self.root.join(path)
        })
    }

    fn ensure_inside(&self, path: &Path) -> Result<()> {
        if !path.starts_with(self.root.as_ref()) {
            bail!("拒绝访问工作区之外的路径: {}", path.display());
        }
        Ok(())
    }

    fn display(&self, path: &Path) -> String {
        path.strip_prefix(self.root.as_ref())
            .ok()
            .filter(|relative| !relative.as_os_str().is_empty())
            .map(|relative| relative.display().to_string())
            .unwrap_or_else(|| ".".into())
    }
}

/// 把 Kimi 风格核心工具注册到 Asteria；读工具 allow，副作用工具 ask。
pub fn register(registry: &mut ToolRegistry) {
    let workspace = Workspace::current();
    registry.register_with_permission(ReadTool(workspace.clone()), ToolPermission::Allow);
    registry.register_with_permission(WriteTool(workspace.clone()), ToolPermission::Ask);
    registry.register_with_permission(EditTool(workspace.clone()), ToolPermission::Ask);
    registry.register_with_permission(GlobTool(workspace.clone()), ToolPermission::Allow);
    registry.register_with_permission(GrepTool(workspace.clone()), ToolPermission::Allow);
    registry.register_with_permission(BashTool(workspace), ToolPermission::Ask);
}

struct ReadTool(Workspace);

#[async_trait::async_trait]
impl AgentTool for ReadTool {
    fn name(&self) -> &str {
        "Read"
    }

    fn schema(&self) -> Value {
        function_schema(
            self.name(),
            "Read a UTF-8 text file inside the workspace with stable line numbers.",
            json!({
                "type":"object",
                "properties":{
                    "path":{"type":"string","description":"Workspace-relative or absolute path inside the workspace"},
                    "offset":{"type":"integer","minimum":1,"description":"First 1-based line, default 1"},
                    "limit":{"type":"integer","minimum":1,"maximum":5000,"description":"Maximum lines, default 2000"}
                },
                "required":["path"]
            }),
        )
    }

    async fn execute(&self, raw_args: &str) -> ToolOutput {
        run_sync(raw_args, |arguments| {
            let path = self
                .0
                .resolve_existing(required_string(arguments, "path")?)?;
            anyhow::ensure!(path.is_file(), "目标不是文件");
            ensure_file_size(&path)?;
            let content = fs::read_to_string(&path)
                .with_context(|| format!("文件不是有效 UTF-8: {}", path.display()))?;
            let offset = optional_usize(arguments, "offset")?.unwrap_or(1);
            anyhow::ensure!(offset > 0, "offset 必须大于 0");
            let limit = optional_usize(arguments, "limit")?
                .unwrap_or(DEFAULT_READ_LINES)
                .min(MAX_READ_LINES);
            anyhow::ensure!(limit > 0, "limit 必须大于 0");
            let lines = content
                .lines()
                .enumerate()
                .skip(offset - 1)
                .take(limit)
                .map(|(index, line)| format!("{}|{line}", index + 1))
                .collect::<Vec<_>>();
            Ok(if lines.is_empty() {
                "(文件为空或 offset 超出范围)".into()
            } else {
                lines.join("\n")
            })
        })
    }
}

struct WriteTool(Workspace);

#[async_trait::async_trait]
impl AgentTool for WriteTool {
    fn name(&self) -> &str {
        "Write"
    }

    fn schema(&self) -> Value {
        function_schema(
            self.name(),
            "Create or completely overwrite a UTF-8 file inside the workspace.",
            json!({
                "type":"object",
                "properties":{
                    "path":{"type":"string"},
                    "content":{"type":"string"}
                },
                "required":["path","content"]
            }),
        )
    }

    async fn execute(&self, raw_args: &str) -> ToolOutput {
        run_sync(raw_args, |arguments| {
            let path = self
                .0
                .resolve_for_write(required_string(arguments, "path")?)?;
            let content = required_value_string(arguments, "content")?;
            anyhow::ensure!(
                content.len() <= MAX_WRITE_BYTES,
                "写入内容超过 {} 字节上限",
                MAX_WRITE_BYTES
            );
            fs::write(&path, content)?;
            Ok(format!(
                "已写入 {}（{} 字节）",
                self.0.display(&path),
                content.len()
            ))
        })
    }
}

struct EditTool(Workspace);

#[async_trait::async_trait]
impl AgentTool for EditTool {
    fn name(&self) -> &str {
        "Edit"
    }

    fn schema(&self) -> Value {
        function_schema(
            self.name(),
            "Replace an exact string in one UTF-8 workspace file. Fails on ambiguous matches unless replace_all is true.",
            json!({
                "type":"object",
                "properties":{
                    "path":{"type":"string"},
                    "old_string":{"type":"string"},
                    "new_string":{"type":"string"},
                    "replace_all":{"type":"boolean","default":false}
                },
                "required":["path","old_string","new_string"]
            }),
        )
    }

    async fn execute(&self, raw_args: &str) -> ToolOutput {
        run_sync(raw_args, |arguments| {
            let path = self
                .0
                .resolve_existing(required_string(arguments, "path")?)?;
            anyhow::ensure!(path.is_file(), "目标不是文件");
            ensure_file_size(&path)?;
            let old = required_value_string(arguments, "old_string")?;
            let new = required_value_string(arguments, "new_string")?;
            anyhow::ensure!(!old.is_empty(), "old_string 不能为空");
            anyhow::ensure!(new.len() <= MAX_WRITE_BYTES, "new_string 超过安全长度上限");
            let replace_all = arguments
                .get("replace_all")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let content = fs::read_to_string(&path)?;
            let matches = content.match_indices(old).count();
            anyhow::ensure!(matches > 0, "old_string 未在文件中找到");
            if !replace_all {
                anyhow::ensure!(
                    matches == 1,
                    "old_string 出现 {matches} 次；请提供更多上下文或设置 replace_all=true"
                );
            }
            let updated = if replace_all {
                content.replace(old, new)
            } else {
                content.replacen(old, new, 1)
            };
            anyhow::ensure!(
                updated.len() <= MAX_WRITE_BYTES,
                "修改后文件超过 {} 字节上限",
                MAX_WRITE_BYTES
            );
            fs::write(&path, updated)?;
            Ok(format!(
                "已修改 {}（替换 {} 处）",
                self.0.display(&path),
                if replace_all { matches } else { 1 }
            ))
        })
    }
}

struct GlobTool(Workspace);

#[async_trait::async_trait]
impl AgentTool for GlobTool {
    fn name(&self) -> &str {
        "Glob"
    }

    fn schema(&self) -> Value {
        function_schema(
            self.name(),
            "Find workspace files by glob pattern such as **/*.rs.",
            json!({
                "type":"object",
                "properties":{
                    "pattern":{"type":"string"},
                    "path":{"type":"string","description":"Directory to search, default workspace root"}
                },
                "required":["pattern"]
            }),
        )
    }

    async fn execute(&self, raw_args: &str) -> ToolOutput {
        run_sync(raw_args, |arguments| {
            let pattern = required_string(arguments, "pattern")?;
            let matcher = Glob::new(pattern)
                .with_context(|| format!("无效 Glob pattern: {pattern}"))?
                .compile_matcher();
            let start = self
                .0
                .resolve_directory(optional_string(arguments, "path").unwrap_or("."))?;
            let mut paths = WalkDir::new(&start)
                .follow_links(false)
                .into_iter()
                .filter_map(std::result::Result::ok)
                .filter(|entry| entry.file_type().is_file())
                .filter_map(|entry| {
                    let relative = entry.path().strip_prefix(&start).ok()?;
                    matcher
                        .is_match(relative)
                        .then(|| self.0.display(entry.path()))
                })
                .take(MAX_SEARCH_RESULTS)
                .collect::<Vec<_>>();
            paths.sort();
            Ok(no_results_or_join(paths))
        })
    }
}

struct GrepTool(Workspace);

#[async_trait::async_trait]
impl AgentTool for GrepTool {
    fn name(&self) -> &str {
        "Grep"
    }

    fn schema(&self) -> Value {
        function_schema(
            self.name(),
            "Search UTF-8 workspace file contents with a regular expression.",
            json!({
                "type":"object",
                "properties":{
                    "pattern":{"type":"string","description":"Rust regular expression"},
                    "path":{"type":"string","description":"File or directory, default workspace root"},
                    "glob":{"type":"string","description":"Optional file glob filter such as **/*.rs"},
                    "case_insensitive":{"type":"boolean","default":false},
                    "max_results":{"type":"integer","minimum":1,"maximum":2000,"default":200}
                },
                "required":["pattern"]
            }),
        )
    }

    async fn execute(&self, raw_args: &str) -> ToolOutput {
        run_sync(raw_args, |arguments| {
            let pattern = required_string(arguments, "pattern")?;
            let case_insensitive = arguments
                .get("case_insensitive")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let regex = RegexBuilder::new(pattern)
                .case_insensitive(case_insensitive)
                .build()
                .with_context(|| format!("无效正则表达式: {pattern}"))?;
            let glob = optional_string(arguments, "glob")
                .map(Glob::new)
                .transpose()
                .context("无效 glob 过滤器")?
                .map(|glob| glob.compile_matcher());
            let limit = optional_usize(arguments, "max_results")?
                .unwrap_or(200)
                .clamp(1, MAX_SEARCH_RESULTS);
            let start = self
                .0
                .resolve_existing(optional_string(arguments, "path").unwrap_or("."))?;
            let entries: Box<dyn Iterator<Item = PathBuf>> = if start.is_file() {
                Box::new(std::iter::once(start))
            } else {
                Box::new(
                    WalkDir::new(start)
                        .follow_links(false)
                        .into_iter()
                        .filter_map(std::result::Result::ok)
                        .filter(|entry| entry.file_type().is_file())
                        .map(|entry| entry.into_path()),
                )
            };
            let mut results = Vec::new();
            for path in entries {
                let relative = path.strip_prefix(self.0.root.as_ref()).unwrap_or(&path);
                if glob.as_ref().is_some_and(|glob| !glob.is_match(relative)) {
                    continue;
                }
                let Ok(metadata) = fs::metadata(&path) else {
                    continue;
                };
                if metadata.len() > MAX_FILE_BYTES {
                    continue;
                }
                let Ok(content) = fs::read_to_string(&path) else {
                    continue;
                };
                for (index, line) in content.lines().enumerate() {
                    if regex.is_match(line) {
                        results.push(format!("{}:{}:{line}", self.0.display(&path), index + 1));
                        if results.len() >= limit {
                            return Ok(results.join("\n"));
                        }
                    }
                }
            }
            Ok(no_results_or_join(results))
        })
    }
}

struct BashTool(Workspace);

impl BashTool {
    async fn run(&self, raw_args: &str, cancel: Option<&CancelToken>) -> ToolOutput {
        let arguments = match parse_arguments(raw_args) {
            Ok(arguments) => arguments,
            Err(error) => return error_output(error),
        };
        let command = match required_string(&arguments, "command") {
            Ok(command) => command,
            Err(error) => return error_output(error),
        };
        let cwd = match self
            .0
            .resolve_directory(optional_string(&arguments, "cwd").unwrap_or("."))
        {
            Ok(cwd) => cwd,
            Err(error) => return error_output(error),
        };
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".into());
        let mut process = tokio::process::Command::new(shell);
        process
            .arg("-lc")
            .arg(command)
            .current_dir(cwd)
            .kill_on_drop(true);
        let output = match cancel {
            Some(cancel) => {
                tokio::select! {
                    output = process.output() => output,
                    _ = cancel.cancelled() => {
                        return ToolOutput {
                            content: "Bash 已取消".into(),
                            is_error: true,
                        };
                    }
                }
            }
            None => process.output().await,
        };
        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let mut content = String::new();
                if !stdout.is_empty() {
                    content.push_str(&stdout);
                }
                if !stderr.is_empty() {
                    if !content.is_empty() && !content.ends_with('\n') {
                        content.push('\n');
                    }
                    content.push_str("[stderr]\n");
                    content.push_str(&stderr);
                }
                if content.trim().is_empty() {
                    content = format!("命令完成，退出码：{}", output.status);
                }
                ToolOutput {
                    content,
                    is_error: !output.status.success(),
                }
            }
            Err(error) => error_output(error.into()),
        }
    }
}

#[async_trait::async_trait]
impl AgentTool for BashTool {
    fn name(&self) -> &str {
        "Bash"
    }

    fn schema(&self) -> Value {
        function_schema(
            self.name(),
            "Run a shell command. High risk: can modify the system; always requires approval by default.",
            json!({
                "type":"object",
                "properties":{
                    "command":{"type":"string"},
                    "cwd":{"type":"string","description":"Workspace directory, default root"}
                },
                "required":["command"]
            }),
        )
    }

    fn execution_timeout(&self) -> Option<Duration> {
        Some(Duration::from_secs(120))
    }

    async fn execute(&self, raw_args: &str) -> ToolOutput {
        self.run(raw_args, None).await
    }

    async fn execute_with_context(
        &self,
        raw_args: &str,
        context: ToolExecutionContext,
    ) -> ToolOutput {
        self.run(raw_args, Some(&context.cancel)).await
    }
}

fn function_schema(name: &str, description: &str, parameters: Value) -> Value {
    json!({
        "type":"function",
        "function":{
            "name":name,
            "description":description,
            "parameters":parameters
        }
    })
}

fn run_sync(
    raw_args: &str,
    operation: impl FnOnce(&Map<String, Value>) -> Result<String>,
) -> ToolOutput {
    match parse_arguments(raw_args).and_then(|arguments| operation(&arguments)) {
        Ok(content) => ToolOutput {
            content,
            is_error: false,
        },
        Err(error) => error_output(error),
    }
}

fn parse_arguments(raw_args: &str) -> Result<Map<String, Value>> {
    match serde_json::from_str(raw_args)? {
        Value::Object(arguments) => Ok(arguments),
        _ => bail!("工具参数必须是 JSON Object"),
    }
}

fn required_string<'a>(arguments: &'a Map<String, Value>, name: &str) -> Result<&'a str> {
    arguments
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .with_context(|| format!("缺少字符串参数 {name}"))
}

fn required_value_string<'a>(arguments: &'a Map<String, Value>, name: &str) -> Result<&'a str> {
    arguments
        .get(name)
        .and_then(Value::as_str)
        .with_context(|| format!("缺少字符串参数 {name}"))
}

fn optional_string<'a>(arguments: &'a Map<String, Value>, name: &str) -> Option<&'a str> {
    arguments.get(name).and_then(Value::as_str)
}

fn optional_usize(arguments: &Map<String, Value>, name: &str) -> Result<Option<usize>> {
    arguments
        .get(name)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .with_context(|| format!("{name} 必须是非负整数"))
        })
        .transpose()
}

fn ensure_file_size(path: &Path) -> Result<()> {
    let size = fs::metadata(path)?.len();
    anyhow::ensure!(
        size <= MAX_FILE_BYTES,
        "文件为 {size} 字节，超过 {} 字节读取上限",
        MAX_FILE_BYTES
    );
    Ok(())
}

fn no_results_or_join(results: Vec<String>) -> String {
    if results.is_empty() {
        "(无匹配结果)".into()
    } else {
        results.join("\n")
    }
}

fn error_output(error: anyhow::Error) -> ToolOutput {
    ToolOutput {
        content: format!("工具执行失败: {error:#}"),
        is_error: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(1);

    fn fixture() -> (PathBuf, Workspace) {
        let id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("asteria-core-tools-{}-{id}", std::process::id()));
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/lib.rs"), "fn one() {}\nfn two() {}\n").unwrap();
        let workspace = Workspace {
            root: Arc::new(fs::canonicalize(&root).unwrap()),
        };
        (root, workspace)
    }

    #[tokio::test]
    async fn read_write_and_edit_are_exact() {
        let (root, workspace) = fixture();
        let read = ReadTool(workspace.clone())
            .execute(r#"{"path":"src/lib.rs","offset":2,"limit":1}"#)
            .await;
        assert_eq!(read.content, "2|fn two() {}");

        let write = WriteTool(workspace.clone())
            .execute(r#"{"path":"src/new.txt","content":"old"}"#)
            .await;
        assert!(!write.is_error);
        let edit = EditTool(workspace)
            .execute(
                r#"{"path":"src/new.txt","old_string":"old","new_string":"new","replace_all":false}"#,
            )
            .await;
        assert!(!edit.is_error);
        assert_eq!(fs::read_to_string(root.join("src/new.txt")).unwrap(), "new");
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn glob_and_grep_find_workspace_files() {
        let (root, workspace) = fixture();
        let glob = GlobTool(workspace.clone())
            .execute(r#"{"pattern":"**/*.rs"}"#)
            .await;
        assert_eq!(glob.content, "src/lib.rs");
        let grep = GrepTool(workspace)
            .execute(r#"{"pattern":"two","glob":"**/*.rs"}"#)
            .await;
        assert_eq!(grep.content, "src/lib.rs:2:fn two() {}");
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn file_tools_reject_workspace_escape() {
        let (root, workspace) = fixture();
        let outside = root.parent().unwrap().display().to_string();
        let output = ReadTool(workspace)
            .execute(&json!({"path":outside}).to_string())
            .await;
        assert!(output.is_error);
        assert!(output.content.contains("工作区之外"));
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn bash_runs_in_workspace() {
        let (root, workspace) = fixture();
        let output = BashTool(workspace).execute(r#"{"command":"pwd"}"#).await;
        assert!(!output.is_error);
        assert_eq!(
            output.content.trim(),
            fs::canonicalize(&root).unwrap().display().to_string()
        );
        let _ = fs::remove_dir_all(root);
    }
}
