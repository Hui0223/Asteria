use anyhow::{Context, Result, bail};
use http::{HeaderName, HeaderValue, Uri};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
};

const DEFAULT_STARTUP_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_TOOL_TIMEOUT_MS: u64 = 60_000;

/// 单个 MCP Server 的声明；第一阶段仅执行 command 类型的 stdio Server。
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerConfig {
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    pub cwd: Option<PathBuf>,
    pub url: Option<String>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_ms: u64,
    #[serde(default = "default_tool_timeout")]
    pub tool_timeout_ms: u64,
    pub enabled_tools: Option<BTreeSet<String>>,
    #[serde(default)]
    pub disabled_tools: BTreeSet<String>,
}

impl McpServerConfig {
    pub fn validate_transport(&self) -> Result<()> {
        match (self.command.as_deref(), self.url.as_deref()) {
            (Some(_), None) | (None, Some(_)) => {}
            (Some(_), Some(_)) => bail!("MCP Server 不能同时配置 command 和 url"),
            (None, None) => bail!("MCP Server 必须配置 command 或 url"),
        }
        if let Some(url) = &self.url {
            let uri = url
                .parse::<Uri>()
                .with_context(|| format!("无效的 MCP URL: {url}"))?;
            let scheme = uri.scheme_str().context("MCP URL 缺少协议")?;
            let host = uri.host().context("MCP URL 缺少主机")?;
            let local = matches!(host, "localhost" | "127.0.0.1" | "::1");
            if scheme != "https" && !(scheme == "http" && local) {
                bail!("远程 MCP 必须使用 HTTPS；HTTP 只允许 localhost");
            }
        }
        Ok(())
    }

    pub fn tool_enabled(&self, name: &str) -> bool {
        self.enabled_tools
            .as_ref()
            .is_none_or(|enabled| enabled.contains(name))
            && !self.disabled_tools.contains(name)
    }

    pub fn resolved_args(&self, workspace_root: &Path) -> Vec<String> {
        self.args
            .iter()
            .map(|argument| expand_workspace_folder(argument, workspace_root))
            .collect()
    }

    pub fn resolved_cwd(&self, config_path: &Path, workspace_root: &Path) -> Option<PathBuf> {
        self.cwd.as_ref().map(|cwd| {
            let expanded = PathBuf::from(expand_workspace_folder(
                &cwd.to_string_lossy(),
                workspace_root,
            ));
            if expanded.is_absolute() {
                expanded
            } else {
                config_path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(expanded)
            }
        })
    }

    pub fn resolved_env(&self) -> Result<Vec<(String, String)>> {
        self.env
            .iter()
            .map(|(name, value)| {
                let resolved = environment_reference(value)
                    .map(std::env::var)
                    .transpose()
                    .with_context(|| format!("MCP 环境变量 {name} 引用不存在"))?
                    .unwrap_or_else(|| value.clone());
                Ok((name.clone(), resolved))
            })
            .collect()
    }

    pub fn resolved_http_headers(
        &self,
    ) -> Result<(Option<String>, HashMap<HeaderName, HeaderValue>)> {
        let mut bearer_token = None;
        let mut headers = HashMap::new();
        for (raw_name, raw_value) in &self.headers {
            let name = HeaderName::from_bytes(raw_name.as_bytes())
                .with_context(|| format!("无效的 MCP HTTP Header 名称: {raw_name}"))?;
            let value = expand_environment_references(raw_value)
                .with_context(|| format!("无法解析 MCP HTTP Header `{raw_name}`"))?;
            if name == http::header::AUTHORIZATION {
                if !raw_value.contains("${") {
                    bail!("Authorization 必须通过环境变量引用，不能明文写入 mcp.json");
                }
                let (scheme, token) = value
                    .split_once(' ')
                    .context("Authorization 必须使用 Bearer Token")?;
                if !scheme.eq_ignore_ascii_case("bearer") || token.trim().is_empty() {
                    bail!("Authorization 必须使用 Bearer Token");
                }
                bearer_token = Some(token.trim().to_owned());
                continue;
            }
            let value = HeaderValue::from_str(&value)
                .with_context(|| format!("无效的 MCP HTTP Header 值: {raw_name}"))?;
            headers.insert(name, value);
        }
        Ok((bearer_token, headers))
    }
}

#[derive(Clone, Debug)]
pub struct McpServerDefinition {
    pub name: String,
    pub config_path: PathBuf,
    pub config: McpServerConfig,
}

#[derive(Default)]
pub struct LoadedMcpConfig {
    pub servers: Vec<McpServerDefinition>,
    pub loaded_files: Vec<PathBuf>,
    pub skipped_project_file: Option<PathBuf>,
}

#[derive(Deserialize)]
struct McpConfigFile {
    #[serde(default, rename = "mcpServers")]
    servers: BTreeMap<String, McpServerConfig>,
}

/// 加载用户级配置，并在显式信任时让项目级配置覆盖同名 Server。
pub fn load_default(trust_project: bool) -> Result<LoadedMcpConfig> {
    let global = std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".asteria/mcp.json"));
    let project = std::env::current_dir()?.join(".asteria/mcp.json");
    load_from_paths(global.as_deref(), Some(&project), trust_project)
}

fn load_from_paths(
    global: Option<&Path>,
    project: Option<&Path>,
    trust_project: bool,
) -> Result<LoadedMcpConfig> {
    let mut merged = BTreeMap::<String, McpServerDefinition>::new();
    let mut loaded_files = Vec::new();
    if let Some(path) = global
        && path.is_file()
    {
        merge_file(path, &mut merged)?;
        loaded_files.push(path.to_owned());
    }

    let mut skipped_project_file = None;
    if let Some(path) = project
        && path.is_file()
        && global != Some(path)
    {
        if trust_project {
            merge_file(path, &mut merged)?;
            loaded_files.push(path.to_owned());
        } else {
            skipped_project_file = Some(path.to_owned());
        }
    }

    Ok(LoadedMcpConfig {
        servers: merged.into_values().collect(),
        loaded_files,
        skipped_project_file,
    })
}

fn merge_file(path: &Path, merged: &mut BTreeMap<String, McpServerDefinition>) -> Result<()> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("无法读取 MCP 配置 {}", path.display()))?;
    let config: McpConfigFile = serde_json::from_str(&content)
        .with_context(|| format!("无法解析 MCP 配置 {}", path.display()))?;
    for (name, config) in config.servers {
        if name.trim().is_empty() {
            bail!("{} 包含空 MCP Server 名称", path.display());
        }
        config
            .validate_transport()
            .with_context(|| format!("{} 中的 MCP Server `{name}` 配置无效", path.display()))?;
        merged.insert(
            name.clone(),
            McpServerDefinition {
                name,
                config_path: path.to_owned(),
                config,
            },
        );
    }
    Ok(())
}

fn environment_reference(value: &str) -> Option<&str> {
    value
        .strip_prefix("${")
        .and_then(|value| value.strip_suffix('}'))
        .filter(|name| !name.is_empty())
}

fn expand_workspace_folder(value: &str, workspace_root: &Path) -> String {
    value.replace(
        "${workspaceFolder}",
        workspace_root.to_string_lossy().as_ref(),
    )
}

fn expand_environment_references(value: &str) -> Result<String> {
    let mut output = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(start) = rest.find("${") {
        output.push_str(&rest[..start]);
        let reference = &rest[start + 2..];
        let end = reference.find('}').context("环境变量引用缺少右花括号")?;
        let name = &reference[..end];
        if name.is_empty() {
            bail!("环境变量名称不能为空");
        }
        output.push_str(
            &std::env::var(name).with_context(|| format!("MCP 环境变量 {name} 引用不存在"))?,
        );
        rest = &reference[end + 1..];
    }
    output.push_str(rest);
    Ok(output)
}

const fn default_true() -> bool {
    true
}

const fn default_startup_timeout() -> u64 {
    DEFAULT_STARTUP_TIMEOUT_MS
}

const fn default_tool_timeout() -> u64 {
    DEFAULT_TOOL_TIMEOUT_MS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("asteria-mcp-config-{id}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn project_config_requires_explicit_trust_and_overrides_global() {
        let dir = temp_dir();
        let global = dir.join("global.json");
        let project = dir.join("project.json");
        fs::write(
            &global,
            r#"{"mcpServers":{"same":{"command":"global"},"global":{"command":"g"}}}"#,
        )
        .unwrap();
        fs::write(
            &project,
            r#"{"mcpServers":{"same":{"command":"project"},"local":{"command":"l"}}}"#,
        )
        .unwrap();

        let untrusted = load_from_paths(Some(&global), Some(&project), false).unwrap();
        assert_eq!(untrusted.servers.len(), 2);
        assert_eq!(untrusted.skipped_project_file, Some(project.clone()));

        let trusted = load_from_paths(Some(&global), Some(&project), true).unwrap();
        assert_eq!(trusted.servers.len(), 3);
        let same = trusted
            .servers
            .iter()
            .find(|server| server.name == "same")
            .unwrap();
        assert_eq!(same.config.command.as_deref(), Some("project"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn filters_tools_with_allowlist_and_blocklist() {
        let config: McpServerConfig = serde_json::from_str(
            r#"{"command":"x","enabledTools":["read","write"],"disabledTools":["write"]}"#,
        )
        .unwrap();
        assert!(config.tool_enabled("read"));
        assert!(!config.tool_enabled("write"));
        assert!(!config.tool_enabled("other"));
        assert_eq!(config.startup_timeout_ms, 30_000);
        assert_eq!(config.tool_timeout_ms, 60_000);
    }

    #[test]
    fn validates_remote_transport_and_resolves_safe_headers() {
        let config: McpServerConfig = serde_json::from_str(
            r#"{
                "url":"https://example.com/mcp",
                "headers":{"X-MCP-Readonly":"true"}
            }"#,
        )
        .unwrap();
        config.validate_transport().unwrap();
        let (token, headers) = config.resolved_http_headers().unwrap();
        assert!(token.is_none());
        assert_eq!(
            headers
                .get(&HeaderName::from_static("x-mcp-readonly"))
                .unwrap(),
            "true"
        );

        let insecure: McpServerConfig =
            serde_json::from_str(r#"{"url":"http://example.com/mcp"}"#).unwrap();
        assert!(insecure.validate_transport().is_err());
        let local: McpServerConfig =
            serde_json::from_str(r#"{"url":"http://localhost:8080/mcp"}"#).unwrap();
        local.validate_transport().unwrap();
    }

    #[test]
    fn rejects_ambiguous_transport_and_plaintext_authorization() {
        let ambiguous: McpServerConfig =
            serde_json::from_str(r#"{"command":"server","url":"https://example.com/mcp"}"#)
                .unwrap();
        assert!(ambiguous.validate_transport().is_err());

        let plaintext: McpServerConfig = serde_json::from_str(
            r#"{
                "url":"https://example.com/mcp",
                "headers":{"Authorization":"Bearer secret"}
            }"#,
        )
        .unwrap();
        assert!(plaintext.resolved_http_headers().is_err());
    }

    #[test]
    fn expands_workspace_folder_in_args_and_cwd() {
        let config: McpServerConfig = serde_json::from_str(
            r#"{
                "command":"server",
                "args":["--root","${workspaceFolder}","${workspaceFolder}/docs"],
                "cwd":"${workspaceFolder}/tools"
            }"#,
        )
        .unwrap();
        let root = Path::new("/tmp/asteria-workspace");
        assert_eq!(
            config.resolved_args(root),
            vec![
                "--root",
                "/tmp/asteria-workspace",
                "/tmp/asteria-workspace/docs"
            ]
        );
        assert_eq!(
            config.resolved_cwd(Path::new("/tmp/config/mcp.json"), root),
            Some(PathBuf::from("/tmp/asteria-workspace/tools"))
        );
    }
}
