use super::{
    config::{LoadedMcpConfig, McpServerDefinition},
    oauth::{FileCredentialStore, authorize_in_browser},
    tool::{McpToolAdapter, namespaced_tool_name},
};
use anyhow::{Context, Result};
use rmcp::{
    RmcpError, RoleClient, ServiceExt,
    service::RunningService,
    transport::{
        StreamableHttpClientTransport, TokioChildProcess,
        auth::{AuthClient, AuthorizationManager},
        streamable_http_client::{
            AuthRequiredError, InsufficientScopeError, StreamableHttpClientTransportConfig,
        },
    },
};
use std::{
    collections::{BTreeMap, HashSet},
    path::PathBuf,
    time::Duration,
};
use tokio::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McpServerState {
    Disabled,
    Connected,
    Disconnected,
    AuthRequired,
    Failed,
}

impl std::fmt::Display for McpServerState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Disabled => "disabled",
            Self::Connected => "connected",
            Self::Disconnected => "disconnected",
            Self::AuthRequired => "auth_required",
            Self::Failed => "failed",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OAuthFlow {
    StoredOnly,
    InteractiveIfRequired,
}

#[derive(Clone, Debug)]
pub struct McpServerStatus {
    pub name: String,
    pub state: McpServerState,
    pub tools: Vec<String>,
    pub detail: Option<String>,
}

/// 持有所有 stdio/Streamable HTTP Client Service，确保连接贯穿 Asteria 生命周期。
pub struct McpManager {
    connections: BTreeMap<String, RunningService<RoleClient, ()>>,
    statuses: Vec<McpServerStatus>,
    loaded_files: Vec<PathBuf>,
    skipped_project_file: Option<PathBuf>,
    global_config: Option<PathBuf>,
    project_config: Option<PathBuf>,
}

pub struct McpLoadResult {
    pub manager: McpManager,
    pub tools: Vec<McpToolAdapter>,
}

impl McpManager {
    /// 连接全部启用的 Server；单个 Server 失败不会阻止其他 Server。
    pub async fn connect(config: LoadedMcpConfig) -> McpLoadResult {
        Self::connect_with_oauth(config, OAuthFlow::StoredOnly).await
    }

    pub(crate) async fn connect_with_oauth(
        config: LoadedMcpConfig,
        oauth: OAuthFlow,
    ) -> McpLoadResult {
        let mut connections = BTreeMap::new();
        let mut statuses = Vec::new();
        let mut adapters = Vec::new();
        let mut exposed_names = HashSet::new();

        for definition in config.servers {
            if !definition.config.enabled {
                statuses.push(status(
                    &definition,
                    McpServerState::Disabled,
                    Vec::new(),
                    None,
                ));
                continue;
            }
            let connected = if definition.config.url.is_some() {
                connect_http_server(&definition, oauth).await
            } else {
                connect_stdio_server(&definition).await
            };
            match connected {
                Ok((service, discovered)) => {
                    let peer = service.peer().clone();
                    let timeout = Duration::from_millis(definition.config.tool_timeout_ms);
                    let mut tool_names = Vec::new();
                    let mut collisions = Vec::new();
                    for tool in discovered {
                        if !definition.config.tool_enabled(tool.name.as_ref()) {
                            continue;
                        }
                        let exposed = namespaced_tool_name(&definition.name, tool.name.as_ref());
                        if !exposed_names.insert(exposed.clone()) {
                            collisions.push(exposed);
                            continue;
                        }
                        tool_names.push(exposed);
                        adapters.push(McpToolAdapter::new(
                            &definition.name,
                            tool,
                            peer.clone(),
                            timeout,
                        ));
                    }
                    let detail = (!collisions.is_empty())
                        .then(|| format!("跳过命名冲突工具: {}", collisions.join(", ")));
                    statuses.push(status(
                        &definition,
                        McpServerState::Connected,
                        tool_names,
                        detail,
                    ));
                    connections.insert(definition.name.clone(), service);
                }
                Err(error) => {
                    let state = connection_error_state(&error);
                    statuses.push(status(
                        &definition,
                        state,
                        Vec::new(),
                        Some(format!("{error:#}")),
                    ));
                }
            }
        }

        McpLoadResult {
            manager: McpManager {
                connections,
                statuses,
                loaded_files: config.loaded_files,
                skipped_project_file: config.skipped_project_file,
                global_config: config.global_config,
                project_config: config.project_config,
            },
            tools: adapters,
        }
    }

    pub fn statuses(&self) -> &[McpServerStatus] {
        &self.statuses
    }

    pub fn loaded_files(&self) -> &[PathBuf] {
        &self.loaded_files
    }

    pub fn skipped_project_file(&self) -> Option<&std::path::Path> {
        self.skipped_project_file.as_deref()
    }

    pub fn global_config(&self) -> Option<&std::path::Path> {
        self.global_config.as_deref()
    }

    pub fn project_config(&self) -> Option<&std::path::Path> {
        self.project_config.as_deref()
    }

    pub fn active_connections(&self) -> usize {
        self.connections.len()
    }

    pub fn has_blocking_failures(&self) -> bool {
        self.statuses
            .iter()
            .any(|status| status.state == McpServerState::Failed)
    }

    pub fn failure_summary(&self) -> String {
        self.statuses
            .iter()
            .filter(|status| {
                matches!(
                    status.state,
                    McpServerState::AuthRequired | McpServerState::Failed
                )
            })
            .map(|status| {
                format!(
                    "{}: {}",
                    status.name,
                    status.detail.as_deref().unwrap_or("连接失败")
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    pub fn server_tools(&self, server_name: &str) -> Option<&[String]> {
        self.statuses
            .iter()
            .find(|status| status.name == server_name)
            .map(|status| status.tools.as_slice())
    }

    /// 手动关闭一个 Server，并保留 disconnected 状态供 TUI 展示。
    pub async fn disconnect(&mut self, server_name: &str) -> Result<Vec<String>> {
        let status = self
            .statuses
            .iter_mut()
            .find(|status| status.name == server_name)
            .with_context(|| format!("未知 MCP Server: {server_name}"))?;
        let tools = std::mem::take(&mut status.tools);
        if let Some(service) = self.connections.remove(server_name) {
            service
                .cancel()
                .await
                .with_context(|| format!("关闭 MCP Server `{server_name}` 失败"))?;
        }
        status.state = McpServerState::Disconnected;
        status.detail = Some("已由用户手动断开".into());
        Ok(tools)
    }

    /// 候选 Server 已连接成功后，用它替换当前同名连接。
    pub async fn replace_server(
        &mut self,
        server_name: &str,
        mut candidate: McpManager,
    ) -> Result<()> {
        let new_status = candidate
            .statuses
            .iter()
            .find(|status| status.name == server_name)
            .cloned()
            .with_context(|| format!("候选配置中没有 MCP Server: {server_name}"))?;
        anyhow::ensure!(
            new_status.state == McpServerState::Connected,
            "MCP Server `{server_name}` 未连接成功（{}{}）",
            new_status.state,
            new_status
                .detail
                .as_ref()
                .map(|detail| format!(": {detail}"))
                .unwrap_or_default()
        );
        let new_connection = candidate
            .connections
            .remove(server_name)
            .context("候选 MCP 连接缺失")?;
        if let Some(old) = self.connections.remove(server_name) {
            let _ = old.cancel().await;
        }
        self.connections
            .insert(server_name.to_owned(), new_connection);
        self.statuses.retain(|status| status.name != server_name);
        self.statuses.push(new_status);
        self.statuses
            .sort_by(|left, right| left.name.cmp(&right.name));
        self.loaded_files = candidate.loaded_files;
        self.skipped_project_file = candidate.skipped_project_file;
        self.global_config = candidate.global_config;
        self.project_config = candidate.project_config;
        Ok(())
    }

    pub async fn shutdown(mut self) {
        let connections = std::mem::take(&mut self.connections);
        for (_, service) in connections {
            let _ = service.cancel().await;
        }
    }
}

async fn connect_http_server(
    definition: &McpServerDefinition,
    oauth: OAuthFlow,
) -> Result<(RunningService<RoleClient, ()>, Vec<rmcp::model::Tool>)> {
    let url = definition
        .config
        .url
        .as_deref()
        .context("Streamable HTTP MCP Server 缺少 url")?;
    let (bearer_token, custom_headers) = definition.config.resolved_http_headers()?;
    let mut config =
        StreamableHttpClientTransportConfig::with_uri(url).custom_headers(custom_headers);
    if let Some(token) = bearer_token {
        config = config.auth_header(token);
        return handshake_http(
            definition,
            StreamableHttpClientTransport::from_config(config),
        )
        .await;
    }

    match connect_http_with_stored_oauth(definition, config.clone()).await {
        Ok(connected) => Ok(connected),
        Err(error) if oauth == OAuthFlow::InteractiveIfRequired && is_auth_required(&error) => {
            let manager = authorize_in_browser(
                url,
                &definition.name,
                definition.config.oauth_client_id.as_deref(),
                auth_challenge_from_error(&error).as_deref(),
                |message| eprintln!("{message}"),
            )
            .await?;
            handshake_authorized(definition, config, manager).await
        }
        Err(error) => Err(annotate_auth_required(definition, error)),
    }
}

async fn connect_http_with_stored_oauth(
    definition: &McpServerDefinition,
    config: StreamableHttpClientTransportConfig,
) -> Result<(RunningService<RoleClient, ()>, Vec<rmcp::model::Tool>)> {
    let url = definition
        .config
        .url
        .as_deref()
        .context("Streamable HTTP MCP Server 缺少 url")?;
    let mut manager = AuthorizationManager::new(url)
        .await
        .with_context(|| format!("无法初始化 MCP Server `{}` 的 OAuth", definition.name))?;
    if let Some(store) = FileCredentialStore::for_server(&definition.name) {
        manager.set_credential_store(store);
        let _ = manager.initialize_from_store().await;
    }
    handshake_authorized(definition, config, manager).await
}

async fn handshake_authorized(
    definition: &McpServerDefinition,
    config: StreamableHttpClientTransportConfig,
    manager: AuthorizationManager,
) -> Result<(RunningService<RoleClient, ()>, Vec<rmcp::model::Tool>)> {
    let client = AuthClient::new(mcp_http_client()?, manager);
    handshake_http(
        definition,
        StreamableHttpClientTransport::with_client(client, config),
    )
    .await
}

async fn handshake_http<T, E, A>(
    definition: &McpServerDefinition,
    transport: T,
) -> Result<(RunningService<RoleClient, ()>, Vec<rmcp::model::Tool>)>
where
    T: rmcp::transport::IntoTransport<RoleClient, E, A>,
    E: std::error::Error + Send + Sync + 'static,
{
    let startup_timeout = Duration::from_millis(definition.config.startup_timeout_ms);
    let service = tokio::time::timeout(startup_timeout, ().serve(transport))
        .await
        .with_context(|| format!("MCP Server `{}` HTTP 初始化超时", definition.name))?
        .with_context(|| format!("MCP Server `{}` HTTP 初始化失败", definition.name))?;
    let tools = tokio::time::timeout(startup_timeout, service.list_all_tools())
        .await
        .with_context(|| format!("MCP Server `{}` tools/list 超时", definition.name))?
        .with_context(|| format!("MCP Server `{}` tools/list 失败", definition.name))?;
    Ok((service, tools))
}

fn mcp_http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .pool_max_idle_per_host(0)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("无法创建 MCP HTTP 客户端")
}

fn annotate_auth_required(definition: &McpServerDefinition, error: anyhow::Error) -> anyhow::Error {
    if is_auth_required(&error) {
        error.context(format!(
            "MCP Server `{}` 需要浏览器登录，输入 /mcp auth {}",
            definition.name, definition.name
        ))
    } else {
        error
    }
}

fn is_auth_required(error: &anyhow::Error) -> bool {
    auth_challenge_from_error(error).is_some()
        || error.chain().any(|cause| {
            cause.downcast_ref::<AuthRequiredError>().is_some()
                || cause.downcast_ref::<InsufficientScopeError>().is_some()
        })
}

fn auth_challenge_from_error(error: &anyhow::Error) -> Option<String> {
    for cause in error.chain() {
        if let Some(required) = cause.downcast_ref::<AuthRequiredError>() {
            return Some(required.www_authenticate_header.clone());
        }
        if let Some(scope) = cause.downcast_ref::<InsufficientScopeError>() {
            return Some(scope.www_authenticate_header.clone());
        }
        if let Some(RmcpError::ClientInitialize(initialize)) = cause.downcast_ref::<RmcpError>()
            && let Some(challenge) = initialize.auth_challenge()
        {
            return Some(challenge.to_owned());
        }
    }
    None
}

async fn connect_stdio_server(
    definition: &McpServerDefinition,
) -> Result<(RunningService<RoleClient, ()>, Vec<rmcp::model::Tool>)> {
    let command_name = definition
        .config
        .command
        .as_deref()
        .context("stdio MCP Server 缺少 command")?;
    let workspace_root = std::env::current_dir().context("无法确定 MCP 工作区目录")?;
    let mut command = Command::new(command_name);
    command.args(definition.config.resolved_args(&workspace_root));
    command.envs(definition.config.resolved_env()?);
    if let Some(cwd) = definition
        .config
        .resolved_cwd(&definition.config_path, &workspace_root)
    {
        command.current_dir(cwd);
    }
    let transport = TokioChildProcess::new(command)
        .with_context(|| format!("无法启动 MCP Server `{}`", definition.name))?;
    let startup_timeout = Duration::from_millis(definition.config.startup_timeout_ms);
    let service = tokio::time::timeout(startup_timeout, ().serve(transport))
        .await
        .with_context(|| format!("MCP Server `{}` 初始化超时", definition.name))?
        .with_context(|| format!("MCP Server `{}` 初始化失败", definition.name))?;
    let tools = tokio::time::timeout(startup_timeout, service.list_all_tools())
        .await
        .with_context(|| format!("MCP Server `{}` tools/list 超时", definition.name))?
        .with_context(|| format!("MCP Server `{}` tools/list 失败", definition.name))?;
    Ok((service, tools))
}

fn connection_error_state(error: &anyhow::Error) -> McpServerState {
    if error.chain().any(|cause| {
        cause.downcast_ref::<AuthRequiredError>().is_some()
            || cause.downcast_ref::<InsufficientScopeError>().is_some()
    }) {
        McpServerState::AuthRequired
    } else {
        McpServerState::Failed
    }
}

fn status(
    definition: &McpServerDefinition,
    state: McpServerState,
    tools: Vec<String>,
    detail: Option<String>,
) -> McpServerStatus {
    McpServerStatus {
        name: definition.name.clone(),
        state,
        tools,
        detail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::config::McpServerConfig;
    use crate::tools::AgentTool;

    fn definition(name: &str, json: &str) -> McpServerDefinition {
        McpServerDefinition {
            name: name.into(),
            config_path: PathBuf::from("mcp.json"),
            config: serde_json::from_str::<McpServerConfig>(json).unwrap(),
        }
    }

    #[tokio::test]
    async fn disabled_servers_do_not_start() {
        let config = LoadedMcpConfig {
            servers: vec![definition(
                "off",
                r#"{"command":"missing","enabled":false}"#,
            )],
            ..Default::default()
        };
        let loaded = McpManager::connect(config).await;
        assert!(loaded.tools.is_empty());
        assert_eq!(loaded.manager.active_connections(), 0);
        assert_eq!(loaded.manager.statuses()[0].state, McpServerState::Disabled);
    }

    #[test]
    fn classifies_http_authentication_errors() {
        let error = anyhow::Error::new(AuthRequiredError::new(
            "Bearer resource_metadata=https://example.com".into(),
        ));
        assert_eq!(connection_error_state(&error), McpServerState::AuthRequired);
        assert_eq!(
            auth_challenge_from_error(&error).as_deref(),
            Some("Bearer resource_metadata=https://example.com")
        );
        assert!(is_auth_required(&error));
    }

    #[tokio::test]
    async fn connects_discovers_and_calls_stdio_tool() {
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mcp_echo_server.py");
        let definition = McpServerDefinition {
            name: "fixture".into(),
            config_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("mcp.json"),
            config: serde_json::from_value(serde_json::json!({
                "command": "python3",
                "args": [fixture],
                "startupTimeoutMs": 5_000,
                "toolTimeoutMs": 2_000
            }))
            .unwrap(),
        };
        let loaded = McpManager::connect(LoadedMcpConfig {
            servers: vec![definition],
            ..Default::default()
        })
        .await;

        assert_eq!(loaded.manager.active_connections(), 1);
        assert_eq!(
            loaded.manager.statuses()[0].state,
            McpServerState::Connected
        );
        assert_eq!(loaded.tools.len(), 1);
        assert_eq!(loaded.tools[0].name(), "mcp__fixture__echo");
        let output = loaded.tools[0].execute(r#"{"text":"hello"}"#).await;
        assert!(!output.is_error);
        assert_eq!(output.content, "echo:hello");

        let mut manager = loaded.manager;
        let removed = manager.disconnect("fixture").await.unwrap();
        assert_eq!(removed, vec!["mcp__fixture__echo"]);
        assert_eq!(manager.active_connections(), 0);
        assert_eq!(manager.statuses()[0].state, McpServerState::Disconnected);
        assert!(manager.statuses()[0].tools.is_empty());
    }

    #[tokio::test]
    #[ignore = "需要访问真实第三方 Cloudflare MCP"]
    async fn connects_discovers_and_calls_cloudflare_docs_tool() {
        let remote = definition(
            "cloudflare-docs",
            r#"{
                "url":"https://docs.mcp.cloudflare.com/mcp",
                "startupTimeoutMs":30000,
                "toolTimeoutMs":30000
            }"#,
        );
        let loaded = McpManager::connect(LoadedMcpConfig {
            servers: vec![remote],
            ..Default::default()
        })
        .await;

        assert_eq!(loaded.manager.active_connections(), 1);
        let tool = loaded
            .tools
            .iter()
            .find(|tool| tool.name().ends_with("search_cloudflare_documentation"))
            .expect("Cloudflare 文档搜索工具未发现");
        let output = tool
            .execute(r#"{"query":"Cloudflare Workers KV overview"}"#)
            .await;
        assert!(!output.is_error, "{}", output.content);
        assert!(!output.content.trim().is_empty());
    }

    async fn call_live_remote_tool(
        server_name: &str,
        url: &str,
        tool_suffix: &str,
        arguments: &str,
    ) {
        let remote = definition(
            server_name,
            &format!(
                r#"{{
                    "url":"{url}",
                    "startupTimeoutMs":30000,
                    "toolTimeoutMs":30000
                }}"#
            ),
        );
        let loaded = McpManager::connect(LoadedMcpConfig {
            servers: vec![remote],
            ..Default::default()
        })
        .await;
        assert_eq!(loaded.manager.active_connections(), 1);
        let tool = loaded
            .tools
            .iter()
            .find(|tool| tool.name().ends_with(tool_suffix))
            .unwrap_or_else(|| panic!("{server_name} 未发现工具 {tool_suffix}"));
        let output = tool.execute(arguments).await;
        assert!(!output.is_error, "{}", output.content);
        assert!(!output.content.trim().is_empty());
    }

    #[tokio::test]
    #[ignore = "需要访问真实第三方 Context7 MCP"]
    async fn calls_context7_library_resolution_tool() {
        call_live_remote_tool(
            "context7",
            "https://mcp.context7.com/mcp",
            "resolve-library-id",
            r#"{"libraryName":"tokio","query":"Rust async runtime"}"#,
        )
        .await;
    }

    #[tokio::test]
    #[ignore = "需要访问真实第三方 DeepWiki MCP"]
    async fn calls_deepwiki_repository_question_tool() {
        call_live_remote_tool(
            "deepwiki",
            "https://mcp.deepwiki.com/mcp",
            "ask_question",
            r#"{"repoName":"tokio-rs/tokio","question":"What is the Tokio runtime?"}"#,
        )
        .await;
    }

    #[tokio::test]
    #[ignore = "需要访问真实第三方 Microsoft Learn MCP"]
    async fn calls_microsoft_learn_search_tool() {
        call_live_remote_tool(
            "microsoft-learn",
            "https://learn.microsoft.com/api/mcp",
            "microsoft_docs_search",
            r#"{"query":"Azure Functions HTTP trigger"}"#,
        )
        .await;
    }
}
