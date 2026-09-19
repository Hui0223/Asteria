use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use rmcp::transport::auth::{
    AuthError, AuthorizationManager, AuthorizationRequest, AuthorizationSession, CredentialStore,
    StoredCredentials,
};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

const CALLBACK_TIMEOUT: Duration = Duration::from_secs(300);

/// Per-server OAuth token store under `~/.asteria/oauth/<server>.json`.
#[derive(Clone, Debug)]
pub struct FileCredentialStore {
    path: PathBuf,
}

impl FileCredentialStore {
    pub fn for_server(server_name: &str) -> Option<Self> {
        Some(Self {
            path: oauth_store_path(server_name)?,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[async_trait]
impl CredentialStore for FileCredentialStore {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        if !self.path.is_file() {
            return Ok(None);
        }
        let json = std::fs::read_to_string(&self.path).map_err(store_io_error)?;
        serde_json::from_str(&json).map_err(|error| {
            AuthError::CredentialStoreError(format!(
                "无法解析 OAuth 凭据 {}: {error}",
                self.path.display()
            ))
        })
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        let parent = self.path.parent().ok_or_else(|| {
            AuthError::CredentialStoreError(format!("OAuth 凭据路径无效: {}", self.path.display()))
        })?;
        std::fs::create_dir_all(parent).map_err(store_io_error)?;
        let json = serde_json::to_string_pretty(&credentials).map_err(|error| {
            AuthError::CredentialStoreError(format!("无法序列化 OAuth 凭据: {error}"))
        })?;
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, json).map_err(store_io_error)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))
                .map_err(store_io_error)?;
        }
        std::fs::rename(&tmp, &self.path).map_err(store_io_error)
    }

    async fn clear(&self) -> Result<(), AuthError> {
        match std::fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(store_io_error(error)),
        }
    }
}

pub fn oauth_store_path(server_name: &str) -> Option<PathBuf> {
    Some(oauth_dir()?.join(format!("{}.json", sanitize_server_name(server_name))))
}

pub fn clear_oauth_credentials(server_name: &str) -> Result<()> {
    let Some(store) = FileCredentialStore::for_server(server_name) else {
        return Ok(());
    };
    std::fs::remove_file(store.path()).or_else(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            Ok(())
        } else {
            Err(error)
        }
    })?;
    Ok(())
}

/// Open a loopback callback, start PKCE authorization, and return an authorized manager.
pub async fn authorize_in_browser(
    mcp_url: &str,
    server_name: &str,
    oauth_client_id: Option<&str>,
    challenge: Option<&str>,
    on_status: impl Fn(&str),
) -> Result<AuthorizationManager> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("无法绑定 OAuth 回调端口")?;
    let port = listener.local_addr()?.port();
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");

    let mut manager = AuthorizationManager::new(mcp_url)
        .await
        .with_context(|| format!("无法初始化 MCP Server `{server_name}` 的 OAuth"))?;
    if let Some(store) = FileCredentialStore::for_server(server_name) {
        manager.set_credential_store(store);
    }

    let resolution = manager
        .resolve_metadata_from_challenge(challenge)
        .await
        .with_context(|| format!("MCP Server `{server_name}` 无法发现 OAuth 元数据"))?;
    manager.set_metadata(resolution.metadata);

    let mut request = AuthorizationRequest::new(redirect_uri)
        .with_client_name("Asteria")
        .with_application_type("native");
    if let Some(challenge) = challenge {
        request = request.with_challenge(challenge);
    }
    if let Some(client_id) = oauth_client_id.filter(|id| !id.is_empty()) {
        request = request.with_preregistered_client(client_id);
    }

    let session = AuthorizationSession::new(manager, request)
        .await
        .map_err(|(_, error)| error)
        .with_context(|| {
            format!(
                "MCP Server `{server_name}` 无法开始 OAuth。若该服务不支持动态注册，请在 mcp.json 中设置 oauthClientId"
            )
        })?;
    let auth_url = session.get_authorization_url().to_string();
    on_status(&format!(
        "[MCP] 请在浏览器中授权 `{server_name}`（5 分钟内完成）：\n{auth_url}"
    ));
    if let Err(error) = open_browser(&auth_url) {
        on_status(&format!(
            "[MCP] 无法自动打开浏览器（{error}），请手动打开上面的地址。"
        ));
    }

    let callback_url = wait_for_callback(listener)
        .await
        .with_context(|| format!("MCP Server `{server_name}` 未在超时时间内完成浏览器授权"))?;
    session
        .handle_callback_url(&callback_url)
        .await
        .with_context(|| format!("MCP Server `{server_name}` 交换授权码失败"))?;
    on_status(&format!("[MCP] `{server_name}` 授权成功，正在连接..."));
    Ok(session.auth_manager)
}

fn oauth_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("ASTERIA_OAUTH_DIR") {
        return Some(PathBuf::from(dir));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".asteria").join("oauth"))
}

fn sanitize_server_name(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "server".into()
    } else {
        sanitized
    }
}

fn store_io_error(error: std::io::Error) -> AuthError {
    AuthError::CredentialStoreError(error.to_string())
}

fn open_browser(url: &str) -> Result<()> {
    let mut command = if cfg!(target_os = "macos") {
        let mut command = std::process::Command::new("open");
        command.arg(url);
        command
    } else if cfg!(target_os = "windows") {
        let mut command = std::process::Command::new("cmd");
        command.args(["/C", "start", "", url]);
        command
    } else {
        let mut command = std::process::Command::new("xdg-open");
        command.arg(url);
        command
    };
    command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("无法启动浏览器")?
        .wait()
        .ok();
    Ok(())
}

async fn wait_for_callback(listener: TcpListener) -> Result<String> {
    tokio::select! {
        result = accept_callback(listener) => result,
        _ = tokio::time::sleep(CALLBACK_TIMEOUT) => {
            bail!("OAuth 回调等待超时")
        }
        _ = tokio::signal::ctrl_c() => {
            bail!("已取消 OAuth 授权")
        }
    }
}

async fn accept_callback(listener: TcpListener) -> Result<String> {
    loop {
        let (mut socket, _) = listener.accept().await.context("等待 OAuth 回调失败")?;
        let mut buffer = vec![0_u8; 8192];
        let read = socket
            .read(&mut buffer)
            .await
            .context("读取 OAuth 回调失败")?;
        let request = String::from_utf8_lossy(&buffer[..read]);
        let path = request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or_default();
        let html = if path.contains("error=") {
            "<html><body>授权失败，可以关闭此窗口返回 Asteria。</body></html>"
        } else if path.contains("code=") && path.contains("state=") {
            "<html><body>Asteria 已完成授权，可以关闭此窗口。</body></html>"
        } else {
            let _ =
                write_http_response(&mut socket, 404, "<html><body>Not Found</body></html>").await;
            continue;
        };
        let _ = write_http_response(&mut socket, 200, html).await;
        if path.contains("error=") {
            bail!("OAuth 授权被拒绝或失败: {path}");
        }
        return Ok(format!("http://127.0.0.1{path}"));
    }
}

async fn write_http_response(
    socket: &mut tokio::net::TcpStream,
    status: u16,
    body: &str,
) -> Result<()> {
    let reason = if status == 200 { "OK" } else { "Not Found" };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    socket.write_all(response.as_bytes()).await?;
    socket.shutdown().await.ok();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn temp_oauth_dir() -> PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("asteria-oauth-{id}"));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn sanitizes_server_names_for_credential_files() {
        assert_eq!(
            sanitize_server_name("binance-mcp-server"),
            "binance-mcp-server"
        );
        assert_eq!(sanitize_server_name("weird name/.."), "weird_name___");
        assert_eq!(sanitize_server_name(""), "server");
    }

    #[tokio::test]
    async fn file_store_round_trips_credentials() {
        let dir = temp_oauth_dir();
        let path = dir.join("binance-mcp-server.json");
        let store = FileCredentialStore { path: path.clone() };
        let credentials: StoredCredentials = serde_json::from_value(serde_json::json!({
            "client_id": "grok",
            "token_response": {
                "access_token": "tok",
                "token_type": "bearer",
                "expires_in": 3600,
                "refresh_token": "ref"
            },
            "granted_scopes": ["account"],
            "token_received_at": 1,
            "issuer": "https://agent.binance.com"
        }))
        .unwrap();
        store.save(credentials).await.unwrap();
        let loaded = store.load().await.unwrap().unwrap();
        assert_eq!(loaded.client_id, "grok");
        assert_eq!(loaded.issuer.as_deref(), Some("https://agent.binance.com"));
        store.clear().await.unwrap();
        assert!(store.load().await.unwrap().is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn callback_server_reads_code_and_state() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let accept = tokio::spawn(accept_callback(listener));
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(
                b"GET /callback?code=abc&state=xyz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
            )
            .await
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).await.unwrap();
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let callback = accept.await.unwrap().unwrap();
        assert!(callback.contains("code=abc"));
        assert!(callback.contains("state=xyz"));
    }
}
