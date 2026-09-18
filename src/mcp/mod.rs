mod config;
mod manager;
mod tool;

pub use manager::{McpLoadResult, McpManager, McpServerState, McpServerStatus};

/// 加载默认 MCP 配置并连接启用的 Server。
pub async fn load_default(trust_project: bool) -> anyhow::Result<McpLoadResult> {
    Ok(McpManager::connect(config::load_default(trust_project)?).await)
}

/// 只连接配置中的一个 Server，供运行时 connect/reconnect 使用。
pub async fn load_named(trust_project: bool, server_name: &str) -> anyhow::Result<McpLoadResult> {
    let mut config = config::load_default(trust_project)?;
    anyhow::ensure!(
        config
            .servers
            .iter()
            .any(|server| server.name == server_name),
        "MCP 配置中没有 Server `{server_name}`"
    );
    config.servers.retain(|server| server.name == server_name);
    Ok(McpManager::connect(config).await)
}
