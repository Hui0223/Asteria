use anyhow::Result;
use asteria_agent::{agent::Asteria, rag::RagStore};

/// 侧栏 RPC 和 TUI 共用的启动结果；不包含 Terminal，避免 GUI 路径启动 Reedline。
pub struct PreparedAgent {
    pub agent: Asteria,
    pub rag_chunks: Option<usize>,
    pub trust_project_mcp: bool,
}

/// 加载会话、RAG 和项目 MCP 信任开关，不连接 MCP、不创建终端。
pub fn prepare_agent(arguments: &[String]) -> Result<PreparedAgent> {
    let mut agent = Asteria::new()?;
    let rag_chunks = match load_default_rag_store()? {
        Some(store) => {
            let count = store.len();
            agent.enable_search_docs(store);
            Some(count)
        }
        None => None,
    };
    let trust_project_mcp = asteria_agent::mcp::project_config_trusted(
        arguments
            .iter()
            .any(|argument| argument == "--no-project-mcp"),
    );
    Ok(PreparedAgent {
        agent,
        rag_chunks,
        trust_project_mcp,
    })
}

pub fn rag_label(rag_chunks: Option<usize>) -> String {
    rag_chunks.map_or_else(
        || "RAG 未加载".into(),
        |count| format!("RAG {count} chunks"),
    )
}

pub fn mcp_label(agent: &Asteria) -> String {
    let Some(manager) = agent.mcp_manager() else {
        return "MCP 未加载".into();
    };
    let tools = manager
        .statuses()
        .iter()
        .map(|server| server.tools.len())
        .sum::<usize>();
    format!(
        "MCP {}/{} servers · {tools} tools",
        manager.active_connections(),
        manager.statuses().len()
    )
}

fn load_default_rag_store() -> Result<Option<RagStore>> {
    if std::env::var_os("ASTERIA_NO_RAG").is_some() {
        return Ok(None);
    }
    let Some(path) = discover_rag_docs() else {
        return Ok(None);
    };
    Ok(Some(RagStore::from_dir(&path, 500, 50)?))
}

fn discover_rag_docs() -> Option<std::path::PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    if let Ok(canonical) = dir.canonicalize() {
        dir = canonical;
    }
    loop {
        let candidate = dir.join("docs/rag-docs");
        if candidate.is_dir() {
            return Some(candidate);
        }
        if !dir.pop() {
            break;
        }
    }
    let compiled = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs/rag-docs");
    compiled.is_dir().then_some(compiled)
}
