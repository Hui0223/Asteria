use anyhow::Result;
use asteria_agent::rag::RagStore;
use std::env;

/// 运行本地 RAG 检索示例，打印检索片段和增强 Prompt。
fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "docs/rag-demo".into());
    let query = args.collect::<Vec<_>>().join(" ");
    if query.trim().is_empty() {
        eprintln!("用法：cargo run --bin asteria-rag -- <文档目录> <问题>");
        return Ok(());
    }
    let store = RagStore::from_dir(&dir, 500, 50)?;
    println!(
        "知识库片段数：{}\n\n{}",
        store.len(),
        store.build_prompt(&query, 3)
    );
    Ok(())
}
