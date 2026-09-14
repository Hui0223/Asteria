use anyhow::Result;
use asteria_agent::{agent::Asteria, rag::RagStore};
use std::env;

/// 运行本地 RAG 检索示例，可选地调用 DeepSeek 生成最终答案。
#[tokio::main]
async fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let generate = args.next().as_deref() == Some("--generate");
    if !generate {
        args = env::args().skip(1);
    }
    let dir = args.next().unwrap_or_else(|| "docs/rag-demo".into());
    let query = args.collect::<Vec<_>>().join(" ");
    if query.trim().is_empty() {
        eprintln!("用法：cargo run --bin asteria-rag -- <文档目录> <问题>");
        return Ok(());
    }
    let store = RagStore::from_dir(&dir, 500, 50)?;
    let prompt = store.build_prompt(&query, 3);
    println!("知识库片段数：{}\n\n增强 Prompt：\n{}", store.len(), prompt);
    if generate {
        let mut agent = Asteria::new()?;
        let answer = agent.ask(&prompt).await?;
        println!("\nDeepSeek 答案：\n{}", answer);
    }
    Ok(())
}
