use anyhow::{Context, Result};
use asteria_agent::{agent::Asteria, rag::RagStore};
use std::{
    env,
    io::{self, Write},
};

/// 启动持续对话式 RAG：知识源只加载一次，每个问题自动检索并生成回答。
#[tokio::main]
async fn main() -> Result<()> {
    // 独立二进制不会经过主 CLI，因此需要自行加载项目根目录的 .env。
    dotenvy::dotenv().ok();
    let mut args = env::args().skip(1);
    let generate = args.next().as_deref() == Some("--generate");
    if !generate {
        args = env::args().skip(1);
    }
    let first = args.next().unwrap_or_else(|| "docs/rag-demo".into());
    let remote = first == "--url";
    let source = if remote {
        args.next().context("--url 后缺少 URL")?
    } else {
        first
    };
    let is_default_source = !remote && source == "docs/rag-demo";
    let query = args.collect::<Vec<_>>().join(" ");
    let cache = std::path::Path::new(".asteria/rag-cache.json");
    let store = if remote {
        let store = RagStore::from_urls(&[source], 500, 50).await?;
        store.save(cache)?;
        store
    } else if is_default_source && cache.exists() {
        RagStore::load(cache).or_else(|_| RagStore::from_dir(&source, 500, 50))?
    } else {
        RagStore::from_dir(&source, 500, 50)?
    };
    if generate {
        let chunk_count = store.len();
        let mut agent = Asteria::new()?;
        agent.enable_search_docs(store);
        println!("知识库已加载：{chunk_count} 个片段。需要资料时会调用 search_docs；/exit 退出。");
        if !query.trim().is_empty() {
            answer_query(&mut agent, &query).await?;
        }
        loop {
            print!("\n你: ");
            io::stdout().flush()?;
            let mut question = String::new();
            if io::stdin().read_line(&mut question)? == 0 || question.trim() == "/exit" {
                break;
            }
            if question.trim().is_empty() {
                continue;
            }
            answer_query(&mut agent, question.trim()).await?;
        }
    } else {
        println!(
            "知识库片段数：{}\n\n增强 Prompt：\n{}",
            store.len(),
            store.build_prompt(&query, 3)
        );
    }
    Ok(())
}

/// 把原始问题交给 Agent；是否检索由 search_docs 工具决定。
async fn answer_query(agent: &mut Asteria, question: &str) -> Result<()> {
    let answer = agent.ask(question).await?;
    println!("\n=== Asteria RAG 回答 ===\n{answer}");
    Ok(())
}
