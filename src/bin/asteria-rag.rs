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
    let query = args.collect::<Vec<_>>().join(" ");
    let store = if remote {
        RagStore::from_urls(&[source], 500, 50).await?
    } else {
        RagStore::from_dir(&source, 500, 50)?
    };
    if generate {
        let mut agent = Asteria::new()?;
        println!(
            "知识库已加载：{} 个片段。输入问题，/exit 退出。",
            store.len()
        );
        if !query.trim().is_empty() {
            answer_query(&store, &mut agent, &query).await?;
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
            answer_query(&store, &mut agent, question.trim()).await?;
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

/// 为单个问题检索资料、调用 Asteria 并打印最终答案。
async fn answer_query(store: &RagStore, agent: &mut Asteria, question: &str) -> Result<()> {
    let prompt = store.build_prompt(question, 3);
    let answer = agent.ask(&prompt).await?;
    println!("\nAsteria RAG: {answer}");
    Ok(())
}
