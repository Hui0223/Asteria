use anyhow::{Context, Result, bail};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

/// 一段可被检索并注入 Prompt 的本地文档片段。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub source: PathBuf,
    pub index: usize,
    pub text: String,
}

/// 检索结果及其关键词重合分数。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: usize,
}

/// 本地 RAG 知识库：负责加载、切分和关键词检索文档。
pub struct RagStore {
    chunks: Vec<Chunk>,
}

impl RagStore {
    /// 从目录递归读取 UTF-8 文本文件并切分成片段。
    pub fn from_dir(
        path: impl AsRef<Path>,
        chunk_chars: usize,
        overlap_chars: usize,
    ) -> Result<Self> {
        if chunk_chars == 0 || overlap_chars >= chunk_chars {
            bail!("chunk_chars 必须大于 overlap_chars，且不能为 0");
        }
        let mut files = Vec::new();
        collect_files(path.as_ref(), &mut files)?;
        let mut chunks = Vec::new();
        for file in files {
            let text = fs::read_to_string(&file)
                .with_context(|| format!("无法读取 {}", file.display()))?;
            chunks.extend(split_document(&file, &text, chunk_chars, overlap_chars));
        }
        Ok(Self { chunks })
    }

    /// 从 HTTP/HTTPS URL 下载 UTF-8 文本并切分，单个响应最多 2 MiB。
    pub async fn from_urls(
        urls: &[String],
        chunk_chars: usize,
        overlap_chars: usize,
    ) -> Result<Self> {
        if chunk_chars == 0 || overlap_chars >= chunk_chars {
            bail!("chunk_chars 必须大于 overlap_chars，且不能为 0");
        }
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        let mut chunks = Vec::new();
        for url in urls {
            let response = client
                .get(url)
                .send()
                .await
                .with_context(|| format!("无法访问远端文档 {url}"))?
                .error_for_status()
                .with_context(|| format!("远端文档返回错误 {url}"))?;
            if response
                .content_length()
                .is_some_and(|length| length > 2 * 1024 * 1024)
            {
                bail!("远端文档超过 2 MiB 限制: {url}");
            }
            let bytes = response
                .bytes()
                .await
                .with_context(|| format!("无法读取远端文档 {url}"))?;
            if bytes.len() > 2 * 1024 * 1024 {
                bail!("远端文档超过 2 MiB 限制: {url}");
            }
            let text = std::str::from_utf8(&bytes)
                .with_context(|| format!("远端文档不是 UTF-8 文本 {url}"))?;
            chunks.extend(split_document(
                Path::new(url),
                text,
                chunk_chars,
                overlap_chars,
            ));
        }
        Ok(Self { chunks })
    }

    /// 返回知识库中的片段数量。
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// 判断知识库是否没有可检索片段。
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// 按查询词与片段词的重合数量排序，返回最多 top_k 条结果。
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let query_terms = terms(query);
        let minimum_score = minimum_relevance_score(query_terms.len());
        let mut results: Vec<_> = self
            .chunks
            .iter()
            .filter_map(|chunk| {
                let chunk_terms = terms(&chunk.text);
                let score = query_terms.intersection(&chunk_terms).count();
                let coverage_ok = score * 5 >= query_terms.len();
                // 同时要求最低分和至少 20% 的查询词命中，减少长问题误召回。
                (score >= minimum_score && coverage_ok).then(|| SearchResult {
                    chunk: chunk.clone(),
                    score,
                })
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .cmp(&a.score)
                .then_with(|| a.chunk.source.cmp(&b.chunk.source))
                .then_with(|| a.chunk.index.cmp(&b.chunk.index))
        });
        results.truncate(top_k);
        results
    }

    /// 把检索结果组装为带来源标记的增强 Prompt，供 Asteria 或其他模型使用。
    pub fn build_prompt(&self, query: &str, top_k: usize) -> String {
        let results = self.search(query, top_k);
        let context = if results.is_empty() {
            "没有检索到相关本地资料。".into()
        } else {
            results
                .iter()
                .enumerate()
                .map(|(index, result)| {
                    format!(
                        "[资料 {} | {} | score={}]\n{}",
                        index + 1,
                        result.chunk.source.display(),
                        result.score,
                        result.chunk.text
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        };
        format!(
            "请只根据下面的本地资料回答问题；资料不足时明确说不知道。\n\n本地资料：\n{context}\n\n问题：{query}"
        )
    }
}

/// 根据查询关键词数量计算最低相关分数，避免固定阈值造成召回失衡。
fn minimum_relevance_score(term_count: usize) -> usize {
    if term_count == 0 {
        return usize::MAX;
    }
    3.max(term_count.div_ceil(6))
}

/// 递归收集文本文件，跳过隐藏文件和目录。
fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        files.push(path.to_owned());
        return Ok(());
    }
    for entry in fs::read_dir(path).with_context(|| format!("无法读取目录 {}", path.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        collect_files(&entry.path(), files)?;
    }
    files.sort();
    Ok(())
}

/// 优先按 Markdown 段落切分文档，再对过长段落按字符切分。
fn split_document(
    source: &Path,
    text: &str,
    chunk_chars: usize,
    overlap_chars: usize,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut index = 0;
    for section in text
        .split("\n\n")
        .filter(|section| !section.trim().is_empty())
    {
        let chars: Vec<char> = section.chars().collect();
        let mut start = 0;
        while start < chars.len() {
            let end = (start + chunk_chars).min(chars.len());
            let text: String = chars[start..end].iter().collect();
            chunks.push(Chunk {
                source: source.to_owned(),
                index,
                text,
            });
            index += 1;
            if end == chars.len() {
                break;
            }
            start = end - overlap_chars;
        }
    }
    chunks
}

/// 将英文按词、中文按连续双字词归一化，避免单个汉字造成误命中。
fn terms(text: &str) -> HashSet<String> {
    let mut result = HashSet::new();
    let mut ascii = String::new();
    let mut cjk = String::new();
    let flush = |result: &mut HashSet<String>, ascii: &mut String| {
        if !ascii.is_empty() {
            result.insert(ascii.to_lowercase());
            ascii.clear();
        }
    };
    let flush_cjk = |result: &mut HashSet<String>, cjk: &mut String| {
        let chars: Vec<char> = cjk.chars().collect();
        for window in chars.windows(2) {
            result.insert(window.iter().collect());
        }
        cjk.clear();
    };
    for character in text.chars() {
        if character.is_ascii_alphanumeric() {
            flush_cjk(&mut result, &mut cjk);
            ascii.push(character);
        } else {
            flush(&mut result, &mut ascii);
            if character.is_alphanumeric()
                && !character.is_whitespace()
                && !"，。！？、：；（）()[]{}<>\"'".contains(character)
            {
                cjk.push(character);
            } else {
                flush_cjk(&mut result, &mut cjk);
            }
        }
    }
    flush(&mut result, &mut ascii);
    flush_cjk(&mut result, &mut cjk);
    result.retain(|term| !is_query_stopword(term));
    result
}

/// 移除中文疑问句中的功能词，避免句式影响文档相关性评分。
fn is_query_stopword(term: &str) -> bool {
    matches!(
        term,
        "有没有" | "是否" | "介绍" | "什么" | "如何" | "怎么" | "请问" | "以及" | "可以" | "能够"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// 创建测试目录中的文件并在结束后清理目录。
    fn fixture() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "asteria-rag-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        fs::write(
            path.join("guide.md"),
            "Rust ownership makes memory safe.\nAsteria uses a Context Kernel for memory.",
        )
        .unwrap();
        path
    }

    #[test]
    /// 验证文档能够加载并切分。
    fn loads_and_chunks_documents() {
        let path = fixture();
        let store = RagStore::from_dir(&path, 30, 5).unwrap();
        assert!(store.len() >= 2);
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    /// 验证相关关键词片段排名高于无关片段。
    fn ranks_matching_chunks() {
        let path = fixture();
        let store = RagStore::from_dir(&path, 200, 0).unwrap();
        let results = store.search("Asteria Context Kernel", 1);
        assert_eq!(results[0].score, 3);
        assert!(results[0].chunk.text.contains("Context Kernel"));
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    /// 验证检索结果包含来源和用户问题，形成可发送 Prompt。
    fn builds_grounded_prompt() {
        let path = fixture();
        let store = RagStore::from_dir(&path, 200, 0).unwrap();
        let prompt = store.build_prompt("Asteria Context Kernel 是什么？", 2);
        assert!(prompt.contains("guide.md"));
        assert!(prompt.contains("Context Kernel 是什么？"));
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    /// 验证无结果时明确提示资料不足。
    fn reports_missing_context() {
        let path = fixture();
        let store = RagStore::from_dir(&path, 200, 0).unwrap();
        assert!(
            store
                .build_prompt("quantum banana", 2)
                .contains("没有检索到相关本地资料")
        );
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    /// 防止共享单个汉字导致完全无关的问题误命中文档。
    fn avoids_false_positive_from_single_chinese_characters() {
        let path = fixture();
        let store = RagStore::from_dir(&path, 200, 0).unwrap();
        assert!(store.search("量子计算机和火星移民", 3).is_empty());
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    /// 长问题需要更高的关键词重合度，避免少量词命中无关片段。
    fn raises_threshold_for_long_queries() {
        assert_eq!(minimum_relevance_score(3), 3);
        assert_eq!(minimum_relevance_score(18), 3);
        assert_eq!(minimum_relevance_score(24), 4);
        assert_eq!(minimum_relevance_score(0), usize::MAX);
    }

    #[test]
    /// 疑问句功能词不应参与文档相关性评分。
    fn ignores_question_stopwords() {
        let path = fixture();
        let store = RagStore::from_dir(&path, 200, 0).unwrap();
        let results = store.search("Asteria 有没有介绍 Context Kernel？", 1);
        assert_eq!(results[0].score, 3);
        let _ = fs::remove_dir_all(path);
    }
}
