use crate::tools::{AgentTool, ToolOutput};
use anyhow::{Context, Result, bail};
use quick_xml::events::Event;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

const SEARCH_DOCS_MAX_RESULT_TOKENS: usize = 2_000;

/// 一段可被检索并注入 Prompt 的本地文档片段。
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    pub source: PathBuf,
    pub index: usize,
    pub text: String,
}

/// 检索结果及其 BM25 分数（乘以 100 后取整，便于终端展示）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: usize,
}

/// 完整 RAG 正文压缩后保留的短来源引用，不参与后续模型上下文。
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RagSourceRef {
    pub source: PathBuf,
    pub chunk_indices: Vec<usize>,
}

/// 本地 RAG 知识库：负责加载、切分和关键词检索文档。
pub struct RagStore {
    chunks: Vec<Chunk>,
}

impl RagStore {
    /// 将已切分的知识库保存为 JSON，供下次启动快速恢复。
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_vec_pretty(&self.chunks)?;
        fs::write(path, data).with_context(|| format!("无法保存 RAG 缓存 {}", path.display()))?;
        Ok(())
    }

    /// 从 JSON 恢复已切分的知识库；损坏缓存会返回明确错误。
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let data =
            fs::read(path).with_context(|| format!("无法读取 RAG 缓存 {}", path.display()))?;
        let chunks = serde_json::from_slice(&data)
            .with_context(|| format!("RAG 缓存格式无效 {}", path.display()))?;
        Ok(Self { chunks })
    }

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
            chunks.extend(chunk_document(&file, chunk_chars, overlap_chars)?);
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

    /// 使用 BM25 风格评分排序，返回最多 top_k 条结果。
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        if let Some(number) = requested_chapter_number(query) {
            let results = self.search_chapter(number, top_k.max(8));
            if !results.is_empty() {
                return results;
            }
        }
        let mut results = self.search_keywords(query, top_k);
        if is_outline_query(query) {
            prepend_contents_chunks(self, &mut results, top_k);
        }
        results
    }

    /// 按目录编号取出某一章正文，直到下一章一级标题为止。
    fn search_chapter(&self, number: u32, top_k: usize) -> Vec<SearchResult> {
        let Some(title) = self.numbered_chapter_title(number) else {
            return Vec::new();
        };
        let titles = self.numbered_chapter_titles();
        let mut started = false;
        let mut results = Vec::new();
        for chunk in &self.chunks {
            if is_contents_chunk(chunk) {
                continue;
            }
            if let Some(heading) = chapter_heading_of(chunk, &titles) {
                if started {
                    break;
                }
                if heading == title || heading.starts_with(&title) || title.starts_with(heading) {
                    started = true;
                }
            }
            if started {
                results.push(SearchResult {
                    chunk: chunk.clone(),
                    score: 100,
                });
                if results.len() >= top_k {
                    break;
                }
            }
        }
        results
    }

    fn numbered_chapter_title(&self, number: u32) -> Option<String> {
        self.numbered_chapter_titles()
            .into_iter()
            .find(|(n, _)| *n == number)
            .map(|(_, title)| title)
    }

    fn numbered_chapter_titles(&self) -> Vec<(u32, String)> {
        self.chunks
            .iter()
            .find(|chunk| is_contents_chunk(chunk))
            .map(|chunk| numbered_chapter_titles(&chunk.text))
            .unwrap_or_default()
    }

    fn search_keywords(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        self.search_keywords_in_sources(query, top_k, None)
    }

    fn search_keywords_in_sources(
        &self,
        query: &str,
        top_k: usize,
        sources: Option<&HashSet<PathBuf>>,
    ) -> Vec<SearchResult> {
        let (query_terms, expanded) = query_terms(query);
        let minimum_score = if expanded {
            2
        } else {
            minimum_relevance_score(query_terms.len())
        };
        let documents: Vec<HashSet<String>> = self.chunks.iter().map(|c| terms(&c.text)).collect();
        let avg_len = documents.iter().map(HashSet::len).sum::<usize>() as f64
            / documents.len().max(1) as f64;
        let mut results: Vec<_> = self
            .chunks
            .iter()
            .enumerate()
            .filter_map(|(index, chunk)| {
                if sources.is_some_and(|sources| !sources.contains(&chunk.source)) {
                    return None;
                }
                let chunk_terms = &documents[index];
                let matched = query_terms.intersection(chunk_terms).count();
                let coverage_ok = expanded && matched >= 2 || matched * 5 >= query_terms.len();
                let score = bm25_score(&query_terms, chunk_terms, &documents, avg_len);
                // 同时要求最低分和至少 20% 的查询词命中，减少长问题误召回。
                (matched >= minimum_score && coverage_ok).then(|| SearchResult {
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

    /// 按文件名定位文档并返回其前几个片段，适合用户明确指定某个文件的场景。
    pub fn search_document(&self, file_name: &str, top_k: usize) -> Vec<SearchResult> {
        let sources = self.document_sources(file_name);
        self.chunks
            .iter()
            .filter(|chunk| sources.contains(&chunk.source))
            .take(top_k)
            .map(|chunk| SearchResult {
                chunk: chunk.clone(),
                score: 0,
            })
            .collect()
    }

    /// 在用户指定的文档范围内继续按问题评分，而不是固定返回文档开头。
    fn search_in_document(&self, file_name: &str, query: &str, top_k: usize) -> Vec<SearchResult> {
        let sources = self.document_sources(file_name);
        if sources.is_empty() {
            return Vec::new();
        }
        self.search_keywords_in_sources(query, top_k, Some(&sources))
    }

    /// 优先精确文件名；没有精确结果时允许 troubleshooting.docx 这类简称匹配长文件名。
    fn document_sources(&self, file_name: &str) -> HashSet<PathBuf> {
        let exact = self
            .chunks
            .iter()
            .filter(|chunk| {
                chunk
                    .source
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().eq_ignore_ascii_case(file_name))
            })
            .map(|chunk| chunk.source.clone())
            .collect::<HashSet<_>>();
        if !exact.is_empty() {
            return exact;
        }
        let requested = normalized_file_stem(file_name);
        if requested.len() < 4 {
            return HashSet::new();
        }
        self.chunks
            .iter()
            .filter(|chunk| {
                chunk
                    .source
                    .file_name()
                    .map(|name| normalized_file_stem(&name.to_string_lossy()))
                    .is_some_and(|candidate| candidate.contains(&requested))
            })
            .map(|chunk| chunk.source.clone())
            .collect()
    }

    /// 把检索结果组装为带来源标记的增强 Prompt，供独立 RAG 演示使用。
    pub fn build_prompt(&self, query: &str, top_k: usize) -> String {
        let results = self.search(query, top_k);
        self.build_prompt_from_results(query, &results)
    }

    /// 把检索命中格式化为工具结果；默认预算与 search_docs 工具一致。
    pub fn format_results(&self, results: &[SearchResult]) -> String {
        self.format_results_with_token_budget(results, SEARCH_DOCS_MAX_RESULT_TOKENS)
    }

    /// 按 Token 预算格式化并按内容哈希去重，避免重复片段挤占上下文。
    pub fn format_results_with_token_budget(
        &self,
        results: &[SearchResult],
        max_tokens: usize,
    ) -> String {
        if results.is_empty() {
            return "没有检索到相关本地资料。".into();
        }
        let mut seen = HashSet::new();
        let mut sections = Vec::new();
        let mut used_tokens = 0usize;
        for result in results {
            if !seen.insert(content_hash(&result.chunk.text)) {
                continue;
            }
            let header = format!(
                "[资料 {} | {} | 片段 {} | score={}]",
                sections.len() + 1,
                result.chunk.source.display(),
                result.chunk.index,
                result.score
            );
            let separator_tokens = if sections.is_empty() {
                0
            } else {
                estimate_rag_tokens("\n\n")
            };
            let header_tokens = estimate_rag_tokens(&header) + estimate_rag_tokens("\n");
            let remaining = max_tokens.saturating_sub(
                used_tokens
                    .saturating_add(separator_tokens)
                    .saturating_add(header_tokens),
            );
            if remaining == 0 {
                break;
            }
            let text = fit_text_to_token_budget(&result.chunk.text, remaining);
            if text.is_empty() {
                break;
            }
            let section = format!("{header}\n{text}");
            let section_tokens = estimate_rag_tokens(&section);
            used_tokens = used_tokens
                .saturating_add(separator_tokens)
                .saturating_add(section_tokens);
            sections.push(section);
            if used_tokens >= max_tokens {
                break;
            }
        }
        if sections.is_empty() {
            "没有可放入当前 Token 预算的本地资料。".into()
        } else {
            sections.join("\n\n")
        }
    }

    /// 使用指定的检索结果组装增强 Prompt，支持 BM25 和文件定位共用模板。
    pub fn build_prompt_from_results(&self, query: &str, results: &[SearchResult]) -> String {
        format!(
            "请只根据下面的本地资料回答问题；资料不足时明确说不知道。\n\n本地资料：\n{}\n\n问题：{query}",
            self.format_results(results)
        )
    }
}

/// 供模型按需检索本地知识库的只读工具。
pub struct SearchDocsTool {
    store: Arc<RagStore>,
}

impl SearchDocsTool {
    /// 使用已加载的知识库创建检索工具。
    pub fn new(store: Arc<RagStore>) -> Self {
        Self { store }
    }
}

#[async_trait::async_trait]
impl AgentTool for SearchDocsTool {
    /// 返回工具名 search_docs。
    fn name(&self) -> &str {
        "search_docs"
    }

    /// 返回本地文档检索工具的 Schema。
    fn schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "从本地知识库检索相关文档片段。回答文档、手册、配置、第N章或目录问题时必须先调用。query 尽量使用用户原话；只有用户提供了明确文件名时才填写 document，禁止猜测文件名。不要凭记忆编造。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "检索问题或关键词"
                        },
                        "document": {
                            "type": "string",
                            "description": "可选；用户明确给出的完整文件名或唯一简称。不确定真实文件名时不要填写"
                        }
                    },
                    "required": ["query"]
                }
            }
        })
    }

    /// 按问题检索，或在用户点名文件时直接定位该文档。
    async fn execute(&self, raw_args: &str) -> ToolOutput {
        let args: Value = match serde_json::from_str(raw_args) {
            Ok(args) => args,
            Err(error) => {
                return ToolOutput {
                    content: format!("工具执行失败: {error}"),
                    is_error: true,
                };
            }
        };
        let Some(query) = args["query"].as_str().map(str::trim) else {
            return ToolOutput {
                content: "工具执行失败: 缺少 query 参数".into(),
                is_error: true,
            };
        };
        if query.is_empty() {
            return ToolOutput {
                content: "工具执行失败: query 不能为空".into(),
                is_error: true,
            };
        }
        if query.chars().count() > 500 {
            return ToolOutput {
                content: "工具执行失败: query 过长".into(),
                is_error: true,
            };
        }
        let document = args["document"]
            .as_str()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| mentioned_document_file(query));
        let mut notice = None;
        let results = if requested_chapter_number(query).is_some() {
            self.store.search(query, 8)
        } else if let Some(file_name) = document {
            let sources = self.store.document_sources(&file_name);
            let results = self.store.search_in_document(&file_name, query, 5);
            if results.is_empty() {
                notice = Some(if sources.is_empty() {
                    format!("未找到文件名 `{file_name}`，已改为在整个知识库检索。")
                } else {
                    format!("在 `{file_name}` 内未命中，已改为在整个知识库检索。")
                });
                self.store.search(query, 5)
            } else {
                results
            }
        } else {
            self.store.search(query, 5)
        };
        if requested_chapter_number(query).is_some() && results.is_empty() {
            return ToolOutput {
                content: "没有检索到该章节的正文。".into(),
                is_error: false,
            };
        }
        let content = self
            .store
            .format_results_with_token_budget(&results, SEARCH_DOCS_MAX_RESULT_TOKENS);
        ToolOutput {
            content: match notice {
                Some(notice) => format!("{notice}\n\n{content}"),
                None => content,
            },
            is_error: false,
        }
    }
}

/// 从 search_docs 工具正文中提取并聚合短来源引用。
pub fn source_refs_from_outputs(outputs: &[String]) -> Vec<RagSourceRef> {
    let mut grouped: BTreeMap<PathBuf, BTreeSet<usize>> = BTreeMap::new();
    for line in outputs.iter().flat_map(|output| output.lines()) {
        let Some(header) = line
            .strip_prefix("[资料 ")
            .and_then(|line| line.strip_suffix(']'))
        else {
            continue;
        };
        let fields: Vec<_> = header.split(" | ").collect();
        if fields.len() < 4 {
            continue;
        }
        let Some(index) = fields[2]
            .strip_prefix("片段 ")
            .and_then(|value| value.parse::<usize>().ok())
        else {
            continue;
        };
        grouped
            .entry(PathBuf::from(fields[1]))
            .or_default()
            .insert(index);
    }
    grouped
        .into_iter()
        .map(|(source, indices)| RagSourceRef {
            source,
            chunk_indices: indices.into_iter().collect(),
        })
        .collect()
}

/// 从用户问题中提取明确提到的 DOCX 文件名，支持带空格的路径末段。
fn mentioned_document_file(question: &str) -> Option<String> {
    let marker = ".docx";
    let end = question.to_ascii_lowercase().find(marker)? + marker.len();
    let start = question[..end].rfind('/').map_or(0, |index| index + 1);
    let file_name = question[start..end].trim_matches(['"', '\'', '`']);
    (!file_name.is_empty()).then(|| file_name.to_owned())
}

/// 从单个文件生成片段：目录单独成块，正文再按段落打包。
fn chunk_document(path: &Path, chunk_chars: usize, overlap_chars: usize) -> Result<Vec<Chunk>> {
    let extracted = extract_document(path)?;
    let mut chunks = Vec::new();
    if let Some(outline) = extracted.outline {
        chunks.push(Chunk {
            source: path.to_owned(),
            index: 0,
            text: outline,
        });
    }
    let offset = chunks.len();
    chunks.extend(
        split_document(path, &extracted.text, chunk_chars, overlap_chars)
            .into_iter()
            .map(|mut chunk| {
                chunk.index += offset;
                chunk
            }),
    );
    Ok(chunks)
}

/// 读取普通文本或 Word 文档；Word 会抽出目录大纲并去掉域代码。
struct ExtractedDocument {
    text: String,
    outline: Option<String>,
}

fn extract_document(path: &Path) -> Result<ExtractedDocument> {
    if path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("docx"))
    {
        extract_docx(path)
    } else {
        Ok(ExtractedDocument {
            text: fs::read_to_string(path)
                .with_context(|| format!("无法读取 {}", path.display()))?,
            outline: None,
        })
    }
}

/// 抽取 DOCX 可见正文：跳过域指令、保留制表符，并把 TOC 标题收成独立目录块。
fn extract_docx(path: &Path) -> Result<ExtractedDocument> {
    let file = fs::File::open(path).with_context(|| format!("无法打开 {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("不是有效的 DOCX 文件 {}", path.display()))?;
    let mut xml = String::new();
    archive
        .by_name("word/document.xml")
        .with_context(|| format!("DOCX 缺少正文 {}", path.display()))?
        .read_to_string(&mut xml)?;
    let mut reader = quick_xml::Reader::from_str(&xml);
    let mut body = String::new();
    let mut paragraph = String::new();
    let mut style = String::new();
    let mut outline = Vec::new();
    let mut skip_field = false;
    let mut table_depth = 0usize;
    loop {
        match reader.read_event()? {
            Event::Start(element) => handle_docx_tag(
                &element,
                true,
                &mut style,
                &mut paragraph,
                &mut skip_field,
                &mut table_depth,
            ),
            Event::Empty(element) => handle_docx_tag(
                &element,
                false,
                &mut style,
                &mut paragraph,
                &mut skip_field,
                &mut table_depth,
            ),
            Event::Text(value) => {
                if !skip_field {
                    paragraph.push_str(&value.unescape()?);
                }
            }
            Event::End(value) => {
                let name = value.name();
                let name = name.as_ref();
                if is_docx_tag(name, b"p") {
                    flush_docx_paragraph(&mut body, &mut outline, &paragraph, &style, table_depth);
                    paragraph.clear();
                    style.clear();
                } else if is_docx_tag(name, b"tc") {
                    body.push_str(" | ");
                } else if is_docx_tag(name, b"tr") {
                    body.push('\n');
                } else if is_docx_tag(name, b"tbl") {
                    table_depth = table_depth.saturating_sub(1);
                    body.push_str("\n\n");
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }
    let outline = (!outline.is_empty()).then(|| format!("# Contents\n\n{}", outline.join("\n")));
    Ok(ExtractedDocument {
        text: body,
        outline,
    })
}

/// 把当前段落写入正文，或把 TOC 标题收进目录大纲。
fn flush_docx_paragraph(
    body: &mut String,
    outline: &mut Vec<String>,
    paragraph: &str,
    style: &str,
    table_depth: usize,
) {
    if table_depth > 0 {
        body.push_str(paragraph.trim());
        return;
    }
    if style == "TOC1" || style == "TOC2" {
        if let Some(entry) = format_outline_entry(paragraph) {
            outline.push(entry);
        }
        return;
    }
    if style.starts_with("TOC") {
        return;
    }
    let text = paragraph.trim();
    if text.is_empty() {
        return;
    }
    let prefix = match style {
        "Heading1" | "Title1" | "Title" => "# ",
        "Heading2" => "## ",
        "Heading3" => "### ",
        _ => "",
    };
    body.push_str(prefix);
    body.push_str(text);
    body.push_str("\n\n");
}

/// 用制表符拆开 Word 目录：编号、标题与页码不再粘成一串。
fn format_outline_entry(paragraph: &str) -> Option<String> {
    let mut parts: Vec<&str> = paragraph
        .split('\t')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    if parts
        .last()
        .is_some_and(|part| part.chars().all(|character| character.is_ascii_digit()))
    {
        parts.pop();
    }
    if parts.is_empty() {
        return None;
    }
    if parts.len() >= 2 && is_toc_number(parts[0]) {
        Some(format!("{} {}", parts[0], parts[1..].join(" ")))
    } else {
        Some(parts.join(" "))
    }
}

fn is_toc_number(value: &str) -> bool {
    let mut saw_digit = false;
    for character in value.chars() {
        if character.is_ascii_digit() {
            saw_digit = true;
        } else if character != '.' {
            return false;
        }
    }
    saw_digit
}

fn handle_docx_tag(
    element: &quick_xml::events::BytesStart<'_>,
    is_start: bool,
    style: &mut String,
    paragraph: &mut String,
    skip_field: &mut bool,
    table_depth: &mut usize,
) {
    let name = element.name().as_ref().to_vec();
    if is_docx_tag(&name, b"pStyle") {
        if let Some(value) = docx_attr(element, b"val") {
            *style = value;
        }
    } else if is_docx_tag(&name, b"tab") {
        if !*skip_field {
            paragraph.push('\t');
        }
    } else if is_docx_tag(&name, b"br") {
        if !*skip_field {
            paragraph.push('\n');
        }
    } else if is_docx_tag(&name, b"fldChar") {
        match docx_attr(element, b"fldCharType").as_deref() {
            Some("begin") => *skip_field = true,
            Some("separate" | "end") => *skip_field = false,
            _ => {}
        }
    } else if is_start && is_docx_tag(&name, b"tbl") {
        *table_depth += 1;
    }
}

fn is_contents_chunk(chunk: &Chunk) -> bool {
    chunk.text.starts_with("# Contents")
}

pub(crate) fn is_outline_query(query: &str) -> bool {
    if requested_chapter_number(query).is_some() {
        return false;
    }
    let lower = query.to_ascii_lowercase();
    [
        "章节",
        "目录",
        "几章",
        "多少章",
        "第几章",
        "章",
        "contents",
        "chapters",
        "how many chapter",
    ]
    .iter()
    .any(|needle| query.contains(needle) || lower.contains(needle))
}

pub(crate) fn requested_chapter_number(query: &str) -> Option<u32> {
    if let Some(offset) = query.find('第') {
        let rest = &query[offset + '第'.len_utf8()..];
        let mut digits = String::new();
        let mut after = String::new();
        for character in rest.chars() {
            if digits.is_empty() && character.is_whitespace() {
                continue;
            }
            if character.is_ascii_digit() && after.is_empty() {
                digits.push(character);
                continue;
            }
            after.push(character);
            if after.chars().count() >= 6 {
                break;
            }
        }
        if !digits.is_empty() && after.contains('章') {
            return digits.parse().ok();
        }
    }
    let lower = query.to_ascii_lowercase();
    if let Some(offset) = lower.find("chapter") {
        let rest = &query[offset + 7..];
        let digits: String = rest
            .chars()
            .skip_while(|character| !character.is_ascii_digit())
            .take_while(|character| character.is_ascii_digit())
            .collect();
        if !digits.is_empty() {
            return digits.parse().ok();
        }
    }
    None
}

fn numbered_chapter_titles(outline: &str) -> Vec<(u32, String)> {
    outline
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let (number, title) = line.split_once(' ')?;
            Some((number.parse().ok()?, title.trim().to_string()))
        })
        .collect()
}

fn chapter_heading_of<'a>(chunk: &'a Chunk, titles: &[(u32, String)]) -> Option<&'a str> {
    let first = chunk.text.lines().next()?.strip_prefix("# ")?;
    if first.starts_with('#') {
        return None;
    }
    let first = first.trim();
    titles
        .iter()
        .any(|(_, title)| {
            first == title || first.starts_with(title.as_str()) || title.starts_with(first)
        })
        .then_some(first)
}

fn prepend_contents_chunks(store: &RagStore, results: &mut Vec<SearchResult>, top_k: usize) {
    let outlines: Vec<_> = store
        .chunks
        .iter()
        .filter(|chunk| is_contents_chunk(chunk))
        .cloned()
        .collect();
    if outlines.is_empty() {
        return;
    }
    for chunk in outlines.into_iter().rev() {
        if let Some(index) = results.iter().position(|result| {
            result.chunk.source == chunk.source && result.chunk.index == chunk.index
        }) {
            let hit = results.remove(index);
            results.insert(0, hit);
        } else {
            results.insert(
                0,
                SearchResult {
                    chunk,
                    score: results
                        .first()
                        .map(|result| result.score.saturating_add(1))
                        .unwrap_or(1),
                },
            );
        }
    }
    results.truncate(top_k.max(1));
}

fn docx_attr(element: &quick_xml::events::BytesStart<'_>, local: &[u8]) -> Option<String> {
    element.attributes().flatten().find_map(|attr| {
        let key = attr.key.as_ref();
        let matches = key == local
            || key
                .rsplit(|byte| *byte == b':')
                .next()
                .is_some_and(|name| name == local);
        matches.then(|| String::from_utf8_lossy(&attr.value).into_owned())
    })
}

/// 判断带命名空间前缀的 DOCX XML 标签，例如 `w:p` 或 `w:tc`。
fn is_docx_tag(name: &[u8], local_name: &[u8]) -> bool {
    name == local_name
        || name
            .strip_prefix(b"w:")
            .is_some_and(|name| name == local_name)
}

/// 计算 BM25：稀有词权重更高，词频递增受限，长片段会被长度修正。
fn bm25_score(
    query: &HashSet<String>,
    document: &HashSet<String>,
    documents: &[HashSet<String>],
    avg_len: f64,
) -> usize {
    let n = documents.len() as f64;
    let mut total = 0.0;
    for term in query {
        if !document.contains(term) {
            continue;
        }
        let df = documents.iter().filter(|doc| doc.contains(term)).count() as f64;
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
        let length_norm = 1.0 / (0.5 + 0.5 * document.len() as f64 / avg_len.max(1.0));
        total += idf * length_norm;
    }
    (total * 100.0).round() as usize
}

/// 根据查询关键词数量计算最低相关分数，避免固定阈值造成召回失衡。
fn minimum_relevance_score(term_count: usize) -> usize {
    if term_count == 0 {
        return usize::MAX;
    }
    if term_count <= 4 {
        return 1;
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

/// 短段落打包到预算内；一级标题开始新块；超长段再按字符切分。
fn split_document(
    source: &Path,
    text: &str,
    chunk_chars: usize,
    overlap_chars: usize,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut index = 0;
    let mut current = String::new();
    for section in text
        .split("\n\n")
        .map(str::trim)
        .filter(|section| !section.is_empty())
    {
        if is_heading1(section) && !current.is_empty() {
            push_chunk(source, &mut chunks, &mut index, &mut current);
        }
        let section_chars = section.chars().count();
        if section_chars > chunk_chars {
            if !current.is_empty() {
                push_chunk(source, &mut chunks, &mut index, &mut current);
            }
            let chars: Vec<char> = section.chars().collect();
            let mut start = 0;
            while start < chars.len() {
                let end = (start + chunk_chars).min(chars.len());
                chunks.push(Chunk {
                    source: source.to_owned(),
                    index,
                    text: chars[start..end].iter().collect(),
                });
                index += 1;
                if end == chars.len() {
                    break;
                }
                start = end - overlap_chars;
            }
            continue;
        }
        let next_chars =
            current.chars().count() + if current.is_empty() { 0 } else { 2 } + section_chars;
        if !current.is_empty() && next_chars > chunk_chars {
            push_chunk(source, &mut chunks, &mut index, &mut current);
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(section);
    }
    if !current.is_empty() {
        push_chunk(source, &mut chunks, &mut index, &mut current);
    }
    chunks
}

fn is_heading1(section: &str) -> bool {
    section.starts_with("# ") && !section.starts_with("## ")
}

fn push_chunk(source: &Path, chunks: &mut Vec<Chunk>, index: &mut usize, current: &mut String) {
    let text = current.trim();
    if !text.is_empty() {
        chunks.push(Chunk {
            source: source.to_owned(),
            index: *index,
            text: text.to_owned(),
        });
        *index += 1;
    }
    current.clear();
}

/// 使用稳定 FNV-1a 对规范化正文计算内容哈希，供单次检索结果去重。
fn content_hash(text: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut previous_whitespace = false;
    for character in text.chars() {
        if character.is_whitespace() {
            if previous_whitespace {
                continue;
            }
            previous_whitespace = true;
            for byte in " ".as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        } else {
            previous_whitespace = false;
            let mut bytes = [0u8; 4];
            for byte in character.encode_utf8(&mut bytes).as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
    }
    hash
}

/// 与上下文预算一致的中英文启发式 Token 估算。
fn estimate_rag_tokens(text: &str) -> usize {
    let mut ascii = 0usize;
    let mut non_ascii = 0usize;
    for character in text.chars() {
        if character.is_ascii() {
            ascii += 1;
        } else {
            non_ascii += 1;
        }
    }
    non_ascii + ascii.div_ceil(3)
}

/// 在 UTF-8 边界上裁剪文本，使估算 Token 数不超过给定预算。
fn fit_text_to_token_budget(text: &str, budget: usize) -> String {
    if budget == 0 {
        return String::new();
    }
    if estimate_rag_tokens(text) <= budget {
        return text.to_owned();
    }
    let marker = "\n[片段因 Token 预算截断]";
    let marker_tokens = estimate_rag_tokens(marker);
    if budget <= marker_tokens {
        return fit_text_prefix(marker, budget);
    }
    let mut result = fit_text_prefix(text, budget - marker_tokens);
    result.push_str(marker);
    result
}

fn fit_text_prefix(text: &str, budget: usize) -> String {
    let characters: Vec<char> = text.chars().collect();
    let mut low = 0usize;
    let mut high = characters.len();
    while low < high {
        let middle = (low + high).div_ceil(2);
        let candidate: String = characters[..middle].iter().collect();
        if estimate_rag_tokens(&candidate) <= budget {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    characters[..low].iter().collect()
}

fn normalized_file_stem(file_name: &str) -> String {
    let stem = Path::new(file_name)
        .file_stem()
        .unwrap_or_else(|| std::ffi::OsStr::new(file_name))
        .to_string_lossy();
    stem.chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// 对常见中英混合故障描述补充英文手册术语；扩展词按 OR 召回处理。
fn query_terms(text: &str) -> (HashSet<String>, bool) {
    let mut normalized = text
        .to_lowercase()
        .replace("trouble shooting", "troubleshooting");
    for phrase in ["告诉我", "查看", "文档", "时该如何", "该如何"] {
        normalized = normalized.replace(phrase, " ");
    }
    let bilingual = ["反复重启", "连续重启", "重启", "排查", "调试"]
        .iter()
        .any(|phrase| normalized.contains(phrase));
    normalized = normalized
        .replace(
            "反复重启",
            " continuous reboot rebooting restart restarting startup ",
        )
        .replace(
            "连续重启",
            " continuous reboot rebooting restart restarting startup ",
        )
        .replace("重启", " reboot rebooting restart restarting startup ")
        .replace("排查", " debug troubleshooting ")
        .replace("调试", " debug troubleshooting ");
    (terms(&normalized), bilingual)
}

/// 将英文按词、中文按连续双字词归一化，避免单个汉字造成误命中。
fn terms(text: &str) -> HashSet<String> {
    let normalized = remove_query_phrases(text);
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
    for character in normalized.chars() {
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

/// 删除常见口语问法，让有效主题词不会被无意义的跨词组合干扰。
fn remove_query_phrases(text: &str) -> String {
    ["有啥", "有哪些", "有什么", "能做什么", "能干什么", "介绍"]
        .into_iter()
        .fold(text.to_owned(), |text, phrase| text.replace(phrase, " "))
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
        assert!(results[0].score > 0);
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
        assert_eq!(minimum_relevance_score(1), 1);
        assert_eq!(minimum_relevance_score(2), 1);
        assert_eq!(minimum_relevance_score(3), 1);
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
        assert!(results[0].score > 0);
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    /// 口语化短问题仍应命中文档中的核心主题词。
    fn handles_informal_short_queries() {
        let path = fixture();
        let store = RagStore::from_dir(&path, 200, 0).unwrap();
        assert!(!store.search("Asteria有啥功能", 1).is_empty());
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    /// 自然中文短问句应能命中主题词，而不会被固定阈值过滤。
    fn handles_natural_feature_question() {
        let path = fixture();
        let store = RagStore::from_dir(&path, 200, 0).unwrap();
        assert!(!store.search("介绍Asteria基本功能", 1).is_empty());
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    /// 用户明确指定文件名时，应能定位该文档而不依赖问题关键词得分。
    fn searches_document_by_file_name() {
        let path = fixture();
        let store = RagStore::from_dir(&path, 200, 0).unwrap();
        let results = store.search_document("guide.md", 2);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].score, 0);
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    /// 短段落应打包进同一片段，避免目录被拆成无法计数的碎片。
    fn packs_short_paragraphs_into_one_chunk() {
        let chunks = split_document(Path::new("guide.md"), "第一段\n\n第二段", 100, 0);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("第一段"));
        assert!(chunks[0].text.contains("第二段"));
    }

    #[test]
    /// 一级标题应开启新片段，避免把相邻章节正文粘在一起。
    fn heading_starts_a_new_chunk() {
        let chunks = split_document(
            Path::new("guide.md"),
            "# Chapter 1\n\nbody one\n\n# Chapter 2\n\nbody two",
            500,
            0,
        );
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.contains("Chapter 1"));
        assert!(chunks[1].text.contains("Chapter 2"));
    }

    #[test]
    /// Word 目录必须去掉域代码，并用制表符拆开标题与页码。
    fn docx_skips_field_codes_and_keeps_tab_separated_titles() {
        let dir = fixture();
        let docx = write_sample_docx(
            &dir,
            r#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:pStyle w:val="TOCHeading"/></w:pPr><w:r><w:t>Contents</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="TOC1"/></w:pPr>
<w:r><w:fldChar w:fldCharType="begin"/></w:r>
<w:r><w:instrText> TOC \o "1-3" </w:instrText></w:r>
<w:r><w:fldChar w:fldCharType="separate"/></w:r>
<w:r><w:t>Acronyms</w:t></w:r><w:r><w:tab/></w:r>
<w:r><w:fldChar w:fldCharType="begin"/></w:r>
<w:r><w:instrText> PAGEREF x </w:instrText></w:r>
<w:r><w:fldChar w:fldCharType="separate"/></w:r>
<w:r><w:t>6</w:t></w:r>
<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="TOC1"/></w:pPr>
<w:r><w:t>1</w:t></w:r><w:r><w:tab/></w:r>
<w:r><w:t>Pre-condition</w:t></w:r><w:r><w:tab/></w:r>
<w:r><w:t>7</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Pre-condition</w:t></w:r></w:p>
<w:p><w:r><w:t>Visible body</w:t></w:r></w:p>
</w:body></w:document>"#,
        );
        let extracted = extract_docx(&docx).unwrap();
        assert!(!extracted.text.contains("PAGEREF"));
        assert!(!extracted.text.contains("TOC \\o"));
        assert!(extracted.text.contains("# Pre-condition"));
        assert!(extracted.text.contains("Visible body"));
        let outline = extracted.outline.expect("docx outline");
        assert!(outline.starts_with("# Contents"));
        assert!(outline.contains("Acronyms"));
        assert!(outline.contains("1 Pre-condition"));
        assert!(!outline.contains("PAGEREF"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    /// 故障排查指南的目录块应列出 14 个编号章节。
    fn troubleshooting_outline_lists_fourteen_chapters() {
        let path = Path::new(
            "docs/rag-docs/3HH-90086-2001-DFZZA-01P46-Lightspan Fiber Troubleshooting Guide - SI Part.docx",
        );
        if !path.exists() {
            return;
        }
        let store = RagStore::from_dir(path.parent().unwrap(), 500, 50).unwrap();
        let outline = store
            .chunks
            .iter()
            .find(|chunk| is_contents_chunk(chunk))
            .expect("contents chunk");
        let numbered = outline
            .text
            .lines()
            .filter(|line| is_top_level_chapter(line))
            .count();
        assert_eq!(numbered, 14, "outline was:\n{}", outline.text);
        let results = store.search("troubleshooting 一共包含几个章节", 3);
        assert!(
            results
                .iter()
                .any(|result| is_contents_chunk(&result.chunk)),
            "structure query should retrieve the contents chunk"
        );
    }

    #[test]
    /// “第N章”应返回该章正文，而不是只给目录标题。
    fn troubleshooting_retrieves_chapter_fourteen_body() {
        let path = Path::new(
            "docs/rag-docs/3HH-90086-2001-DFZZA-01P46-Lightspan Fiber Troubleshooting Guide - SI Part.docx",
        );
        if !path.exists() {
            return;
        }
        let store = RagStore::from_dir(path.parent().unwrap(), 500, 50).unwrap();
        assert_eq!(requested_chapter_number("第14章节写的什么内容"), Some(14));
        let results = store.search("第14章节写的什么内容", 8);
        assert!(!results.is_empty(), "chapter 14 body should be retrievable");
        assert!(
            results
                .iter()
                .any(|result| result.chunk.text.contains("Persistent storage")),
            "chapter 14 should include its heading/body, got: {:?}",
            results
                .iter()
                .map(|result| result.chunk.text.chars().take(80).collect::<String>())
                .collect::<Vec<_>>()
        );
        assert!(
            results
                .iter()
                .all(|result| !is_contents_chunk(&result.chunk)),
            "chapter-content queries must not return only the table of contents"
        );
    }

    #[test]
    /// 解析章节号：要求“第N章”，不要把“几个章节”当成第N章。
    fn parses_chapter_number_from_natural_questions() {
        assert_eq!(requested_chapter_number("第14章节写的什么内容"), Some(14));
        assert_eq!(
            requested_chapter_number("第14个章节写的是什么内容？"),
            Some(14)
        );
        assert_eq!(requested_chapter_number("第三章讲的啥"), None);
        assert_eq!(requested_chapter_number("一共包含几个章节"), None);
        assert_eq!(requested_chapter_number("chapter 2 body"), Some(2));
    }

    fn is_top_level_chapter(line: &str) -> bool {
        let line = line.trim();
        let mut chars = line.chars().peekable();
        let mut saw_digit = false;
        while chars
            .peek()
            .is_some_and(|character| character.is_ascii_digit())
        {
            saw_digit = true;
            chars.next();
        }
        saw_digit && chars.next() == Some(' ')
    }

    fn write_sample_docx(dir: &Path, document_xml: &str) -> PathBuf {
        use std::io::Write;
        let path = dir.join("sample.docx");
        let file = fs::File::create(&path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        zip.start_file(
            "word/document.xml",
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
        zip.write_all(document_xml.as_bytes()).unwrap();
        zip.finish().unwrap();
        path
    }

    #[test]
    /// 工具结果只包含资料，不把用户问题伪装成新的 User 消息。
    fn formats_results_without_wrapping_the_question() {
        let path = fixture();
        let store = RagStore::from_dir(&path, 200, 0).unwrap();
        let results = store.search("Asteria Context Kernel", 1);
        let formatted = store.format_results(&results);
        assert!(formatted.contains("guide.md"));
        assert!(formatted.contains("Context Kernel"));
        assert!(!formatted.contains("问题："));
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    /// 检索输出必须遵守 Token 预算，而不是只依赖字符上限。
    fn result_formatting_respects_token_budget() {
        let store = RagStore { chunks: Vec::new() };
        let results = vec![SearchResult {
            chunk: Chunk {
                source: PathBuf::from("a.docx"),
                index: 1,
                text: "中".repeat(200),
            },
            score: 100,
        }];
        let formatted = store.format_results_with_token_budget(&results, 80);
        assert!(estimate_rag_tokens(&formatted) <= 80);
        assert!(formatted.contains("a.docx"));
        assert!(formatted.contains("Token 预算截断"));
        assert!(formatted.chars().count() < 200);
    }

    #[test]
    /// 空白差异不应让同一正文重复进入模型上下文。
    fn deduplicates_results_by_normalized_content_hash() {
        let store = RagStore { chunks: Vec::new() };
        let results = vec![
            SearchResult {
                chunk: Chunk {
                    source: PathBuf::from("first.docx"),
                    index: 1,
                    text: "same   content".into(),
                },
                score: 100,
            },
            SearchResult {
                chunk: Chunk {
                    source: PathBuf::from("second.docx"),
                    index: 2,
                    text: "same content".into(),
                },
                score: 90,
            },
        ];
        let formatted = store.format_results_with_token_budget(&results, 1_000);
        assert_eq!(formatted.matches("[资料 ").count(), 1);
        assert!(formatted.contains("first.docx"));
        assert!(!formatted.contains("second.docx"));
    }

    #[test]
    /// 完整工具正文应压缩为按文件聚合的短来源引用。
    fn extracts_short_source_refs_from_tool_outputs() {
        let outputs = vec![
            "[资料 1 | guide.docx | 片段 14 | score=100]\n正文\n\n\
             [资料 2 | guide.docx | 片段 15 | score=90]\n更多正文"
                .into(),
        ];
        let refs = source_refs_from_outputs(&outputs);
        assert_eq!(
            refs,
            vec![RagSourceRef {
                source: PathBuf::from("guide.docx"),
                chunk_indices: vec![14, 15],
            }]
        );
    }

    #[tokio::test]
    /// search_docs 应按关键词返回带来源的片段。
    async fn search_docs_tool_returns_matching_chunks() {
        let path = fixture();
        let store = RagStore::from_dir(&path, 200, 0).unwrap();
        let tool = SearchDocsTool::new(Arc::new(store));
        let output = tool.execute(r#"{"query":"Asteria Context Kernel"}"#).await;
        assert!(!output.is_error);
        assert!(output.content.contains("guide.md"));
        assert!(output.content.contains("Context Kernel"));
        let _ = fs::remove_dir_all(path);
    }

    #[tokio::test]
    /// 指定文件名时应在该文档内检索；错误文件名不能阻断全库回退。
    async fn search_docs_tool_locates_named_document() {
        let path = fixture();
        let store = RagStore::from_dir(&path, 200, 0).unwrap();
        let tool = SearchDocsTool::new(Arc::new(store));
        let output = tool
            .execute(r#"{"query":"Asteria Context Kernel","document":"guide.md"}"#)
            .await;
        assert!(!output.is_error);
        assert!(output.content.contains("guide.md"));
        assert!(output.content.contains("Context Kernel"));
        let missing = tool
            .execute(r#"{"query":"Asteria Context Kernel","document":"missing.docx"}"#)
            .await;
        assert!(!missing.is_error);
        assert!(missing.content.contains("已改为在整个知识库检索"));
        assert!(missing.content.contains("Context Kernel"));
        let _ = fs::remove_dir_all(path);
    }

    #[tokio::test]
    async fn fuzzy_document_alias_and_chinese_reboot_query_find_english_manual() {
        let path = fixture();
        fs::write(
            path.join("3HH-Lightspan Fiber Troubleshooting Guide - SI Part.md"),
            "Overview and unrelated introduction.\n\n### LT Continuous Rebooting\n\
             Inspect the reboot reason and startup alive message flow before debugging Wallander.",
        )
        .unwrap();
        let store = RagStore::from_dir(&path, 300, 0).unwrap();
        let tool = SearchDocsTool::new(Arc::new(store));
        let output = tool
            .execute(
                r#"{"query":"LT 反复重启时该如何 debug 排查","document":"troubleshooting.docx"}"#,
            )
            .await;
        assert!(!output.is_error);
        assert!(!output.content.contains("未找到文件名"));
        assert!(output.content.contains("LT Continuous Rebooting"));
        assert!(output.content.contains("startup alive message"));
        let _ = fs::remove_dir_all(path);
    }

    #[tokio::test]
    /// 空查询必须失败，避免无意义的全库扫描。
    async fn search_docs_tool_rejects_empty_query() {
        let path = fixture();
        let store = RagStore::from_dir(&path, 200, 0).unwrap();
        let tool = SearchDocsTool::new(Arc::new(store));
        let output = tool.execute(r#"{"query":"   "}"#).await;
        assert!(output.is_error);
        assert!(output.content.contains("query 不能为空"));
        let _ = fs::remove_dir_all(path);
    }
}
