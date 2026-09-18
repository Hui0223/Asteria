use crate::message::Message;
use serde_json::Value;
use std::collections::HashSet;

const DEFAULT_MAX_MCP_TOOLS: usize = 8;

/// 根据当前 Turn 的用户意图选择模型可见工具，避免把全部 MCP Schema 注入每个请求。
pub fn route(messages: &[Message], tools: &Value) -> Value {
    route_with_limit(messages, tools, DEFAULT_MAX_MCP_TOOLS)
}

fn route_with_limit(messages: &[Message], tools: &Value, max_mcp_tools: usize) -> Value {
    let Some(all_tools) = tools.as_array() else {
        return tools.clone();
    };
    let latest_user = messages
        .iter()
        .rposition(|message| matches!(message, Message::User { .. }));
    let query = latest_user
        .and_then(|index| match &messages[index] {
            Message::User { content } => Some(normalize(content)),
            _ => None,
        })
        .unwrap_or_default();
    let current_turn = latest_user.map_or(&[][..], |index| &messages[index..]);
    let previously_called = current_turn
        .iter()
        .filter_map(|message| match message {
            Message::Assistant { tool_calls, .. } => Some(tool_calls),
            _ => None,
        })
        .flatten()
        .map(|call| call.name.as_str())
        .collect::<HashSet<_>>();

    let mut selected = Vec::new();
    let mut required_mcp = Vec::new();
    let mut ranked_mcp = Vec::new();

    for tool in all_tools {
        let Some(name) = tool_name(tool) else {
            continue;
        };
        if !name.starts_with("mcp__") {
            if query.contains(&name.to_ascii_lowercase()) || route_core_tool(name, &query) {
                selected.push(tool.clone());
            }
            continue;
        }

        let explicitly_named = query.contains(&name.to_ascii_lowercase());
        if explicitly_named || previously_called.contains(name) {
            required_mcp.push((name.to_owned(), tool.clone()));
            continue;
        }
        let score = score_mcp_tool(name, tool_description(tool), &query);
        if score > 0 {
            ranked_mcp.push((score, name.to_owned(), tool.clone()));
        }
    }

    required_mcp.sort_by(|left, right| left.0.cmp(&right.0));
    selected.extend(required_mcp.into_iter().map(|(_, tool)| tool));

    ranked_mcp.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    let remaining = max_mcp_tools.saturating_sub(
        selected
            .iter()
            .filter(|tool| tool_name(tool).is_some_and(|name| name.starts_with("mcp__")))
            .count(),
    );
    selected.extend(
        ranked_mcp
            .into_iter()
            .take(remaining)
            .map(|(_, _, tool)| tool),
    );
    Value::Array(selected)
}

fn route_core_tool(name: &str, query: &str) -> bool {
    match name {
        // 保留低风险且 Schema 很小的基础能力，避免隐式需求退化成凭记忆回答。
        "Read" | "Glob" | "Grep" | "calculate" | "current_time" => true,
        "Write" | "Edit" | "Bash" => contains_any(
            query,
            &[
                "修改",
                "编辑",
                "编写",
                "开发",
                "修复",
                "优化",
                "实现",
                "新增",
                "添加",
                "删除",
                "重构",
                "运行",
                "执行",
                "测试",
                "构建",
                "编译",
                "命令",
                "write",
                "edit",
                "fix",
                "implement",
                "refactor",
                "run",
                "test",
                "build",
                "cargo",
                "git ",
                "shell",
                "bash",
            ],
        ),
        "wait_for" => contains_any(query, &["等待", "wait_for", "wait for"]),
        "search_docs" => contains_any(
            query,
            &[
                "文档",
                "手册",
                "章节",
                "目录",
                "troubleshooting",
                "trouble shooting",
                "search_docs",
            ],
        ),
        // 非 MCP 的嵌入方自定义工具保持可见，避免改变公共 Registry API 语义。
        _ => true,
    }
}

fn score_mcp_tool(name: &str, description: &str, query: &str) -> usize {
    let Some((server, remote_name)) = mcp_parts(name) else {
        return 0;
    };
    let mut score = 0;
    if query.contains(&normalize(server)) {
        score += 80;
    }
    if server_intent_matches(server, query) {
        score += 60;
    }
    for term in split_terms(remote_name) {
        if term.len() >= 3 && contains_search_term(query, &term) {
            score += 18;
        }
    }
    for term in split_terms(description) {
        if term.len() >= 4
            && !is_generic_description_term(&term)
            && contains_search_term(query, &term)
        {
            score += 3;
        }
    }
    score + tool_intent_score(remote_name, query)
}

fn server_intent_matches(server: &str, query: &str) -> bool {
    let server = normalize(server);
    if server.contains("github") {
        return contains_any(
            query,
            &[
                "github",
                "git hub",
                "仓库",
                "repository",
                "repo",
                "issue",
                "pull request",
                " pr ",
                "合并请求",
            ],
        );
    }
    if server.contains("cloudflare") {
        return contains_any(
            query,
            &["cloudflare", "workers", "pages", "wrangler", "d1", "r2"],
        );
    }
    if server.contains("context7") {
        return contains_any(
            query,
            &[
                "context7",
                "第三方库",
                "库文档",
                "api 文档",
                "sdk 文档",
                "latest docs",
                "library docs",
            ],
        );
    }
    if server.contains("deepwiki") {
        return contains_any(
            query,
            &["deepwiki", "wiki", "仓库架构", "代码库架构", "repo 架构"],
        );
    }
    if server.contains("microsoft") {
        return contains_any(
            query,
            &["microsoft", "微软", "learn", "azure", "dotnet", ".net"],
        );
    }
    false
}

fn tool_intent_score(remote_name: &str, query: &str) -> usize {
    let name = normalize(remote_name);
    let mut score = 0;
    let intents: &[(&str, &[&str])] = &[
        (
            "get_me",
            &["我的账号", "我的用户", "个人资料", "profile", "whoami"],
        ),
        (
            "file",
            &["文件", "内容", "读取", "查看", "read", "contents"],
        ),
        ("search", &["搜索", "查找", "检索", "search", "find"]),
        ("issue", &["issue", "工单", "议题"]),
        (
            "pull_request",
            &["pull request", " pr ", "合并请求", "拉取请求"],
        ),
        ("docs", &["文档", "资料", "docs", "documentation"]),
        ("wiki", &["wiki", "架构", "结构"]),
    ];
    for (name_marker, query_markers) in intents {
        if name.contains(name_marker) && contains_any(query, query_markers) {
            score += 30;
        }
    }
    score
}

fn mcp_parts(name: &str) -> Option<(&str, &str)> {
    name.strip_prefix("mcp__")?.split_once("__")
}

fn tool_name(tool: &Value) -> Option<&str> {
    tool["function"]["name"].as_str()
}

fn tool_description(tool: &Value) -> &str {
    tool["function"]["description"].as_str().unwrap_or_default()
}

fn normalize(value: &str) -> String {
    format!(
        " {} ",
        value.to_ascii_lowercase().replace(['`', '\n', '\t'], " ")
    )
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles
        .iter()
        .any(|needle| contains_search_term(haystack, needle))
}

fn contains_search_term(haystack: &str, needle: &str) -> bool {
    if needle
        .chars()
        .all(|character| character.is_ascii_alphanumeric())
    {
        haystack
            .split(|character: char| !character.is_ascii_alphanumeric())
            .any(|term| term == needle)
    } else {
        haystack.contains(needle)
    }
}

fn is_generic_description_term(term: &str) -> bool {
    matches!(
        term,
        "about"
            | "allows"
            | "array"
            | "available"
            | "call"
            | "calls"
            | "data"
            | "details"
            | "from"
            | "function"
            | "information"
            | "list"
            | "name"
            | "parameters"
            | "request"
            | "response"
            | "result"
            | "return"
            | "returns"
            | "server"
            | "this"
            | "tool"
            | "tools"
            | "using"
            | "with"
    )
}

fn split_terms(value: &str) -> impl Iterator<Item = String> + '_ {
    value
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(str::to_ascii_lowercase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema(name: &str, description: &str) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object"}
            }
        })
    }

    fn user(content: &str) -> Vec<Message> {
        vec![Message::User {
            content: content.into(),
        }]
    }

    fn names(tools: &Value) -> Vec<&str> {
        tools
            .as_array()
            .unwrap()
            .iter()
            .filter_map(tool_name)
            .collect()
    }

    #[test]
    fn plain_chat_keeps_readonly_core_and_hides_unrelated_mcp() {
        let tools = json!([
            schema("Read", "read"),
            schema("Glob", "glob"),
            schema("Grep", "grep"),
            schema("calculate", "calculate"),
            schema("current_time", "current time"),
            schema("Write", "write"),
            schema("mcp__github__get_me", "Get my profile")
        ]);
        let routed = route(&user("你好"), &tools);
        assert_eq!(
            names(&routed),
            vec!["Read", "Glob", "Grep", "calculate", "current_time"]
        );
    }

    #[test]
    fn tool_list_introspection_does_not_match_generic_mcp_descriptions() {
        let tools = json!([
            schema("Read", "Read a file"),
            schema("Glob", "Find files"),
            schema("Grep", "Search file contents"),
            schema("calculate", "Calculate an expression"),
            schema("current_time", "Get current time"),
            schema("Write", "Write a file"),
            schema(
                "mcp__context7__resolve-library-id",
                "MCP tool from server context7. Resolve a library identifier"
            ),
            schema(
                "mcp__microsoft-learn__microsoft_docs_fetch",
                "MCP tool from server microsoft-learn. Fetch API documentation"
            ),
            schema(
                "mcp__deepwiki__ask_question",
                "MCP tool from server deepwiki. Ask a repository question"
            )
        ]);
        let routed = route(
            &user(
                "不要调用工具。只列出本次请求中 API tools 数组里的 function.name，不要根据系统提示补充。",
            ),
            &tools,
        );
        assert_eq!(
            names(&routed),
            vec!["Read", "Glob", "Grep", "calculate", "current_time"]
        );
    }

    #[test]
    fn github_intent_selects_relevant_server_tools() {
        let tools = json!([
            schema("Read", "read"),
            schema(
                "mcp__github__get_file_contents",
                "Get repository file contents"
            ),
            schema("mcp__github__issue_read", "Get issue details"),
            schema("mcp__context7__query-docs", "Query library docs")
        ]);
        let routed = route(&user("查看 GitHub 仓库中的 README 文件"), &tools);
        let names = names(&routed);
        assert!(names.contains(&"mcp__github__get_file_contents"));
        assert!(names.contains(&"mcp__github__issue_read"));
        assert!(!names.contains(&"mcp__context7__query-docs"));
    }

    #[test]
    fn explicitly_named_tool_is_never_hidden() {
        let tools = json!([
            schema("Read", "read"),
            schema("current_time", "Current time"),
            schema("mcp__fixture__echo", "Echo text")
        ]);
        let routed = route(
            &user("必须调用 `mcp__fixture__echo`，参数 text 为 hello"),
            &tools,
        );
        assert!(names(&routed).contains(&"mcp__fixture__echo"));
        let routed = route(&user("调用 current_time"), &tools);
        assert!(names(&routed).contains(&"current_time"));
    }

    #[test]
    fn current_turn_called_tool_survives_followup_step() {
        let messages = vec![
            Message::User {
                content: "处理这个请求".into(),
            },
            Message::Assistant {
                content: None,
                tool_calls: vec![crate::message::ToolCall {
                    id: "call-1".into(),
                    name: "mcp__custom__special".into(),
                    arguments: "{}".into(),
                }],
            },
            Message::Tool {
                call_id: "call-1".into(),
                content: "done".into(),
                is_error: false,
            },
        ];
        let tools = json!([schema("mcp__custom__special", "Special operation")]);
        assert_eq!(
            names(&route(&messages, &tools)),
            vec!["mcp__custom__special"]
        );
    }

    #[test]
    fn mcp_candidates_respect_limit() {
        let tools = Value::Array(
            (0..12)
                .map(|index| schema(&format!("mcp__github__search_{index}"), "Search GitHub"))
                .collect(),
        );
        let routed = route_with_limit(&user("搜索 GitHub 仓库"), &tools, 3);
        assert_eq!(routed.as_array().unwrap().len(), 3);
    }

    #[test]
    fn code_change_enables_mutating_and_shell_tools() {
        let tools = json!([
            schema("Read", "read"),
            schema("Write", "write"),
            schema("Edit", "edit"),
            schema("Bash", "bash"),
            schema("wait_for", "wait")
        ]);
        let routed = route(&user("修复并测试这段代码"), &tools);
        let names = names(&routed);
        for expected in ["Read", "Write", "Edit", "Bash"] {
            assert!(names.contains(&expected));
        }
        assert!(!names.contains(&"wait_for"));
    }
}
