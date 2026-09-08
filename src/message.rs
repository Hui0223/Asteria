/// 与供应商无关的工具调用，参数保留为原始 JSON 字符串。
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Agent 内部使用的强类型消息，避免直接操作供应商 JSON。
#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    User {
        content: String,
    },
    Assistant {
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        call_id: String,
        content: String,
        is_error: bool,
    },
}
