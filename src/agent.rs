use crate::{context::ContextMemory, provider::deepseek::DeepSeekProvider, tools};
use anyhow::{Result, bail};

const SYSTEM: &str = "你是 Asteria，一个可靠、简洁的中文 AI 助手。需要精确计算或当前时间时调用工具，不要猜测工具结果。";

/// 对外提供简单问答接口，并协调上下文、模型与工具。
pub struct Asteria {
    provider: DeepSeekProvider,
    context: ContextMemory,
}

impl Asteria {
    /// 根据环境变量创建 Agent 及其 DeepSeek 适配器。
    pub fn new() -> Result<Self> {
        Ok(Self {
            provider: DeepSeekProvider::from_env()?,
            context: ContextMemory::new(SYSTEM),
        })
    }

    /// 返回当前 Agent 使用的模型名称。
    pub fn model(&self) -> &str {
        self.provider.model()
    }

    /// 清空对话历史，但保留 Agent 的系统设定。
    pub fn reset(&mut self) {
        self.context.reset();
    }

    /// 执行一个用户回合；失败时把上下文回滚到本回合开始之前。
    pub fn ask(&mut self, input: &str) -> Result<String> {
        let checkpoint = self.context.checkpoint();
        self.context.append_user(input)?;
        let result = self.complete_turn();
        if result.is_err() {
            self.context.rollback(checkpoint);
        }
        result
    }

    /// 循环调用模型和工具，直到模型给出最终文本或达到步骤上限。
    fn complete_turn(&mut self) -> Result<String> {
        for _ in 0..8 {
            let message = self.provider.complete(&self.context, tools::schema())?;
            let calls = message.tool_calls;
            self.context
                .append_assistant(message.content.clone(), calls.clone())?;
            if calls.is_empty() {
                return Ok(message.content.unwrap_or_default());
            }
            for call in calls {
                let output = tools::execute(&call.name, &call.arguments);
                self.context
                    .append_tool_result(call.id, output.content, output.is_error)?;
            }
        }
        bail!("工具调用轮次过多，已停止")
    }
}
