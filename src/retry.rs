use crate::agent_loop::CancelToken;
use std::time::Duration;

/// 单个 Step 的请求尝试策略；次数包含第一次请求。
#[derive(Clone, Copy, Debug)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub base_delay: Duration,
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    /// 默认尝试三次，退避从 500ms 开始，上限 8 秒。
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(8),
        }
    }
}

impl RetryPolicy {
    /// 根据失败次数计算有上限的指数退避。
    pub fn delay(&self, failed_attempt: usize) -> Duration {
        self.base_delay
            .saturating_mul(2u32.saturating_pow(failed_attempt.saturating_sub(1).min(31) as u32))
            .min(self.max_delay)
    }
}

/// 仅重试限流、服务端错误、超时及明确的临时网络 I/O 错误。
pub fn is_retryable(error: &anyhow::Error) -> bool {
    if let Some(http) = error.downcast_ref::<reqwest::Error>() {
        if let Some(status) = http.status() {
            return status.as_u16() == 429 || status.is_server_error();
        }
        if http.is_timeout() {
            return true;
        }
    }
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<std::io::Error>())
        .any(|error| {
            matches!(
                error.kind(),
                std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::ConnectionRefused
                    | std::io::ErrorKind::TimedOut
            )
        })
}

/// 异步等待退避或取消信号；不占用线程，取消时返回 false。
pub async fn wait(delay: Duration, cancel: &CancelToken) -> bool {
    tokio::select! {
        biased;
        _ = cancel.cancelled() => false,
        _ = tokio::time::sleep(delay) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    /// 检查退避倍增且不会超过上限。
    fn backoff_is_bounded() {
        let p = RetryPolicy::default();
        assert_eq!(p.delay(1), Duration::from_millis(500));
        assert_eq!(p.delay(2), Duration::from_secs(1));
        assert_eq!(p.delay(100), Duration::from_secs(8));
    }
    #[tokio::test]
    /// 取消信号使长等待立即退出。
    async fn cancelled_wait_returns() {
        let c = CancelToken::default();
        c.cancel();
        assert!(!wait(Duration::from_secs(8), &c).await);
    }
    #[test]
    /// 临时连接中断可重试，普通协议错误不可重试。
    fn network_classification() {
        assert!(is_retryable(
            &std::io::Error::from(std::io::ErrorKind::ConnectionReset).into()
        ));
        assert!(!is_retryable(&anyhow::anyhow!("invalid response")));
    }
}
