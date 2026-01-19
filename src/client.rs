use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

/// Minimal async LLM client interface used by the compression engine.
///
/// Implement this trait using your provider SDK (OpenAI/Anthropic/etc).
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn send_message_simple(&self, messages: Vec<Value>) -> Result<String>;
}
