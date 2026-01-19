use llm_token_saver_rs::UnifiedContextManager;
use serde_json::json;

fn build_messages() -> Vec<serde_json::Value> {
    let mut messages = Vec::new();
    messages.push(json!({"role":"system","content":"You are a helpful assistant."}));

    for i in 0..120 {
        messages.push(json!({
            "role":"user",
            "content": format!(
                "Turn {}: {}{}",
                i,
                "This is repeated filler text intended to be compressed. ".repeat(20),
                if i % 40 == 0 { "ERROR: timeout in src/api.rs:123" } else { "" }
            )
        }));
        messages.push(json!({
            "role":"assistant",
            "content": format!("Turn {}: {}", i, "Acknowledged. ".repeat(30))
        }));
    }

    messages
}

#[tokio::test]
async fn tier1_reduces_token_estimate() -> anyhow::Result<()> {
    let mut mgr = UnifiedContextManager::new("claude-3-5-sonnet");
    let messages = build_messages();

    let before = mgr.estimate_tokens(&messages);
    let compressed = mgr
        .compress_tier1_extractive(messages, Some("Investigate the timeout error"))
        .await?;
    let after = mgr.estimate_tokens(&compressed);

    assert!(before > 0);
    assert!(
        after < before,
        "expected compression to reduce tokens, got before={} after={}",
        before,
        after
    );

    Ok(())
}

#[test]
fn enforce_budget_never_exceeds() {
    let mgr = UnifiedContextManager::new("claude-3-5-sonnet");
    let messages = build_messages();
    let budget = 1_000;

    let bounded = mgr.enforce_budget(messages, budget);
    let tokens = mgr.estimate_tokens(&bounded);
    assert!(
        tokens <= budget,
        "expected bounded tokens <= budget, got {} > {}",
        tokens,
        budget
    );
}
