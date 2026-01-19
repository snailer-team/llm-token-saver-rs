use llm_token_saver_rs::UnifiedContextManager;
use serde_json::json;
use std::time::Instant;

fn build_demo_messages() -> Vec<serde_json::Value> {
    let mut messages = Vec::new();

    messages.push(json!({
        "role": "system",
        "content": "You are a helpful assistant. Be concise."
    }));

    // Simulate a long conversation with repeated low-signal content plus a few high-signal bits.
    for i in 0..300 {
        messages.push(json!({
            "role": "user",
            "content": format!(
                "Iteration {}: Please review the code. Notes: {}\n{}\n{}",
                i,
                "This is mostly boilerplate context that repeats across turns. ".repeat(12),
                "Constraints: keep behavior the same; avoid breaking tool calls; focus on correctness.",
                if i % 50 == 0 { "ERROR: E0308 mismatched types in src/lib.rs:42" } else { "" }
            )
        }));
        messages.push(json!({
            "role": "assistant",
            "content": format!(
                "Iteration {}: Acknowledged. {}\n{}",
                i,
                "I will inspect the project structure and suggest changes. ".repeat(10),
                if i % 75 == 0 { "Key file: src/unified_context_manager.rs" } else { "" }
            )
        }));
    }

    messages
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut mgr = UnifiedContextManager::new("claude-3-5-sonnet");

    let messages = build_demo_messages();
    let tokens_before = mgr.estimate_tokens(&messages);

    let started = Instant::now();
    let compressed = mgr
        .compress_tier1_extractive(messages, Some("Fix the E0308 mismatched types error"))
        .await?;
    let elapsed = started.elapsed();

    let tokens_after = mgr.estimate_tokens(&compressed);
    let saved = tokens_before.saturating_sub(tokens_after);
    let saved_pct = if tokens_before > 0 {
        (saved as f64 / tokens_before as f64) * 100.0
    } else {
        0.0
    };

    println!("llm-token-saver-rs efficiency demo (Tier 1)");
    println!("before_tokens: {}", tokens_before);
    println!("after_tokens:  {}", tokens_after);
    println!("saved_tokens:  {} ({:.1}%)", saved, saved_pct);
    println!(
        "elapsed_ms:    {}",
        (elapsed.as_secs_f64() * 1000.0).round() as u64
    );

    // Hard-cap example (deterministic).
    let budget = 8_000;
    let bounded = mgr.enforce_budget(compressed, budget);
    let bounded_tokens = mgr.estimate_tokens(&bounded);
    println!(
        "budget:        {} (bounded_tokens={})",
        budget, bounded_tokens
    );

    Ok(())
}
