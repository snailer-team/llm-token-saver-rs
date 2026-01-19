# llm-token-saver-rs

Rust crate for reducing LLM token consumption in production applications.

This crate is extracted from `snailer.ai` (AI coding agent; 5.4K+ downloads). In production we reduced average token usage by ~30–40% while maintaining response quality.

## Status
- Alpha: API may change.
- Implemented today:
  - Context compression (Tier 1–5) with tool-call integrity helpers
  - Dynamic truncation & hard-cap budgeting
  - Token estimation utilities (heuristic)
  - Selective context filtering for long text blobs (files/search results)
- Roadmap (not implemented yet in this repo): prefix/context caching, playbook-style token injection, provider integrations.

## Install

### Option A: crates.io (after publishing)

```bash
cargo add llm-token-saver-rs
```

This works only after the crate is published to crates.io.

### Option B: Git dependency (works now)

```bash
cargo add llm-token-saver-rs --git https://github.com/snailer-team/llm-token-saver-rs
```

### Option C: Local path (monorepo / local dev)

Add this to your app’s `Cargo.toml`:

```toml
[dependencies]
llm-token-saver-rs = { path = "../llm-token-saver-rs" }
```

In code, import it as `llm_token_saver_rs`:

```rust
use llm_token_saver_rs::UnifiedContextManager;
```

## Quickstart (Tier 1: no LLM calls)

```rust
use llm_token_saver_rs::UnifiedContextManager;
use serde_json::json;

# async fn demo() -> anyhow::Result<()> {
let mut mgr = UnifiedContextManager::new("claude-3-5-sonnet");

let messages = vec![
  json!({"role":"system","content":"You are a helpful assistant."}),
  json!({"role":"user","content":"Please refactor this module."}),
  json!({"role":"assistant","content":"Sure—what constraints?"}),
];

let tokens_before = mgr.estimate_tokens(&messages);
let compressed = mgr.compress_tier1_extractive(messages, None).await?;
let tokens_after = mgr.estimate_tokens(&compressed);

println!("{} -> {} tokens", tokens_before, tokens_after);
# Ok(()) }
```

## Proof it works (reproducible)

Run the test suite (includes budget invariants and a smoke test that Tier 1 reduces token estimates):

```bash
cargo test
```

Run a deterministic efficiency demo that prints before/after token estimates and time:

```bash
cargo run --release --example efficiency
```

Example output:

```text
llm-token-saver-rs efficiency demo (Tier 1)
before_tokens: 116128
after_tokens:  11589
saved_tokens:  104539 (90.0%)
elapsed_ms:    7
budget:        8000 (bounded_tokens=7775)
```

Notes:
- `estimate_tokens()` is a heuristic (≈ 1 token per 4 chars). Use provider usage numbers for billing-accurate metrics.
- Savings depend on your message shape, tool logs, and how much “middle” context can be compressed.

## Quickstart (Tier 2+: with summarization via your LLM client)

Tier 2–5 can call an LLM to summarize parts of context. You provide the client by implementing `LlmClient`.

You'll typically want these dependencies in your app:

```bash
cargo add anyhow async-trait serde_json
```

```rust
use llm_token_saver_rs::{LlmClient, UnifiedContextManager};
use async_trait::async_trait;
use serde_json::json;

struct MyClient;

#[async_trait]
impl LlmClient for MyClient {
  async fn send_message_simple(&self, messages: Vec<serde_json::Value>) -> anyhow::Result<String> {
    // Call your provider here (OpenAI/Anthropic/etc) and return the text response.
    // This crate intentionally does not bundle a provider SDK.
    let _ = messages;
    Ok("summary".to_string())
  }
}

# async fn demo() -> anyhow::Result<()> {
let mut mgr = UnifiedContextManager::new("claude-3-5-sonnet");
let client = MyClient;

let messages = vec![
  json!({"role":"user","content":"Long history ..."}),
  json!({"role":"assistant","content":"More ..."}),
];

let compressed = mgr
  .compress_tier2_query_aware(messages, "What did we decide about API design?", &client)
  .await?;

println!("tokens: {}", mgr.estimate_tokens(&compressed));
# Ok(()) }
```

Note: examples are `async`; run them inside an async runtime like `tokio`.

## Budgeting (hard cap)

If you must guarantee a max budget, use `enforce_budget()` (deterministic) or `compress_hard_cap()` (may summarize via `LlmClient` depending on tier choices).

```rust
use llm_token_saver_rs::UnifiedContextManager;
use serde_json::json;

let mgr = UnifiedContextManager::new("claude-3-5-sonnet");
let budget = 8_000;
let messages = vec![json!({"role":"user","content":"..."})];
let bounded = mgr.enforce_budget(messages, budget);
assert!(mgr.estimate_tokens(&bounded) <= budget);
```

## Selective context filtering (files/search results)

Use `SelectiveContextFilter` for long text blobs where you want to keep only the most informative parts.

```rust
use llm_token_saver_rs::{BudgetPolicy, SelectiveContextFilter};

# async fn demo() -> anyhow::Result<()> {
let filter = SelectiveContextFilter::new("cheap-model");
let policy = BudgetPolicy::default();

let content = "very long file content ...";
let filtered = filter
  .filter_file_content(content, Some("bug in parser"), &policy)
  .await?;

println!("{}", filtered);
# Ok(()) }
```

## Message format

This crate currently operates on `serde_json::Value` messages to stay compatible with common provider payload formats.

Minimum supported shape:
- `{"role": "...", "content": "..."}` or `{"role":"...", "content":[{"type":"text","text":"..."}]}`

If you use tool calls, you may also want:
- `UnifiedContextManager::repair_claude_tool_integrity()`
- `UnifiedContextManager::repair_tool_call_integrity()`
- `UnifiedContextManager::remove_empty_openai_messages()`

## Configuration (environment variables)

`UnifiedContextManager::new()` can read optional environment overrides:
- `LLM_TOKEN_SAVER_MAX_CONTEXT_TOKENS`
- `LLM_TOKEN_SAVER_CONTEXT_PRIMACY_SIZE` (default `3`)
- `LLM_TOKEN_SAVER_CONTEXT_RECENCY_SIZE` (default `5`)
- `LLM_TOKEN_SAVER_TIER1_THRESHOLD` (default `0.85`)
- `LLM_TOKEN_SAVER_TIER2_THRESHOLD` (default `0.95`)
- `LLM_TOKEN_SAVER_CHUNK_SIZE` (default `128`)
- `LLM_TOKEN_SAVER_FINAL_BUDGET_MARGIN` (default is model-dependent)
- `LLM_TOKEN_SAVER_DEBUG` (set to any value to print debug logs)

## Contributing / Security

- Contributing: see `CONTRIBUTING.md`
- Security issues: see `SECURITY.md`

## License

Apache-2.0. See `LICENSE`.
