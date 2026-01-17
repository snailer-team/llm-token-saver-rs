# llm-token-saver-rs

Rust crate for reducing LLM token consumption in production applications.

## Features
- Prompt compression & summarization
- Prefix/context caching
- Selective token injection (playbook-style)
- Dynamic truncation & budgeting
- Token estimation utilities

## Motivation
In snailer.ai (AI coding agent, 5.1K+ downloads), we reduced average token usage by ~30-40% while maintaining response quality.

## Quick Start
```toml
[dependencies]
llm-token-saver-rs = "0.1.0"  # soon on crates.io
