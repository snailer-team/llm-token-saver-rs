# Contributing

Thanks for contributing!

## Development

- Rust: stable toolchain recommended
- Build: `cargo build`
- Tests: `cargo test`
- Format: `cargo fmt`
- Lint: `cargo clippy --all-targets --all-features -D warnings`

## Guidelines

- Keep changes focused and well-tested.
- Prefer small, composable APIs; avoid coupling to a specific LLM provider SDK.
- Document externally-visible behavior in `README.md` and user-facing APIs in rustdoc.

## Pull requests

- Include a short summary and rationale.
- If you change behavior, add/adjust tests.
- If you add a feature that requires an LLM call, gate it behind a feature flag and keep a non-network default.
