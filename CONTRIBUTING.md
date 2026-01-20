# Contributing

Thanks for contributing!

## Development

- Rust: stable toolchain recommended
- Build: `cargo build`
- Tests: `cargo test`
- Format: `cargo fmt`
- Lint: `cargo clippy --all-targets --all-features`

## Guidelines

- Keep changes focused and well-tested.
- Prefer small, composable APIs; avoid coupling to a specific LLM provider SDK.
- Document externally-visible behavior in `README.md` and user-facing APIs in rustdoc.

## Pull requests

- Include a short summary and rationale.
- If you change behavior, add/adjust tests.
- If you add a feature that requires an LLM call, gate it behind a feature flag and keep a non-network default.

## Repository settings (maintainers)

To prevent merges without review, enable branch protection on the default branch (e.g. `main`):

- Settings → Branches → Add branch protection rule
- Enable:
  - Require a pull request before merging
  - Required approvals: at least 1
  - (Optional) Require review from Code Owners (uses `.github/CODEOWNERS`)
  - Require status checks to pass before merging (select the `PR checks` jobs)
  - Do not allow force pushes
  - (Recommended) Do not allow bypassing the above settings
