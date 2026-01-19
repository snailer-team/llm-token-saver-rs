//! Unified Context Compression System
//!
//! Based on three research papers:
//! 1. "Lost in the Middle" (Stanford, TACL 2024) - Primacy-Recency optimization
//! 2. "Prompt Compression Methods" (UC Berkeley, ICML 2024) - Compression methods comparison
//! 3. "LLMLingua" (arXiv:2310.05736v2) - Protection rules & budget controller
//!
//! Key insights:
//! - Tier 1 (Extractive): 10× compression, < 5% accuracy loss
//! - Tier 2 (Query-Aware): 30× compression, < 10% accuracy loss
//! - Tier 3 (Aggressive): 50-60× compression, < 20% accuracy loss
//! - LLMLingua: Protect numbers, paths, API signatures, error codes

use crate::client::LlmClient;
use anyhow::Result;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// LLMLingua Protection Rules
/// Protects high-value tokens during compression
#[derive(Debug, Clone)]
pub struct ProtectionRules {
    /// Protect numbers (e.g., "42", "3.14")
    pub protect_numbers: bool,
    /// Protect file paths (e.g., "src/main.rs", "/usr/bin")
    pub protect_paths: bool,
    /// Protect API signatures (e.g., "fn main()", "async def process()")
    pub protect_api_signatures: bool,
    /// Protect error codes (e.g., "E0308", "Error: timeout")
    pub protect_error_codes: bool,
    /// Protect identifiers (e.g., variable names, class names)
    pub protect_identifiers: bool,
    /// Protect URLs (e.g., "https://api.example.com")
    pub protect_urls: bool,
    /// Protect table keys/headers
    pub protect_table_keys: bool,
}

impl Default for ProtectionRules {
    fn default() -> Self {
        Self {
            protect_numbers: true,
            protect_paths: true,
            protect_api_signatures: true,
            protect_error_codes: true,
            protect_identifiers: true,
            protect_urls: true,
            protect_table_keys: true,
        }
    }
}

/// Prompt component classification (LLMLingua Budget Controller)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromptComponent {
    /// System instructions (high weight - preserve)
    Instruction,
    /// Examples/demonstrations (low weight - compress first)
    Demonstrations,
    /// User queries (highest weight - preserve)
    Query,
    /// Tool execution results (medium weight)
    ToolResults,
}

/// Compression method types (UC Berkeley classification)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMethod {
    /// Extractive: Select relevant sentences/passages
    /// Best for: All tasks, 10× compression, < 5% loss
    Extractive,

    /// Query-Aware Abstractive: Summarize with user query
    /// Best for: Multi-document QA, 30× compression, < 10% loss
    QueryAware,

    /// Aggressive: Token pruning + heavy compression
    /// Best for: Last resort, 50-60× compression, < 20% loss
    Aggressive,
}

/// Message position in context window (Lost in the Middle)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessagePosition {
    /// Primacy zone (best performance)
    Primacy,
    /// Middle zone (worst performance - compression target)
    Middle,
    /// Recency zone (best performance)
    Recency,
}

/// Message importance classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessageImportance {
    Critical = 4, // User queries → Primacy zone
    High = 3,     // Tool results → Recency zone
    Medium = 2,   // Planning → Extractive target
    Low = 1,      // Reasoning → Aggressive target
}

/// Classified message with metadata
#[derive(Debug, Clone)]
pub struct ClassifiedMessage {
    pub message: Value,
    pub importance: MessageImportance,
    pub position: MessagePosition,
    pub index: usize,
    pub estimated_tokens: usize,
}

/// Scored message for extractive selection
#[derive(Debug, Clone)]
pub struct ScoredMessage {
    pub message: ClassifiedMessage,
    pub score: f32,
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    pub total_compressions: usize,
    pub tier1_count: usize,
    pub tier2_count: usize,
    pub tier3_count: usize,
    pub tier4_count: usize,
    pub tier5_count: usize,
    pub tokens_saved: usize,
    pub total_time_ms: u128,
}

/// Unified Context Manager
///
/// Implements 3-tier progressive compression:
/// - Tier 1: Extractive (10×, safe)
/// - Tier 2: Query-Aware (30×, moderate)
/// - Tier 3: Aggressive (60×, last resort)
pub struct UnifiedContextManager {
    /// Model's maximum context window (tokens)
    pub(crate) max_tokens: usize,

    /// Primacy zone size (Lost in the Middle)
    primacy_size: usize,

    /// Recency zone size (Lost in the Middle)
    recency_size: usize,

    /// Tier 1 threshold (85% → Extractive)
    tier1_threshold: f32,

    /// Tier 2 threshold (95% → Query-Aware)
    tier2_threshold: f32,

    /// Tier 3 threshold (100% → Aggressive)
    tier3_threshold: f32,

    /// Chunk size for summarization (UC Berkeley: 128 tokens optimal)
    chunk_size: usize,

    /// Preserve sentence boundaries when chunking
    preserve_sentence_boundary: bool,

    /// Current compression tier (0-3)
    current_tier: u8,

    /// Last user query for query-aware compression
    last_user_query: Option<String>,

    /// LLMLingua protection rules
    protection_rules: ProtectionRules,

    /// Statistics
    stats: CompressionStats,

    /// Safety margin for hard cap (model-specific)
    safety_margin: f32,
}

impl UnifiedContextManager {
    /// Create new unified context manager
    pub fn new(model: &str) -> Self {
        let mut max_tokens = Self::get_model_max_tokens(model);
        if let Ok(v) = std::env::var("LLM_TOKEN_SAVER_MAX_CONTEXT_TOKENS") {
            if let Ok(n) = v.parse::<usize>() {
                max_tokens = n;
            }
        }

        // Read configuration from environment
        let primacy_size = std::env::var("LLM_TOKEN_SAVER_CONTEXT_PRIMACY_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);

        let recency_size = std::env::var("LLM_TOKEN_SAVER_CONTEXT_RECENCY_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);

        let tier1_threshold = std::env::var("LLM_TOKEN_SAVER_TIER1_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.85);

        let tier2_threshold = std::env::var("LLM_TOKEN_SAVER_TIER2_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.95);

        let chunk_size = std::env::var("LLM_TOKEN_SAVER_CHUNK_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);

        // Model-specific safety margins
        // MiniMax: 85% (more conservative due to strict limits)
        // Grok: 90% (moderate)
        // Others: 95% (generous)
        let model_lower = model.to_lowercase();
        let default_margin = if model_lower.contains("minimax") || model_lower.contains("m2") {
            0.85
        } else if model_lower.contains("grok") {
            0.90
        } else {
            0.95
        };

        let safety_margin = std::env::var("LLM_TOKEN_SAVER_FINAL_BUDGET_MARGIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_margin);

        Self {
            max_tokens,
            primacy_size,
            recency_size,
            tier1_threshold,
            tier2_threshold,
            tier3_threshold: 1.0,
            chunk_size,
            preserve_sentence_boundary: true,
            current_tier: 0,
            last_user_query: None,
            protection_rules: ProtectionRules::default(),
            stats: CompressionStats::default(),
            safety_margin,
        }
    }

    /// Update manager for a new effective model
    /// - Refreshes max_tokens and safety_margin based on the model name
    /// - Keeps other runtime state (tiers, stats, last_user_query) intact
    pub fn set_model(&mut self, model: &str) {
        // Update max tokens
        self.max_tokens = Self::get_model_max_tokens(model);

        // Recompute safety margin using the same policy as `new()`
        let model_lower = model.to_lowercase();
        let default_margin = if model_lower.contains("minimax") || model_lower.contains("m2") {
            0.85
        } else if model_lower.contains("grok") {
            0.90
        } else {
            0.95
        };

        self.safety_margin = std::env::var("LLM_TOKEN_SAVER_FINAL_BUDGET_MARGIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_margin);

        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            eprintln!(
                "DEBUG: Context manager updated → model='{}', max_tokens={}, margin={:.2}",
                model, self.max_tokens, self.safety_margin
            );
        }
    }

    /// Expose current safety margin (model-specific, may be overridden by ENV)
    pub fn current_margin(&self) -> f32 {
        self.safety_margin
    }

    /// Configure aggressive compression for team mode (orchestrator sub-agents)
    /// PRD: Token saving for orchestrator mode - each role agent uses ~200K max
    /// Thresholds lowered to trigger compression earlier:
    /// - Tier 1 at 50% (was 85%)
    /// - Tier 2 at 65% (was 95%)
    /// - Tier 3 at 80% (was 100%)
    pub fn set_team_mode_aggressive(&mut self) {
        // Cap max tokens for sub-agents (200K per role max)
        self.max_tokens = self.max_tokens.min(200_000);

        // Lower compression thresholds for earlier compression
        self.tier1_threshold = 0.50; // Extractive at 50%
        self.tier2_threshold = 0.65; // Query-Aware at 65%
        self.tier3_threshold = 0.80; // Aggressive at 80%

        // More conservative safety margin
        self.safety_margin = 0.75;

        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            eprintln!(
                "DEBUG: Team mode aggressive compression enabled → max_tokens={}, tiers=[{:.0}%, {:.0}%, {:.0}%], margin={:.0}%",
                self.max_tokens,
                self.tier1_threshold * 100.0,
                self.tier2_threshold * 100.0,
                self.tier3_threshold * 100.0,
                self.safety_margin * 100.0
            );
        }
    }

    /// Calculate the effective hard-cap budget for input context tokens.
    /// Uses model-aware max_tokens and safety margin, optionally constrained by env override.
    pub fn hard_cap_budget(&self) -> usize {
        let model_cap = (self.max_tokens as f32 * self.safety_margin) as usize;
        if let Ok(v) = std::env::var("LLM_TOKEN_SAVER_MAX_CONTEXT_TOKENS") {
            if let Ok(env_cap) = v.parse::<usize>() {
                return env_cap.min(model_cap);
            }
        }
        model_cap
    }

    /// Get maximum tokens for model
    fn get_model_max_tokens(model: &str) -> usize {
        let model_lower = model.to_lowercase();

        // xAI / Grok family (be generous; these models have very large windows)
        if model_lower.contains("grok-4.1-fast") || model_lower.contains("grok-4-1-fast") {
            return 4_000_000; // 4M tokens for Grok 4.1 Fast variants
        }
        if model_lower.contains("grok-4-fast-reasoning")
            || model_lower.contains("grok-4-fast-non-reasoning")
        {
            return 2_000_000; // 2M tokens (reasoning/non‑reasoning fast variants)
        }
        if model_lower.contains("grok-code-fast-1") {
            return 256_000; // code‑fast variant
        }
        if model_lower.contains("grok-4") || model_lower.contains("grok") {
            return 131_072; // default Grok‑4/other grok models
        }

        if model_lower.contains("claude-3-5-sonnet") || model_lower.contains("claude-sonnet-4-5") {
            200_000
        } else if model_lower.contains("claude-3-opus") {
            200_000
        } else if model_lower.contains("claude") {
            100_000
        } else if model_lower.contains("gpt-4-turbo") {
            128_000
        } else if model_lower.contains("gpt-4") {
            8_000
        } else if model_lower.contains("gpt-3.5-turbo-16k") {
            16_000
        } else if model_lower.contains("gpt-3.5") {
            4_000
        } else if model_lower.contains("gpt-5") {
            200_000
        }
        // MiniMax family (CRITICAL FIX - was defaulting to 4k)
        else if model_lower.contains("minimax") || model_lower.contains("m2") {
            10_240 // MiniMax API default
        }
        // Gemini family
        else if model_lower.contains("gemini-2.5")
            || model_lower.contains("gemini-2.5-pro")
            || model_lower.contains("gemini-3")
        // Catch-all for gemini-3-pro, gemini-3-flash, etc.
        {
            1_000_000
        } else {
            4_000 // Conservative default
        }
    }

    /// Estimate token count (1 token ≈ 4 chars)
    /// Special handling for bash tool results which include accurate token estimates
    pub fn estimate_tokens(&self, messages: &[Value]) -> usize {
        let mut total = 0;

        for msg in messages {
            // Check if this is a tool_result message
            if let Some(role) = msg.get("role").and_then(|r| r.as_str()) {
                if role == "tool_result" {
                    // Check if content is an array (Claude API format)
                    if let Some(content_array) = msg.get("content").and_then(|c| c.as_array()) {
                        for item in content_array {
                            // Extract tool_name and text from each item
                            if let (Some(tool_name), Some(text)) = (
                                item.get("tool_name").and_then(|n| n.as_str()),
                                item.get("text").and_then(|t| t.as_str()),
                            ) {
                                // For bash tools, try to extract tokens_estimate from JSON
                                if tool_name == "bash_run" || tool_name == "bash_log" {
                                    if let Ok(result_json) = serde_json::from_str::<Value>(text) {
                                        if let Some(est) = result_json
                                            .get("tokens_estimate")
                                            .and_then(|v| v.as_u64())
                                        {
                                            total += est as usize;
                                            continue;
                                        }
                                    }
                                }
                                // Fallback: use text length
                                total += text.len() / 4;
                            } else {
                                // No tool_name, estimate the whole item
                                let item_str = serde_json::to_string(item).unwrap_or_default();
                                total += item_str.len() / 4;
                            }
                        }
                    } else if let Some(content_str) = msg.get("content").and_then(|c| c.as_str()) {
                        // Content is a string (legacy format)
                        total += content_str.len() / 4;
                    }
                } else {
                    // Non-tool_result message, estimate normally
                    let msg_str = serde_json::to_string(msg).unwrap_or_default();
                    total += msg_str.len() / 4;
                }
            } else {
                // No role field, estimate normally
                let msg_str = serde_json::to_string(msg).unwrap_or_default();
                total += msg_str.len() / 4;
            }
        }

        total
    }

    /// Select compression tier based on current token count
    pub fn select_tier(&self, current_tokens: usize) -> u8 {
        let ratio = current_tokens as f32 / self.max_tokens as f32;

        if ratio >= self.tier3_threshold {
            3
        } else if ratio >= self.tier2_threshold {
            2
        } else if ratio >= self.tier1_threshold {
            1
        } else {
            0
        }
    }

    /// Check if compression is needed
    pub fn needs_compression(&self, messages: &[Value]) -> bool {
        let tokens = self.estimate_tokens(messages);
        let tier = self.select_tier(tokens);

        if tier > 0 && std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            eprintln!(
                "DEBUG: Context needs compression - {} tokens ({:.1}% of {}), recommend Tier {}",
                tokens,
                (tokens as f32 / self.max_tokens as f32) * 100.0,
                self.max_tokens,
                tier
            );
        }

        tier > 0
    }

    // ====================================================================
    // LLMLingua Protection Rules (Phase 1)
    // ====================================================================

    /// Check if a token is a number
    fn is_number(token: &str) -> bool {
        token.parse::<f64>().is_ok()
    }

    /// Check if a token is a file path
    fn is_path(token: &str) -> bool {
        token.contains('/')
            || token.contains('\\')
            || token.ends_with(".rs")
            || token.ends_with(".py")
            || token.ends_with(".js")
            || token.ends_with(".ts")
            || token.ends_with(".go")
            || token.ends_with(".java")
            || token.starts_with("/usr/")
            || token.starts_with("/home/")
            || token.starts_with("~/")
            || token.starts_with("./")
            || token.starts_with("../")
    }

    /// Check if a token is an API signature
    fn is_api_signature(token: &str) -> bool {
        (token.contains('(') && token.contains(')'))
            || token.contains("fn ")
            || token.contains("def ")
            || token.contains("function ")
            || token.contains("async ")
            || token.contains("pub fn")
            || token.contains("impl ")
    }

    /// Check if a token is an error code
    fn is_error_code(token: &str) -> bool {
        token.starts_with("E")
            || token.contains("error")
            || token.contains("Error")
            || token.contains("ERROR")
            || token.starts_with("0x")
            || token.contains("exception")
            || token.contains("Exception")
    }

    /// Check if a token is an identifier (variable/class name)
    fn is_identifier(token: &str) -> bool {
        if token.is_empty() {
            return false;
        }

        let first_char = token.chars().next().unwrap();
        if !first_char.is_alphabetic() && first_char != '_' {
            return false;
        }

        token.chars().all(|c| c.is_alphanumeric() || c == '_')
    }

    /// Check if a token is a URL
    fn is_url(token: &str) -> bool {
        token.starts_with("http://")
            || token.starts_with("https://")
            || token.starts_with("ftp://")
            || token.contains("://")
    }

    /// Calculate token importance based on protection rules
    /// Returns weight multiplier (1.0 = normal, >1.0 = protected)
    fn calculate_token_importance(token: &str, rules: &ProtectionRules) -> f32 {
        let mut weight = 1.0;

        // Numbers (LLMLingua: 5× weight)
        if rules.protect_numbers && Self::is_number(token) {
            weight *= 5.0;
        }

        // File paths (4× weight)
        if rules.protect_paths && Self::is_path(token) {
            weight *= 4.0;
        }

        // API/function signatures (4× weight)
        if rules.protect_api_signatures && Self::is_api_signature(token) {
            weight *= 4.0;
        }

        // Error codes (5× weight - very important)
        if rules.protect_error_codes && Self::is_error_code(token) {
            weight *= 5.0;
        }

        // Identifiers (2× weight)
        if rules.protect_identifiers && Self::is_identifier(token) {
            weight *= 2.0;
        }

        // URLs (3× weight)
        if rules.protect_urls && Self::is_url(token) {
            weight *= 3.0;
        }

        weight
    }

    /// Compress content with protection rules
    /// Removes low-importance tokens first, preserves high-importance tokens
    fn compress_with_protection(
        content: &str,
        target_len: usize,
        rules: &ProtectionRules,
    ) -> String {
        if content.len() <= target_len {
            return content.to_string();
        }

        let tokens: Vec<&str> = content.split_whitespace().collect();

        // Score each token
        let mut scored: Vec<(f32, &str)> = tokens
            .iter()
            .map(|&token| (Self::calculate_token_importance(token, rules), token))
            .collect();

        // Sort by importance (lowest first)
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Remove low-importance tokens until we reach target length
        let mut current_len: usize = tokens.iter().map(|t| t.len() + 1).sum();
        let mut to_remove = HashSet::new();

        for (weight, token) in scored.iter() {
            if current_len <= target_len {
                break;
            }

            // High weight = protected, don't remove
            if *weight > 2.0 {
                continue;
            }

            to_remove.insert(*token);
            current_len = current_len.saturating_sub(token.len() + 1); // +1 for space
        }

        // Reassemble, preserving order
        tokens
            .into_iter()
            .filter(|t| !to_remove.contains(t))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Estimate tokens for a string
    pub fn estimate_tokens_str(&self, s: &str) -> usize {
        (s.len() + 3) / 4 // Conservative: 1 token ≈ 4 chars
    }

    // ====================================================================
    // LLMLingua Budget Controller (Phase 2)
    // ====================================================================

    /// Classify components of prompt messages
    /// Returns map of component type → message indices
    fn classify_components(&self, messages: &[Value]) -> HashMap<PromptComponent, Vec<usize>> {
        let mut components: HashMap<PromptComponent, Vec<usize>> = HashMap::new();

        for (idx, msg) in messages.iter().enumerate() {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");

            let component = if role == "system" {
                // System prompts → Instruction (high weight)
                PromptComponent::Instruction
            } else if role == "user" && (content.contains('?') || content.len() < 200) {
                // Short user messages or questions → Query (highest weight)
                PromptComponent::Query
            } else if content.contains("tool_result") || msg.get("tool_use_id").is_some() {
                // Tool results → ToolResults (medium weight)
                PromptComponent::ToolResults
            } else {
                // Everything else → Demonstrations (low weight)
                PromptComponent::Demonstrations
            };

            components.entry(component).or_default().push(idx);
        }

        components
    }

    /// Allocate token budget across components
    /// LLMLingua: Give more budget to important components (Query > Instruction > ToolResults > Demos)
    fn allocate_budget(&self, messages: &[Value], total_budget: usize) -> HashMap<usize, usize> {
        // 1. Classify messages into components
        let components = self.classify_components(messages);

        // 2. Component weights (LLMLingua recommended values)
        let weights = HashMap::from([
            (PromptComponent::Query, 5.0),       // User queries: highest priority
            (PromptComponent::Instruction, 3.0), // System instructions: high priority
            (PromptComponent::ToolResults, 2.0), // Tool results: medium priority
            (PromptComponent::Demonstrations, 1.0), // Examples: lowest priority (compress first)
        ]);

        // 3. Calculate total weighted messages
        let total_weight: f32 = components
            .iter()
            .map(|(comp, msg_indices)| weights.get(comp).unwrap_or(&1.0) * msg_indices.len() as f32)
            .sum();

        // 4. Allocate budget to each message based on component weight
        let mut allocations = HashMap::new();
        let mut component_counts = HashMap::new();

        for (comp, msg_indices) in components {
            let comp_weight = weights.get(&comp).unwrap_or(&1.0);

            // Total budget for this component
            let comp_budget = ((total_budget as f32) * (comp_weight / total_weight)) as usize;

            // Budget per message in this component
            let per_msg_budget = comp_budget / msg_indices.len().max(1);

            // Track component message count for debug
            component_counts.insert(comp, msg_indices.len());

            for idx in msg_indices {
                allocations.insert(idx, per_msg_budget);
            }
        }

        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            eprintln!("DEBUG: Budget allocation:");
            for (comp, count) in component_counts {
                eprintln!("  {:?}: {} messages", comp, count);
            }
        }

        allocations
    }

    // ====================================================================
    // LLMLingua Iterative Compression (Phase 3)
    // ====================================================================

    /// Iterative token-level compression with protection rules
    /// LLMLingua: "한 번에 크게 자르지 않고, 여러 번 소폭으로"
    async fn compress_iterative(
        &self,
        content: String,
        target_tokens: usize,
        max_rounds: usize,
    ) -> Result<String> {
        let mut current = content;
        let mut round = 0;

        while round < max_rounds {
            let current_tokens = self.estimate_tokens_str(&current);

            if current_tokens <= target_tokens {
                break; // Target achieved
            }

            // This round's target: 25% reduction (gradual!)
            let reduction_rate = 0.25;
            let round_target_tokens =
                current_tokens - (current_tokens as f32 * reduction_rate) as usize;
            let round_target_len = (current.len() as f32
                * (round_target_tokens as f32 / current_tokens as f32))
                as usize;

            // Apply protection rules during compression
            current =
                Self::compress_with_protection(&current, round_target_len, &self.protection_rules);

            // Re-measure (LLMLingua key principle: "재평가 매 라운드")
            round += 1;

            if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                let new_tokens = self.estimate_tokens_str(&current);
                eprintln!(
                    "  Iterative round {}/{}: {} tokens ({:.1}% reduction)",
                    round,
                    max_rounds,
                    new_tokens,
                    ((current_tokens - new_tokens) as f32 / current_tokens as f32) * 100.0
                );
            }
        }

        Ok(current)
    }

    /// Classify message importance
    pub fn classify_importance(message: &Value) -> MessageImportance {
        // 1. User messages are always critical
        if message.get("role").and_then(|r| r.as_str()) == Some("user") {
            return MessageImportance::Critical;
        }

        // 2. Tool calls and results are high priority
        if message.get("tool_calls").is_some() {
            return MessageImportance::High;
        }

        if message.get("tool_use_id").is_some() || message.get("tool_result").is_some() {
            return MessageImportance::High;
        }

        // 3. Check content for importance indicators
        let content = message
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("");

        // Error messages → High
        if content.contains("Error:") || content.contains("error:") {
            return MessageImportance::High;
        }

        // File operations → High
        if Self::contains_file_operations(content) {
            return MessageImportance::High;
        }

        // Planning/analysis → Medium
        if Self::contains_planning_keywords(content) {
            return MessageImportance::Medium;
        }

        // Default: Low (intermediate reasoning)
        MessageImportance::Low
    }

    /// Check for planning keywords
    fn contains_planning_keywords(content: &str) -> bool {
        let keywords = [
            "plan",
            "strategy",
            "approach",
            "design",
            "architecture",
            "structure",
            "organize",
            "step 1",
            "step 2",
            "first",
            "second",
        ];

        let lower = content.to_lowercase();
        keywords.iter().any(|&kw| lower.contains(kw))
    }

    /// Check for file operation mentions
    fn contains_file_operations(content: &str) -> bool {
        let operations = [
            "read file",
            "write file",
            "edit file",
            "src/",
            "lib/",
            ".rs",
            ".py",
            ".js",
            "file_path",
            "directory",
        ];

        let lower = content.to_lowercase();
        operations.iter().any(|&op| lower.contains(op))
    }

    /// Determine message position in context
    fn determine_position(
        index: usize,
        total: usize,
        primacy: usize,
        recency: usize,
    ) -> MessagePosition {
        if index < primacy {
            MessagePosition::Primacy
        } else if index >= total.saturating_sub(recency) {
            MessagePosition::Recency
        } else {
            MessagePosition::Middle
        }
    }

    /// Classify all messages with importance and position
    pub fn classify_messages(&self, messages: &[Value]) -> Vec<ClassifiedMessage> {
        let total = messages.len();

        messages
            .iter()
            .enumerate()
            .map(|(index, message)| {
                let importance = Self::classify_importance(message);
                let position =
                    Self::determine_position(index, total, self.primacy_size, self.recency_size);
                let estimated_tokens = self.estimate_tokens(&[message.clone()]);

                ClassifiedMessage {
                    message: message.clone(),
                    importance,
                    position,
                    index,
                    estimated_tokens,
                }
            })
            .collect()
    }

    /// Partition messages by position
    pub fn partition_by_position(
        &self,
        classified: Vec<ClassifiedMessage>,
    ) -> (
        Vec<ClassifiedMessage>,
        Vec<ClassifiedMessage>,
        Vec<ClassifiedMessage>,
    ) {
        let mut primacy = Vec::new();
        let mut middle = Vec::new();
        let mut recency = Vec::new();

        for msg in classified {
            match msg.position {
                MessagePosition::Primacy => primacy.push(msg),
                MessagePosition::Middle => middle.push(msg),
                MessagePosition::Recency => recency.push(msg),
            }
        }

        (primacy, middle, recency)
    }

    /// Ensure zones have complete tool pairs by moving required messages
    /// This maintains Lost in the Middle zones while satisfying API requirements
    fn ensure_zone_tool_completeness(
        &self,
        primacy: &mut Vec<ClassifiedMessage>,
        middle: &mut Vec<ClassifiedMessage>,
        recency: &mut Vec<ClassifiedMessage>,
        all_messages: &[ClassifiedMessage],
    ) {
        use std::collections::{HashMap, HashSet};

        // Build tool_use_id -> message index map
        let mut tool_use_map: HashMap<String, usize> = HashMap::new();
        for msg in all_messages.iter() {
            let tool_ids = Self::extract_tool_use_ids(&msg.message);
            for id in tool_ids {
                tool_use_map.insert(id, msg.index);
            }
        }

        // Collect all tool_result IDs in primacy + recency
        let mut needed_tool_use_indices = HashSet::new();

        for msg in primacy.iter().chain(recency.iter()) {
            let result_ids = Self::extract_tool_result_ids(&msg.message);
            for result_id in result_ids {
                if let Some(&use_idx) = tool_use_map.get(&result_id) {
                    needed_tool_use_indices.insert(use_idx);
                }
            }
        }

        // Move needed tool_use messages from middle to appropriate zone
        let mut to_move_indices: Vec<usize> = needed_tool_use_indices.into_iter().collect();
        to_move_indices.sort();

        for idx in to_move_indices {
            if let Some(pos) = middle.iter().position(|m| m.index == idx) {
                let msg = middle.remove(pos);
                // Add to primacy if closer to start, recency if closer to end
                if msg.index < all_messages.len() / 2 {
                    primacy.push(msg);
                } else {
                    recency.push(msg);
                }
            }
        }

        // Re-sort to maintain order
        primacy.sort_by_key(|m| m.index);
        recency.sort_by_key(|m| m.index);
    }

    /// Extract tool_use IDs from a message
    fn extract_tool_use_ids(message: &Value) -> Vec<String> {
        let mut ids = Vec::new();

        // Check for tool_calls in content array (Anthropic/Claude format)
        if let Some(content) = message.get("content").and_then(|c| c.as_array()) {
            for item in content {
                if item.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                    if let Some(id) = item.get("id").and_then(|i| i.as_str()) {
                        ids.push(id.to_string());
                    }
                }
            }
        }

        // Check for tool_calls field (OpenAI/Moonshot format)
        if let Some(tool_calls) = message.get("tool_calls").and_then(|tc| tc.as_array()) {
            for tool_call in tool_calls {
                if let Some(id) = tool_call.get("id").and_then(|i| i.as_str()) {
                    ids.push(id.to_string());
                }
            }
        }

        ids
    }

    /// Extract tool_result IDs from a message
    fn extract_tool_result_ids(message: &Value) -> Vec<String> {
        let mut ids = Vec::new();

        // Check for tool_result in content array (Anthropic/Claude format)
        if let Some(content) = message.get("content").and_then(|c| c.as_array()) {
            for item in content {
                if item.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                    if let Some(id) = item.get("tool_use_id").and_then(|i| i.as_str()) {
                        ids.push(id.to_string());
                    }
                }
            }
        }

        // Check for tool_call_id field (OpenAI/Moonshot format)
        // In OpenAI format, tool results have role: "tool" and tool_call_id field
        if let Some(role) = message.get("role").and_then(|r| r.as_str()) {
            if role == "tool" {
                if let Some(id) = message.get("tool_call_id").and_then(|i| i.as_str()) {
                    ids.push(id.to_string());
                }
            }
        }

        ids
    }

    /// Build dependency map: tool_result_id -> tool_use message index
    fn build_tool_dependencies(
        messages: &[ClassifiedMessage],
    ) -> std::collections::HashMap<String, usize> {
        use std::collections::HashMap;

        let mut tool_use_map: HashMap<String, usize> = HashMap::new();

        // First pass: collect all tool_use IDs
        for (idx, msg) in messages.iter().enumerate() {
            let tool_ids = Self::extract_tool_use_ids(&msg.message);
            for id in tool_ids {
                tool_use_map.insert(id, idx);
            }
        }

        tool_use_map
    }

    /// Detect if a message has a tool_result block
    fn is_tool_result_message(message: &Value) -> bool {
        // Check Claude format (content array with type: "tool_result")
        if let Some(content) = message.get("content").and_then(|c| c.as_array()) {
            for item in content {
                if item.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                    return true;
                }
            }
        }
        // Check OpenAI format (role: "tool")
        if let Some(role) = message.get("role").and_then(|r| r.as_str()) {
            if role == "tool" {
                return true;
            }
        }
        false
    }

    /// Check if message contains tool_use or tool_result
    fn is_tool_message(message: &Value) -> bool {
        // Check for tool_use/tool_result in content array (Claude format)
        if let Some(content) = message.get("content").and_then(|c| c.as_array()) {
            for item in content {
                let item_type = item.get("type").and_then(|t| t.as_str());
                if item_type == Some("tool_use") || item_type == Some("tool_result") {
                    return true;
                }
            }
        }
        // Check for tool_calls field (OpenAI format - assistant message with tool calls)
        if message.get("tool_calls").is_some() {
            return true;
        }
        // Check for role: "tool" (OpenAI format - tool result message)
        if let Some(role) = message.get("role").and_then(|r| r.as_str()) {
            if role == "tool" {
                return true;
            }
        }
        false
    }

    /// Extract the first tool_result id, if any
    fn first_tool_result_id(message: &Value) -> Option<String> {
        // Check Claude format (content array with type: "tool_result")
        if let Some(content) = message.get("content").and_then(|c| c.as_array()) {
            for item in content {
                if item.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                    if let Some(id) = item.get("tool_use_id").and_then(|s| s.as_str()) {
                        return Some(id.to_string());
                    }
                }
            }
        }
        // Check OpenAI format (role: "tool" with tool_call_id)
        if let Some(role) = message.get("role").and_then(|r| r.as_str()) {
            if role == "tool" {
                if let Some(id) = message.get("tool_call_id").and_then(|s| s.as_str()) {
                    return Some(id.to_string());
                }
            }
        }
        None
    }

    /// Validate that a tool message has a non-empty tool_call_id
    /// Returns true if message is valid (or not a tool message)
    fn validate_tool_message(message: &Value) -> bool {
        // Check OpenAI format: role: "tool" messages must have non-empty tool_call_id
        if let Some(role) = message.get("role").and_then(|r| r.as_str()) {
            if role == "tool" {
                if let Some(id) = message.get("tool_call_id").and_then(|i| i.as_str()) {
                    if id.is_empty() {
                        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                            eprintln!("DEBUG: Filtering out tool message with empty tool_call_id");
                        }
                        return false;
                    }
                } else {
                    if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                        eprintln!("DEBUG: Filtering out tool message missing tool_call_id field");
                    }
                    return false;
                }
            }
        }

        // Check OpenAI format: assistant messages with tool_calls must have non-empty ids
        if let Some(tool_calls) = message.get("tool_calls").and_then(|tc| tc.as_array()) {
            for tool_call in tool_calls {
                if let Some(id) = tool_call.get("id").and_then(|i| i.as_str()) {
                    if id.is_empty() {
                        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                            eprintln!(
                                "DEBUG: Filtering out assistant message with empty tool_call id"
                            );
                        }
                        return false;
                    }
                } else {
                    if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                        eprintln!(
                            "DEBUG: Filtering out assistant message with tool_call missing id"
                        );
                    }
                    return false;
                }
            }
        }

        // Check Claude format: content array with tool_use/tool_result must have non-empty ids
        if let Some(content) = message.get("content").and_then(|c| c.as_array()) {
            for item in content {
                let item_type = item.get("type").and_then(|t| t.as_str());

                if item_type == Some("tool_use") {
                    if let Some(id) = item.get("id").and_then(|i| i.as_str()) {
                        if id.is_empty() {
                            if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                                eprintln!("DEBUG: Filtering out message with empty tool_use id");
                            }
                            return false;
                        }
                    } else {
                        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                            eprintln!("DEBUG: Filtering out message with tool_use missing id");
                        }
                        return false;
                    }
                } else if item_type == Some("tool_result") {
                    if let Some(id) = item.get("tool_use_id").and_then(|i| i.as_str()) {
                        if id.is_empty() {
                            if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                                eprintln!("DEBUG: Filtering out message with empty tool_result tool_use_id");
                            }
                            return false;
                        }
                    } else {
                        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                            eprintln!(
                                "DEBUG: Filtering out message with tool_result missing tool_use_id"
                            );
                        }
                        return false;
                    }
                }
            }
        }

        // All checks passed
        true
    }

    /// Ensure tool_use/tool_result pairs are preserved together
    fn preserve_tool_pairs(
        &self,
        selected: &mut Vec<ClassifiedMessage>,
        all_messages: &[ClassifiedMessage],
    ) {
        use std::collections::HashSet;

        let tool_deps = Self::build_tool_dependencies(all_messages);
        let mut indices_to_add: HashSet<usize> = HashSet::new();

        // For each selected message, check if it has tool_result
        for msg in selected.iter() {
            let result_ids = Self::extract_tool_result_ids(&msg.message);

            // Find corresponding tool_use messages and mark them for inclusion
            for result_id in result_ids {
                if let Some(&use_idx) = tool_deps.get(&result_id) {
                    if !selected.iter().any(|m| m.index == use_idx) {
                        indices_to_add.insert(use_idx);
                    }
                }
            }
        }

        // Also check reverse: if we have tool_use, we need tool_result
        let mut tool_use_ids_in_selected = HashSet::new();
        for msg in selected.iter() {
            let use_ids = Self::extract_tool_use_ids(&msg.message);
            tool_use_ids_in_selected.extend(use_ids);
        }

        // Find tool_results for our tool_uses
        for msg in all_messages.iter() {
            let result_ids = Self::extract_tool_result_ids(&msg.message);
            for result_id in result_ids {
                if tool_use_ids_in_selected.contains(&result_id) {
                    if !selected.iter().any(|m| m.index == msg.index) {
                        indices_to_add.insert(msg.index);
                    }
                }
            }
        }

        // Add the required messages
        for idx in indices_to_add {
            if let Some(msg) = all_messages.iter().find(|m| m.index == idx) {
                selected.push(msg.clone());
            }
        }

        // Re-sort by original index to maintain order
        selected.sort_by_key(|m| m.index);
    }

    /// Update last user query for query-aware compression
    pub fn update_last_query(&mut self, query: Option<String>) {
        self.last_user_query = query;
    }

    /// Escalate to next tier
    pub fn escalate_tier(&mut self) {
        self.current_tier = (self.current_tier + 1).min(3);
    }

    /// Reset tier for next request
    pub fn reset_tier(&mut self) {
        self.current_tier = 0;
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> &CompressionStats {
        &self.stats
    }

    // ============================================
    // Tier 1: Extractive Compression (UC Berkeley + Lost in Middle)
    // ============================================

    /// Tier 1: Extractive Compression
    ///
    /// Strategy (UC Berkeley: "strongest baseline"):
    /// 1. Classify messages by importance + position
    /// 2. Preserve Primacy (Critical) + Recency (High) zones
    /// 3. Extract top-K from Middle zone by relevance
    ///
    /// Expected: 10× compression, < 5% accuracy loss
    pub async fn compress_tier1_extractive(
        &mut self,
        messages: Vec<Value>,
        user_query: Option<&str>,
    ) -> Result<Vec<Value>> {
        let start_time = Instant::now();
        let start_tokens = self.estimate_tokens(&messages);

        // 1. Classify messages
        let classified = self.classify_messages(&messages);

        // 2. Partition by position (Lost in the Middle)
        let (primacy, middle, recency) = self.partition_by_position(classified);

        // 2.5. Collect primacy + recency to preserve (keep middle for compression)
        // DO NOT call preserve_tool_pairs yet - we'll do it after selecting middle
        let mut primacy_recency = primacy.clone();
        primacy_recency.extend(recency.clone());

        if middle.is_empty() {
            if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                eprintln!("DEBUG Tier 1: No middle messages to compress");
            }
            return Ok(messages);
        }

        // 3. Score middle messages by relevance
        let scored_middle = self.score_by_relevance(&middle, user_query);

        // 4. Select top-K from middle (10× compression = keep ~10%)
        let target_tokens = start_tokens / 10;
        let selected_middle = self.select_top_k(scored_middle, target_tokens, &primacy, &recency);

        // 5. Combine: primacy + recency + selected_middle
        let mut all_kept = primacy_recency;
        all_kept.extend(selected_middle);

        // 6. NOW preserve tool pairs across ALL kept messages
        // This ensures tool_use and tool_result stay together
        let all_classified = self.classify_messages(&messages);
        self.preserve_tool_pairs(&mut all_kept, &all_classified);

        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            eprintln!(
                "DEBUG Tier 1: final kept messages = {} (after tool pairs)",
                all_kept.len()
            );
        }

        // 7. Sort by original index to maintain conversation flow
        // CRITICAL: This keeps tool_use immediately before tool_result
        all_kept.sort_by_key(|m| m.index);

        let mut result: Vec<Value> = all_kept.into_iter().map(|cm| cm.message).collect();
        // Repair Claude tool_use/tool_result integrity first (if Claude format detected)
        self.repair_claude_tool_integrity(&mut result);
        // Then repair Moonshot/OpenAI tool-call integrity
        self.repair_tool_call_integrity(&mut result);

        // 6. Update statistics
        let end_tokens = self.estimate_tokens(&result);
        let _compression_ratio = start_tokens as f32 / end_tokens.max(1) as f32;
        let saved = start_tokens.saturating_sub(end_tokens);

        self.stats.tier1_count += 1;
        self.stats.total_compressions += 1;
        self.stats.tokens_saved += saved;
        self.stats.total_time_ms += start_time.elapsed().as_millis();

        Ok(result)
    }

    /// Remove empty content messages for OpenAI/Grok API.
    ///
    /// OpenAI/Grok API requires all messages to have non-empty content (except the final assistant message).
    /// This function removes messages with empty content.
    pub fn remove_empty_openai_messages(&self, messages: &mut Vec<Value>) {
        if messages.is_empty() {
            return;
        }

        // Check if last message is assistant (it's allowed to be empty)
        let last_is_assistant = messages
            .last()
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            == Some("assistant");

        let mut i = 0usize;
        while i < messages.len() {
            // Skip last message if it's assistant (allowed to be empty)
            if last_is_assistant && i == messages.len() - 1 {
                i += 1;
                continue;
            }

            let msg = &messages[i];
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

            // Skip tool messages - they need tool_call_id, not content validation here
            if role == "tool" {
                i += 1;
                continue;
            }

            let is_empty = {
                // Check if content field exists and is non-empty
                if let Some(content) = msg.get("content") {
                    // Content is string
                    if let Some(text) = content.as_str() {
                        text.trim().is_empty()
                    }
                    // Content is array (shouldn't happen in OpenAI format, but handle it)
                    else if let Some(arr) = content.as_array() {
                        arr.is_empty()
                            || arr.iter().all(|item| {
                                if let Some(text) = item.as_str() {
                                    text.trim().is_empty()
                                } else if let Some(text) = item.get("text").and_then(|t| t.as_str())
                                {
                                    text.trim().is_empty()
                                } else {
                                    false
                                }
                            })
                    } else {
                        // Content is neither string nor array
                        true
                    }
                } else {
                    // No content field
                    true
                }
            };

            if is_empty {
                if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                    eprintln!(
                        "DEBUG: Removing empty content message at index {} (role: {})",
                        i, role
                    );
                }
                messages.remove(i);
                // Don't increment i since we removed the element
            } else {
                i += 1;
            }
        }
    }

    /// Remove empty content messages for Claude API.
    ///
    /// Claude API requires all messages (except the final assistant message) to have non-empty content.
    /// This function removes messages with empty content.
    pub fn remove_empty_claude_messages(&self, messages: &mut Vec<Value>) {
        if messages.is_empty() {
            return;
        }

        // Check if last message is assistant (it's allowed to be empty)
        let last_is_assistant = messages
            .last()
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            == Some("assistant");

        let mut i = 0usize;
        while i < messages.len() {
            // Skip last message if it's assistant (allowed to be empty)
            if last_is_assistant && i == messages.len() - 1 {
                i += 1;
                continue;
            }

            let is_empty = {
                let msg = &messages[i];

                // Check if content field exists
                if let Some(content) = msg.get("content") {
                    // Content is string
                    if let Some(text) = content.as_str() {
                        text.trim().is_empty()
                    }
                    // Content is array
                    else if let Some(arr) = content.as_array() {
                        if arr.is_empty() {
                            true
                        } else {
                            // Check if all items are empty text blocks
                            let mut has_non_empty = false;
                            for item in arr {
                                if let Some(item_type) = item.get("type").and_then(|t| t.as_str()) {
                                    // tool_use and tool_result are valid even if empty
                                    if item_type == "tool_use" || item_type == "tool_result" {
                                        has_non_empty = true;
                                        break;
                                    }
                                    // text blocks must be non-empty
                                    if item_type == "text" {
                                        if let Some(text) =
                                            item.get("text").and_then(|t| t.as_str())
                                        {
                                            if !text.trim().is_empty() {
                                                has_non_empty = true;
                                                break;
                                            }
                                        }
                                    } else {
                                        // Unknown type, consider it non-empty
                                        has_non_empty = true;
                                        break;
                                    }
                                } else {
                                    // No type field, might be plain text
                                    if let Some(text) = item.as_str() {
                                        if !text.trim().is_empty() {
                                            has_non_empty = true;
                                            break;
                                        }
                                    }
                                }
                            }
                            !has_non_empty
                        }
                    } else {
                        // Content is neither string nor array
                        true
                    }
                } else {
                    // No content field
                    true
                }
            };

            if is_empty {
                if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                    eprintln!("DEBUG: Removing empty content message at index {}", i);
                }
                messages.remove(i);
                // Don't increment i since we removed the element
            } else {
                i += 1;
            }
        }
    }

    /// Repair Claude tool_use/tool_result integrity in-place.
    ///
    /// Claude API invariants:
    /// - An assistant message with tool_use blocks must be immediately followed by a user
    ///   message with corresponding tool_result blocks.
    /// - Each tool_use.id must have a matching tool_result.tool_use_id in the next message.
    /// - If matching tool_result is missing, remove the tool_use block.
    pub fn repair_claude_tool_integrity(&self, messages: &mut Vec<Value>) {
        let mut i = 0usize;
        while i < messages.len() {
            let role = messages[i]
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("");

            if role != "assistant" {
                i += 1;
                continue;
            }

            // Check if this assistant message has tool_use blocks
            let content = messages[i]
                .get("content")
                .and_then(|c| c.as_array())
                .cloned();

            if content.is_none() {
                i += 1;
                continue;
            }

            let content_arr = content.unwrap();
            let mut tool_use_ids: Vec<String> = Vec::new();
            let mut has_tool_use = false;

            for item in &content_arr {
                if item.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                    if let Some(id) = item.get("id").and_then(|i| i.as_str()) {
                        if !id.is_empty() {
                            tool_use_ids.push(id.to_string());
                            has_tool_use = true;
                        }
                    }
                }
            }

            if !has_tool_use || tool_use_ids.is_empty() {
                i += 1;
                continue;
            }

            // Check if next message is user with matching tool_result blocks
            let next_idx = i + 1;
            let mut found_results: Vec<String> = Vec::new();

            if next_idx < messages.len() {
                let next_role = messages[next_idx]
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("");

                if next_role == "user" {
                    if let Some(next_content) =
                        messages[next_idx].get("content").and_then(|c| c.as_array())
                    {
                        for item in next_content {
                            if item.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                                if let Some(tool_use_id) =
                                    item.get("tool_use_id").and_then(|id| id.as_str())
                                {
                                    if tool_use_ids.contains(&tool_use_id.to_string()) {
                                        found_results.push(tool_use_id.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Remove tool_use blocks that don't have matching tool_result
            if found_results.len() != tool_use_ids.len() {
                if let Some(content_mut) = messages[i]
                    .get_mut("content")
                    .and_then(|c| c.as_array_mut())
                {
                    let mut new_content: Vec<Value> = Vec::new();
                    for item in content_mut.iter() {
                        let item_type = item.get("type").and_then(|t| t.as_str());
                        if item_type == Some("tool_use") {
                            if let Some(id) = item.get("id").and_then(|i| i.as_str()) {
                                if found_results.contains(&id.to_string()) {
                                    new_content.push(item.clone());
                                } else {
                                    if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                                        eprintln!("DEBUG: Removing tool_use '{}' without matching tool_result", id);
                                    }
                                }
                            } else {
                                new_content.push(item.clone());
                            }
                        } else {
                            new_content.push(item.clone());
                        }
                    }
                    *content_mut = new_content;
                }
            }

            i += 1;
        }
    }

    /// Repair Moonshot/OpenAI/Grok tool-call integrity in-place.
    ///
    /// Provider invariants:
    /// - An assistant message with tool_calls must be immediately followed by one tool
    ///   message per tool_call.id, in order.
    /// - Each tool message must have a non-empty tool_call_id that matches a preceding
    ///   assistant.tool_calls[*].id.
    /// - If matching tools are missing, drop those tool_calls (and remove field if empty).
    ///
    /// This handles:
    /// - OpenAI format (tool_calls array, tool role messages)
    /// - Moonshot/Kimi format (same as OpenAI, but requires 'name' field)
    /// - Grok/xAI format (same as OpenAI, with strict tool_call_id matching)
    pub fn repair_tool_call_integrity(&self, messages: &mut Vec<Value>) {
        // 0) Remove obviously invalid tool messages (no/empty tool_call_id)
        let mut i = 0usize;
        while i < messages.len() {
            let role = messages[i]
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("");
            if role == "tool" {
                let tid = messages[i]
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if tid.is_empty() {
                    if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                        eprintln!("DEBUG: Dropping tool message with empty/missing tool_call_id during integrity repair");
                    }
                    messages.remove(i);
                    continue; // stay on i (next element shifted)
                }
            }
            i += 1;
        }

        // 1) For each assistant with tool_calls, enforce immediate adjacency by
        //    moving matching tool messages to directly follow, and pruning unmatched ids.
        let mut idx = 0usize;
        while idx < messages.len() {
            let role = messages[idx]
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("");
            if role != "assistant" {
                idx += 1;
                continue;
            }

            // Extract tool_calls ids (OpenAI/Moonshot format)
            let mut ids: Vec<String> = vec![];
            if let Some(tc_arr) = messages[idx].get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tc_arr {
                    if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                        if !id.is_empty() {
                            ids.push(id.to_string());
                        }
                    }
                }
            }

            if ids.is_empty() {
                idx += 1;
                continue; // no tool_calls here
            }

            // Ensure that for each id, the very next message(s) are the corresponding tool results in order
            let mut insertion_pos = idx + 1;
            let mut kept_ids: Vec<String> = Vec::new();

            for expected_id in ids.iter() {
                // If already in place, accept and advance
                let already_ok = if insertion_pos < messages.len() {
                    let r = messages[insertion_pos]
                        .get("role")
                        .and_then(|r| r.as_str())
                        .unwrap_or("");
                    let tid = messages[insertion_pos]
                        .get("tool_call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    r == "tool" && tid == expected_id
                } else {
                    false
                };

                if already_ok {
                    kept_ids.push(expected_id.clone());
                    insertion_pos += 1;
                    continue;
                }

                // Search for a matching tool message later in the list
                let mut found_at: Option<usize> = None;
                let mut k = insertion_pos;
                while k < messages.len() {
                    let r = messages[k]
                        .get("role")
                        .and_then(|r| r.as_str())
                        .unwrap_or("");
                    if r == "tool" {
                        let tid = messages[k]
                            .get("tool_call_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if tid == expected_id {
                            found_at = Some(k);
                            break;
                        }
                    }
                    k += 1;
                }

                if let Some(found_idx) = found_at {
                    // Move the tool message to the correct position (insertion_pos)
                    let tool_msg = messages.remove(found_idx);
                    messages.insert(insertion_pos, tool_msg);
                    kept_ids.push(expected_id.clone());
                    insertion_pos += 1;
                } else {
                    // Not found → drop this id from tool_calls (will prune below)
                    if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                        eprintln!(
                            "DEBUG: Missing tool message for tool_call id '{}'; pruning from assistant.tool_calls",
                            expected_id
                        );
                    }
                }
            }

            // Rewrite tool_calls list to only include kept_ids; remove field if empty
            if let Some(tc_arr) = messages[idx].get_mut("tool_calls") {
                if let Some(arr) = tc_arr.as_array_mut() {
                    arr.retain(|tc| {
                        tc.get("id")
                            .and_then(|v| v.as_str())
                            .map(|s| kept_ids.contains(&s.to_string()))
                            .unwrap_or(false)
                    });
                }
            }
            // If tool_calls now empty, remove the field
            let empty = messages[idx]
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .map(|a| a.is_empty())
                .unwrap_or(false);
            if empty {
                if let Some(obj) = messages[idx].as_object_mut() {
                    obj.remove("tool_calls");
                }
            }

            idx = insertion_pos; // skip past the repaired block
        }

        // 2) Drop any stray tool messages that do not have a matching assistant.tool_calls anywhere before
        // Build a set of all valid tool_call ids present in assistant messages
        use std::collections::HashSet;
        let mut valid_ids: HashSet<String> = HashSet::new();
        for msg in messages.iter() {
            if msg.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                if let Some(arr) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                    for tc in arr {
                        if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                            if !id.is_empty() {
                                valid_ids.insert(id.to_string());
                            }
                        }
                    }
                }
            }
        }
        let mut j = 0usize;
        while j < messages.len() {
            let role = messages[j]
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("");
            if role == "tool" {
                let tid = messages[j]
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if tid.is_empty() || !valid_ids.contains(tid) {
                    if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                        eprintln!("DEBUG: Dropping stray tool message (no matching assistant.tool_calls) — id='{}'", tid);
                    }
                    messages.remove(j);
                    continue;
                }
            }
            j += 1;
        }
    }

    /// Debug-only: Assert Moonshot/OpenAI tool-call sequence invariants.
    /// Returns true if valid. Logs diagnostics when LLM_TOKEN_SAVER_DEBUG=1.
    pub fn assert_tool_sequence(&self, messages: &Vec<Value>) -> bool {
        let debug = std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok();
        let mut ok = true;

        let mut i = 0usize;
        while i < messages.len() {
            let role = messages[i]
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("");
            if role == "assistant" {
                // Collect ids
                let mut ids: Vec<String> = vec![];
                if let Some(arr) = messages[i].get("tool_calls").and_then(|v| v.as_array()) {
                    for (k, tc) in arr.iter().enumerate() {
                        let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        if id.is_empty() {
                            ok = false;
                            if debug {
                                eprintln!("ASSERT: assistant.tool_calls[{}] has empty id", k);
                            }
                        } else {
                            ids.push(id.to_string());
                        }
                    }
                }
                if !ids.is_empty() {
                    // Check immediate adjacency
                    let mut j = i + 1;
                    for (k, expected) in ids.iter().enumerate() {
                        if j >= messages.len() {
                            ok = false;
                            if debug {
                                eprintln!(
                                    "ASSERT: Missing tool message for id '{}' after assistant",
                                    expected
                                );
                            }
                            break;
                        }
                        let r = messages[j]
                            .get("role")
                            .and_then(|r| r.as_str())
                            .unwrap_or("");
                        let tid = messages[j]
                            .get("tool_call_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if !(r == "tool" && tid == expected) {
                            ok = false;
                            if debug {
                                eprintln!("ASSERT: Expected role=tool with tool_call_id='{}' at offset {}, found role='{}' id='{}'", expected, k, r, tid);
                            }
                        }
                        j += 1;
                    }
                    i = j; // skip validated block
                    continue;
                }
            }
            // Basic check for role=tool non-empty id
            if role == "tool" {
                let tid = messages[i]
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if tid.is_empty() {
                    ok = false;
                    if debug {
                        eprintln!("ASSERT: role=tool has empty tool_call_id");
                    }
                }
            }
            i += 1;
        }

        ok
    }

    /// Strict Kimi invariant (K1) checker.
    /// Returns Ok(()) if the sequence is valid or there is no assistant with tool_calls.
    /// Returns Err(String) with a short diagnostic otherwise.
    pub fn check_kimi_invariants(&self, messages: &Vec<Value>) -> Result<(), String> {
        // 1) Find last assistant that has tool_calls
        let mut a_idx: Option<usize> = None;
        for (idx, m) in messages.iter().enumerate() {
            if m.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                if let Some(tc) = m.get("tool_calls").and_then(|v| v.as_array()) {
                    if !tc.is_empty() {
                        a_idx = Some(idx);
                    }
                }
            }
        }
        let Some(ai) = a_idx else {
            return Ok(());
        };

        // 2) Collect ids and names
        let mut ids: Vec<String> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        let tcs = messages[ai]
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "assistant.tool_calls not an array".to_string())?;
        for (i, tc) in tcs.iter().enumerate() {
            let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
            if id.is_empty() {
                return Err(format!("assistant.tool_calls[{}] has empty id", i));
            }
            let fname = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            if fname.is_empty() {
                return Err(format!(
                    "assistant.tool_calls[{}].function.name is empty",
                    i
                ));
            }
            ids.push(id.to_string());
            names.push(fname.to_string());
        }

        // 3) Check that the next N messages are tool with matching ids (and name if present)
        for (j, expected_id) in ids.iter().enumerate() {
            let k = ai + 1 + j;
            if k >= messages.len() {
                return Err(format!(
                    "missing role=tool for id '{}' at offset {}",
                    expected_id, j
                ));
            }
            let r = messages[k]
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("");
            if r != "tool" {
                return Err(format!(
                    "expected role=tool at position {}, found '{}'",
                    k, r
                ));
            }
            let tid = messages[k]
                .get("tool_call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if tid != expected_id {
                return Err(format!(
                    "tool_call_id mismatch: expected '{}' got '{}'",
                    expected_id, tid
                ));
            }

            // name must match function.name when present
            if let Some(name) = messages[k].get("name").and_then(|v| v.as_str()) {
                if name != names[j] {
                    return Err(format!(
                        "tool name mismatch for id '{}': expected '{}' got '{}'",
                        expected_id, names[j], name
                    ));
                }
            }
            // content must be a string
            if !messages[k]
                .get("content")
                .map(|v| v.is_string())
                .unwrap_or(false)
            {
                return Err(format!(
                    "tool message content for id '{}' is not a string",
                    expected_id
                ));
            }
        }

        // 4) After the N tool messages, there must be no further role=tool messages
        let end = ai + 1 + ids.len();
        for x in end..messages.len() {
            if messages[x].get("role").and_then(|r| r.as_str()) == Some("tool") {
                return Err(format!(
                    "extra role=tool message found at position {} after tool_calls block",
                    x
                ));
            }
        }

        Ok(())
    }

    /// Production-safe fallback: drop the last assistant(tool_calls) block and its
    /// following role=tool messages so that Moonshot never receives invalid tool payloads.
    pub fn apply_kimi_fallback(&self, messages: &mut Vec<Value>) {
        // Find last assistant with tool_calls
        let mut a_idx: Option<usize> = None;
        for (idx, m) in messages.iter().enumerate() {
            if m.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                if let Some(tc) = m.get("tool_calls").and_then(|v| v.as_array()) {
                    if !tc.is_empty() {
                        a_idx = Some(idx);
                    }
                }
            }
        }
        let Some(ai) = a_idx else {
            return;
        };

        // 1) Remove tool_calls field from assistant
        if let Some(obj) = messages[ai].as_object_mut() {
            obj.remove("tool_calls");
        }

        // 2) Remove successive role=tool messages immediately after
        let mut k = ai + 1;
        while k < messages.len() {
            let role = messages[k]
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("");
            if role == "tool" {
                messages.remove(k);
            } else {
                break;
            }
        }

        // 3) Optional system memo in debug mode
        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            messages.push(json!({
                "role": "system",
                "content": "[snailer/kimi] tool_call block dropped due to integrity violation; letting model re-issue tools."
            }));
        }
    }

    /// Score messages by relevance to user query
    fn score_by_relevance(
        &self,
        messages: &[ClassifiedMessage],
        user_query: Option<&str>,
    ) -> Vec<ScoredMessage> {
        messages
            .iter()
            .map(|msg| {
                let content = msg
                    .message
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or("");

                // Base score from importance (cast through u8 first)
                let mut score = (msg.importance as u8) as f32;

                // Query-aware boost
                if let Some(query) = user_query {
                    let query_lower = query.to_lowercase();
                    let content_lower = content.to_lowercase();

                    // Count keyword matches
                    let keywords: Vec<&str> = query_lower
                        .split_whitespace()
                        .filter(|w| w.len() > 3)
                        .collect();

                    let matches = keywords
                        .iter()
                        .filter(|&&kw| content_lower.contains(kw))
                        .count();

                    score += matches as f32 * 2.0;
                }

                // Penalty for very long messages (noise)
                if content.len() > 2000 {
                    score *= 0.8;
                }

                ScoredMessage {
                    message: msg.clone(),
                    score,
                }
            })
            .collect()
    }

    /// Select top-K messages to meet target token count
    fn select_top_k(
        &self,
        mut scored: Vec<ScoredMessage>,
        target_tokens: usize,
        primacy: &[ClassifiedMessage],
        recency: &[ClassifiedMessage],
    ) -> Vec<ClassifiedMessage> {
        // Sort by score descending
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Account for primacy/recency tokens
        let primacy_tokens = self.estimate_tokens(
            &primacy
                .iter()
                .map(|cm| cm.message.clone())
                .collect::<Vec<_>>(),
        );
        let recency_tokens = self.estimate_tokens(
            &recency
                .iter()
                .map(|cm| cm.message.clone())
                .collect::<Vec<_>>(),
        );

        let available_tokens = target_tokens
            .saturating_sub(primacy_tokens)
            .saturating_sub(recency_tokens);

        // Greedily select until target reached
        let mut selected = Vec::new();
        let mut current_tokens = 0;

        for scored_msg in scored {
            let msg_tokens = scored_msg.message.estimated_tokens;
            if current_tokens + msg_tokens <= available_tokens {
                selected.push(scored_msg.message);
                current_tokens += msg_tokens;
            }

            if current_tokens >= available_tokens {
                break;
            }
        }

        // Sort by original index to preserve order
        selected.sort_by_key(|msg| msg.index);

        selected
    }

    // ============================================
    // Tier 2: Query-Aware Abstractive (UC Berkeley key finding)
    // ============================================

    /// Tier 2: Query-Aware Abstractive Compression
    ///
    /// UC Berkeley finding: "질의-인지 시 +10점 향상"
    /// Lost in Middle: Middle zone 우선 압축, Primacy/Recency 보존
    ///
    /// Expected: 30× compression, < 10% accuracy loss
    pub async fn compress_tier2_query_aware(
        &mut self,
        messages: Vec<Value>,
        user_query: &str,
        api_client: &dyn LlmClient,
    ) -> Result<Vec<Value>> {
        let start_time = Instant::now();
        let start_tokens = self.estimate_tokens(&messages);

        // 1. Partition by position
        let classified = self.classify_messages(&messages);
        let (primacy, middle, recency) = self.partition_by_position(classified);

        if middle.is_empty() {
            return Ok(messages);
        }

        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            eprintln!(
                "DEBUG Tier 2: primacy={}, middle={}, recency={}",
                primacy.len(),
                middle.len(),
                recency.len()
            );
        }

        // 2. Chunk middle zone (UC Berkeley: 128 tokens)
        let chunks = self.chunk_messages(&middle);

        // 3. Summarize each chunk WITH query (Query-Aware!)
        let target_tokens_per_chunk = (start_tokens / 30) / chunks.len().max(1);
        let mut summaries = Vec::new();

        for (i, chunk) in chunks.iter().enumerate() {
            if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                eprintln!("DEBUG: Summarizing chunk {}/{}...", i + 1, chunks.len());
                eprintln!(
                    "DEBUG: Compression using API client - this should be cheap model (minimax-m2)"
                );
            }

            let summary = self
                .summarize_query_aware(chunk, user_query, target_tokens_per_chunk, api_client)
                .await?;

            if !summary.trim().is_empty() {
                summaries.push(json!({
                    "role": "assistant",
                    "content": format!("## Summary (Chunk {})\n{}", i + 1, summary)
                }));
            }
        }

        // 4. Collect kept messages for tool preservation
        let mut kept_messages = primacy;
        // IMPORTANT: Add middle messages that are tool-related BEFORE summarizing
        // This prevents tool_use/tool_result orphans
        for msg in middle.iter() {
            if Self::is_tool_message(&msg.message) {
                kept_messages.push(msg.clone());
            }
        }
        kept_messages.extend(recency);

        // Preserve tool pairs
        let all_classified = self.classify_messages(&messages);
        self.preserve_tool_pairs(&mut kept_messages, &all_classified);

        // Sort by index
        kept_messages.sort_by_key(|m| m.index);

        // Reassemble with summaries at HEAD to avoid splitting tool pairs
        let mut result = Vec::new();
        result.extend(summaries);
        result.extend(kept_messages.into_iter().map(|cm| cm.message));

        // Repair Claude tool_use/tool_result integrity first (if Claude format detected)
        self.repair_claude_tool_integrity(&mut result);
        // Then repair tool-call integrity (Moonshot/OpenAI invariants)
        self.repair_tool_call_integrity(&mut result);

        // 5. Update statistics
        let end_tokens = self.estimate_tokens(&result);
        let _compression_ratio = start_tokens as f32 / end_tokens.max(1) as f32;
        let saved = start_tokens.saturating_sub(end_tokens);

        self.stats.tier2_count += 1;
        self.stats.total_compressions += 1;
        self.stats.tokens_saved += saved;
        self.stats.total_time_ms += start_time.elapsed().as_millis();

        Ok(result)
    }

    /// Chunk messages (UC Berkeley: 128 tokens, sentence boundary)
    fn chunk_messages(&self, messages: &[ClassifiedMessage]) -> Vec<Vec<ClassifiedMessage>> {
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_tokens = 0;

        for msg in messages {
            if current_tokens + msg.estimated_tokens > self.chunk_size && !current_chunk.is_empty()
            {
                chunks.push(current_chunk);
                current_chunk = Vec::new();
                current_tokens = 0;
            }

            current_chunk.push(msg.clone());
            current_tokens += msg.estimated_tokens;
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }

    /// Summarize messages with query awareness (UC Berkeley key finding)
    async fn summarize_query_aware(
        &self,
        messages: &[ClassifiedMessage],
        user_query: &str,
        max_tokens: usize,
        api_client: &dyn LlmClient,
    ) -> Result<String> {
        let messages_json: Vec<Value> = messages.iter().map(|cm| cm.message.clone()).collect();

        let messages_str = serde_json::to_string_pretty(&messages_json)?;

        // ✅ Query-Aware Prompt (UC Berkeley: "+10점 향상")
        let summary_prompt = json!({
            "role": "user",
            "content": format!(
                r#"Summarize the following conversation messages, focusing ONLY on information relevant to this query:

**USER QUERY**: "{}"

Instructions:
- Focus on facts, decisions, and context related to the query
- Ignore unrelated discussions
- Keep summary under {} tokens
- If nothing is relevant, return empty string

MESSAGES TO SUMMARIZE:
{}

Provide a concise, query-focused summary:"#,
                user_query,
                max_tokens,
                messages_str
            )
        });

        match api_client.send_message_simple(vec![summary_prompt]).await {
            Ok(summary) => Ok(summary),
            Err(_) => Ok(Self::local_query_fallback_summary(
                messages, user_query, max_tokens,
            )),
        }
    }

    /// Local fallback summarizer (no API): query-aware extractive bullets
    fn local_query_fallback_summary(
        messages: &[ClassifiedMessage],
        user_query: &str,
        max_tokens: usize,
    ) -> String {
        let q = user_query.to_lowercase();
        let mut kws: Vec<&str> = q.split_whitespace().filter(|w| w.len() > 2).collect();
        if kws.is_empty() {
            kws = vec![user_query];
        }

        // Collect candidate lines containing query keywords
        let mut lines: Vec<String> = Vec::new();
        for m in messages {
            // Extract textual content from message
            let mut text = String::new();
            if let Some(s) = m.message.get("content").and_then(|v| v.as_str()) {
                text.push_str(s);
            } else if let Some(arr) = m.message.get("content").and_then(|v| v.as_array()) {
                for item in arr {
                    if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                        if let Some(t) = item.get("text").and_then(|t| t.as_str()) {
                            text.push_str(t);
                            text.push('\n');
                        }
                    }
                }
            }
            for l in text.lines() {
                let ll = l.to_lowercase();
                if kws.iter().any(|k| ll.contains(k)) {
                    let trimmed = l.trim();
                    if !trimmed.is_empty() {
                        lines.push(trimmed.to_string());
                    }
                }
            }
        }

        if lines.is_empty() {
            // Fallback: take short first lines from a few messages
            for m in messages.iter().take(5) {
                if let Some(s) = m.message.get("content").and_then(|v| v.as_str()) {
                    let first = s.lines().next().unwrap_or("").trim();
                    if !first.is_empty() {
                        lines.push(first.to_string());
                    }
                }
            }
        }

        // Clamp to budget (approx 4 chars per token)
        let mut out = String::new();
        for l in lines {
            let next = format!("- {}\n", l);
            if (out.len() + next.len()) / 4 > max_tokens {
                break;
            }
            out.push_str(&next);
        }
        if out.trim().is_empty() {
            "(Context compressed)".to_string()
        } else {
            out
        }
    }

    // ============================================
    // Tier 3: Aggressive Compression (Last Resort)
    // ============================================

    /// Tier 3: Aggressive Compression (Last Resort)
    ///
    /// Strategy: Recency-only + Tool summary schema
    /// - Keep ONLY recency zone verbatim (with its tool pairs)
    /// - Summarize pre-recency with tool metadata (原文 삭제)
    /// - Head-Tail placement (Lost in the Middle)
    /// - Accept API errors for missing tool references
    ///
    /// Expected: 50-60× compression, < 20% accuracy loss
    pub async fn compress_tier3_aggressive(
        &mut self,
        messages: Vec<Value>,
        api_client: &dyn LlmClient,
    ) -> Result<Vec<Value>> {
        let start_time = Instant::now();
        let start_tokens = self.estimate_tokens(&messages);

        // 1. Identify recency zone by MESSAGE COUNT (Lost-in-the-Middle principle!)
        // Paper: "Keep last N messages where performance is best"
        // NOT token-based - this maintains U-curve position effect
        let recency_start_idx = messages.len().saturating_sub(self.recency_size);

        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            eprintln!(
                "DEBUG Tier 3: total_messages={}, start_tokens={}, recency_size={} messages",
                messages.len(),
                start_tokens,
                self.recency_size
            );
        }

        // Align boundary: avoid starting tail with a bare tool_result without its tool_use
        let mut aligned_start = recency_start_idx;
        if aligned_start < messages.len() && Self::is_tool_result_message(&messages[aligned_start])
        {
            if let Some(result_id) = Self::first_tool_result_id(&messages[aligned_start]) {
                for i in (0..aligned_start).rev() {
                    let use_ids = Self::extract_tool_use_ids(&messages[i]);
                    if use_ids.iter().any(|id| id == &result_id) {
                        aligned_start = i; // include the matching tool_use
                        break;
                    }
                }
            }
        }
        let mut recency_messages: Vec<Value> = messages[aligned_start..].to_vec();

        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            eprintln!(
                "DEBUG Tier 3: recency_start_idx={} (aligned {}), recency_count={}",
                recency_start_idx,
                aligned_start,
                recency_messages.len()
            );
        }

        // 2. Recency zone - DO NOT expand with tool pairs (Tier 3: break the chain!)
        // This is aggressive: we accept that tool_use might be in summary
        let recency_classified: Vec<ClassifiedMessage> = recency_messages
            .iter()
            .enumerate()
            .map(|(i, msg)| ClassifiedMessage {
                message: msg.clone(),
                importance: Self::classify_importance(msg),
                position: MessagePosition::Recency,
                index: aligned_start + i,
                estimated_tokens: self.estimate_tokens(&[msg.clone()]),
            })
            .collect();

        // NO preserve_tool_pairs! This is the key to aggressive compression
        // Tool pairs that span across boundary will be broken (acceptable in Tier 3)
        let mut recency_with_tools = recency_classified;

        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            eprintln!(
                "DEBUG Tier 3: keeping {} recency messages (NO tool expansion)",
                recency_with_tools.len()
            );
        }

        // 3. Pre-recency: Summarize with tool metadata (DELETE originals)
        let keep_indices: std::collections::HashSet<usize> =
            recency_with_tools.iter().map(|m| m.index).collect();

        let pre_recency: Vec<Value> = messages
            .iter()
            .enumerate()
            .filter_map(|(idx, msg)| {
                if !keep_indices.contains(&idx) {
                    Some(msg.clone())
                } else {
                    None
                }
            })
            .collect();

        let digest = if pre_recency.is_empty() {
            String::new()
        } else {
            if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                eprintln!(
                    "DEBUG Tier 3: summarizing {} pre-recency messages",
                    pre_recency.len()
                );
            }

            self.summarize_with_tool_schema(
                &pre_recency,
                500, // Tight budget for digest
                api_client,
            )
            .await?
        };

        // 4. Assemble: HEAD + PRE-RECENCY DIGEST + TAIL (recency)
        // Head-Tail placement per Lost-in-the-Middle
        let mut result = Vec::new();

        // HEAD: Current goal + digest
        if !digest.trim().is_empty() {
            result.push(json!({
                "role": "user",
                "content": format!(
                    "## Previous Context Summary\n\n{}\n\n\
                    **Note**: Tool call logs from this summary were compressed. \
                    If you need specific tool output, re-run the tool with same parameters.",
                    digest
                )
            }));
        }

        // TAIL: Recency verbatim (with tool pairs)
        recency_with_tools.sort_by_key(|m| m.index);
        result.extend(recency_with_tools.into_iter().map(|cm| cm.message));

        // Repair Claude tool_use/tool_result integrity first (if Claude format detected)
        self.repair_claude_tool_integrity(&mut result);
        // Then repair tool-call integrity
        self.repair_tool_call_integrity(&mut result);

        // 4. Update statistics
        let end_tokens = self.estimate_tokens(&result);
        let _compression_ratio = start_tokens as f32 / end_tokens.max(1) as f32;
        let saved = start_tokens.saturating_sub(end_tokens);

        self.stats.tier3_count += 1;
        self.stats.total_compressions += 1;
        self.stats.tokens_saved += saved;
        self.stats.total_time_ms += start_time.elapsed().as_millis();

        Ok(result)
    }

    // ============================================
    // Tier 4: Ultra-Aggressive (Budget Crisis Mode)
    // ============================================

    /// Tier 4: Ultra-Aggressive Compression
    ///
    /// Strategy: Absolute minimum context
    /// - One-sentence summary of ENTIRE conversation
    /// - Last user query only
    /// - No tool context (assume model has general knowledge)
    ///
    /// Expected: 100× compression, ~25% accuracy loss
    /// Use case: Budget almost depleted, need to finish task
    pub async fn compress_tier4_ultra_aggressive(
        &mut self,
        messages: Vec<Value>,
        api_client: &dyn LlmClient,
    ) -> Result<Vec<Value>> {
        let start_time = std::time::Instant::now();
        let start_tokens = self.estimate_tokens(&messages);

        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            eprintln!(
                "DEBUG Tier 4: Extreme compression - {} messages → summary + last query",
                messages.len()
            );
        }

        // 1. Extract last user query
        let last_user_query = messages
            .iter()
            .rev()
            .find(|m| m["role"] == "user")
            .cloned()
            .unwrap_or(json!({"role": "user", "content": "continue"}));

        // 2. Generate ultra-compact summary (max 2 sentences)
        let summary = self
            .generate_ultra_compact_summary(&messages, api_client)
            .await?;

        // 3. Build minimal context
        let mut result = Vec::new();

        // System prompt (if exists)
        if let Some(sys) = messages.iter().find(|m| m["role"] == "system") {
            result.push(sys.clone());
        }

        // Summary as system message
        result.push(json!({
            "role": "system",
            "content": format!("Previous context (ultra-compressed): {}", summary)
        }));

        // Last user query
        result.push(last_user_query);

        // Repair Claude tool_use/tool_result integrity first (if Claude format detected)
        self.repair_claude_tool_integrity(&mut result);
        // Then repair tool-call integrity
        self.repair_tool_call_integrity(&mut result);

        let end_tokens = self.estimate_tokens(&result);
        let _compression_ratio = if end_tokens > 0 {
            start_tokens as f32 / end_tokens as f32
        } else {
            1.0
        };

        // Update stats
        self.stats.total_compressions += 1;
        self.stats.tier4_count += 1;
        self.stats.total_time_ms += start_time.elapsed().as_millis();

        Ok(result)
    }

    /// Generate ultra-compact summary (max 2 sentences)
    async fn generate_ultra_compact_summary(
        &self,
        messages: &[Value],
        api_client: &dyn LlmClient,
    ) -> Result<String> {
        // Build condensed conversation context
        let context: String = messages
            .iter()
            .filter(|m| m["role"] != "system") // Skip system messages
            .take(20) // Last 20 messages only
            .map(|m| {
                let role = m["role"].as_str().unwrap_or("unknown");
                let content = m["content"]
                    .as_str()
                    .or_else(|| {
                        m["content"]
                            .as_array()
                            .and_then(|arr| arr.first())
                            .and_then(|v| v["text"].as_str())
                    })
                    .unwrap_or("");

                // Truncate long content
                let truncated = if content.len() > 200 {
                    format!("{}...", &content[..200])
                } else {
                    content.to_string()
                };

                format!("{}: {}", role, truncated)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let summary_prompt = vec![json!({
            "role": "user",
            "content": format!(
                "Summarize this conversation in EXACTLY 2 sentences (max 40 words). Focus on the core task and current state:\n\n{}",
                context
            )
        })];

        let summary = api_client
            .send_message_simple(summary_prompt)
            .await
            .unwrap_or_else(|_| "User is working on a coding task.".to_string());

        Ok(summary)
    }

    // ============================================
    // Tier 5: Emergency Summarization (Absolute Last Resort)
    // ============================================

    /// Tier 5: Emergency Summarization
    ///
    /// Strategy: Catastrophic compression
    /// - ONE sentence summary of entire conversation
    /// - Last user query only
    /// - No system context
    ///
    /// Expected: 200× compression, ~40% accuracy loss
    /// Use case: Budget completely exhausted, must finish current task
    pub async fn compress_tier5_emergency(
        &mut self,
        messages: Vec<Value>,
        api_client: &dyn LlmClient,
    ) -> Result<Vec<Value>> {
        let start_time = std::time::Instant::now();
        let start_tokens = self.estimate_tokens(&messages);

        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            eprintln!("DEBUG Tier 5: Emergency mode - absolute minimal context");
        }

        // 1. Last user query only
        let last_user_query = messages
            .iter()
            .rev()
            .find(|m| m["role"] == "user")
            .cloned()
            .unwrap_or(json!({"role": "user", "content": "continue"}));

        // 2. ONE sentence summary
        let one_sentence = self
            .generate_one_sentence_summary(&messages, api_client)
            .await?;

        // 3. Absolute minimum context
        let result = vec![
            json!({
                "role": "system",
                "content": format!("Context: {}", one_sentence)
            }),
            last_user_query,
        ];

        let end_tokens = self.estimate_tokens(&result);
        let _compression_ratio = if end_tokens > 0 {
            start_tokens as f32 / end_tokens as f32
        } else {
            1.0
        };

        // Update stats
        self.stats.total_compressions += 1;
        self.stats.tier5_count += 1;
        self.stats.total_time_ms += start_time.elapsed().as_millis();

        Ok(result)
    }

    /// Generate ONE sentence summary (max 20 words)
    async fn generate_one_sentence_summary(
        &self,
        messages: &[Value],
        api_client: &dyn LlmClient,
    ) -> Result<String> {
        // Extract only key info
        let task_hints: Vec<String> = messages
            .iter()
            .filter(|m| m["role"] == "user")
            .filter_map(|m| {
                let content = m["content"]
                    .as_str()
                    .or_else(|| {
                        m["content"]
                            .as_array()
                            .and_then(|arr| arr.first())
                            .and_then(|v| v["text"].as_str())
                    })
                    .unwrap_or("");

                if content.len() > 50 {
                    Some(&content[..50])
                } else if !content.is_empty() {
                    Some(content)
                } else {
                    None
                }
            })
            .map(|s| s.to_string())
            .take(5) // Last 5 user queries only
            .collect();

        let hints = task_hints.join("; ");

        let summary_prompt = vec![json!({
            "role": "user",
            "content": format!(
                "Summarize in ONE sentence (max 20 words): {}",
                if hints.is_empty() { "Coding assistant conversation" } else { &hints }
            )
        })];

        let summary = api_client
            .send_message_simple(summary_prompt)
            .await
            .unwrap_or_else(|_| "Working on a task.".to_string());

        Ok(summary)
    }

    /// Summarize with tool schema (Tier 3)
    /// Extracts tool metadata instead of preserving full logs
    async fn summarize_with_tool_schema(
        &self,
        messages: &[Value],
        max_tokens: usize,
        api_client: &dyn LlmClient,
    ) -> Result<String> {
        let messages_str = serde_json::to_string_pretty(messages)?;

        let prompt = json!({
            "role": "user",
            "content": format!(
                r#"Summarize this conversation history with focus on TOOL USAGE and KEY DECISIONS.

Use this schema for each tool:
[TOOL SUMMARY]
- tool: <name>
- input: <key parameters, 1-2 lines>
- output: <key results, 1-3 lines: numbers/status/errors>
- purpose: <why it was used>

For regular conversation:
- Focus on user goals, decisions, and conclusions
- Omit verbose explanations

Constraints:
- Maximum {} tokens
- Preserve numbers, API signatures, file paths, error codes
- Delete redundant logs

CONVERSATION:
{}

Provide structured summary:"#,
                max_tokens,
                messages_str
            )
        });

        match api_client.send_message_simple(vec![prompt]).await {
            Ok(summary) => Ok(summary),
            Err(_) => Ok(String::new()),
        }
    }

    /// Aggressive summarization (last resort)
    async fn summarize_aggressive(
        &self,
        messages: &[Value],
        max_tokens: usize,
        api_client: &dyn LlmClient,
    ) -> Result<String> {
        let messages_str = serde_json::to_string_pretty(messages)?;

        let prompt = json!({
            "role": "user",
            "content": format!(
                r#"Create an EXTREMELY concise summary of this conversation.

Constraints:
- Maximum {} tokens
- Focus ONLY on critical information
- Omit all redundant details
- Use bullet points

CONVERSATION:
{}

Provide ultra-concise summary:"#,
                max_tokens,
                messages_str
            )
        });

        match api_client.send_message_simple(vec![prompt]).await {
            Ok(summary) => Ok(summary),
            Err(_) => Ok(String::new()),
        }
    }

    /// Final hard-cap compression ensuring tokens <= budget
    ///
    /// CRITICAL: This is the LAST LINE OF DEFENSE
    /// - Uses while-loop to iteratively trim until budget satisfied
    /// - Measures ACTUAL JSON payload (not approximation)
    /// - NO reassembly after this point
    ///
    /// Strategy: Iterative trimming in order:
    /// 1. Head details (bullets → one-liners)
    /// 2. Pre-recency summary pruning (coarse → fine)
    /// 3. Tail oldest messages (keep last N tool pairs)
    /// 4. Tool result downsampling (full logs → summaries)
    /// 5. Hard truncate tail to absolute limit
    /// Deterministically pack messages to a hard token budget.
    ///
    /// Guarantees `estimate_tokens(result) <= budget` by dropping older messages first.
    /// This is a last-resort guardrail and must never fail.
    fn pack_to_budget(&self, messages: Vec<Value>, budget: usize) -> Vec<Value> {
        let minimal_stub = |mgr: &UnifiedContextManager, budget: usize| -> Vec<Value> {
            let mut msg = json!({
                "role": "user",
                "content": "Continue."
            });
            if budget > 0 && mgr.estimate_tokens(&[msg.clone()]) > budget {
                msg["content"] = json!("");
            }
            vec![msg]
        };

        if self.estimate_tokens(&messages) <= budget {
            return messages;
        }

        if messages.is_empty() {
            return minimal_stub(self, budget);
        }

        // Keep as much recency as fits.
        let mut total = 0usize;
        let mut keep_from = messages.len();
        for (i, msg) in messages.iter().enumerate().rev() {
            let msg_tokens = self.estimate_tokens(&[msg.clone()]);
            if total.saturating_add(msg_tokens) > budget {
                keep_from = i + 1;
                break;
            }
            total = total.saturating_add(msg_tokens);
        }

        if keep_from >= messages.len() {
            // If even the last message doesn't fit, replace with a minimal stub.
            return minimal_stub(self, budget);
        }

        let original = messages;
        let mut kept = original[keep_from..].to_vec();

        // If the first kept message starts with an orphan tool_result, try to reattach its tool_use.
        let first_orphan_tool_result_id = |msg: &serde_json::Value| -> Option<String> {
            let arr = msg.get("content").and_then(|c| c.as_array())?;
            for item in arr {
                if item.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
                    return item
                        .get("tool_use_id")
                        .and_then(|s| s.as_str())
                        .map(|s| s.to_string());
                }
            }
            None
        };

        if let Some(first) = kept.first() {
            if let Some(tool_id) = first_orphan_tool_result_id(first) {
                for i in (0..keep_from).rev() {
                    if let Some(arr) = original[i].get("content").and_then(|c| c.as_array()) {
                        let found = arr.iter().any(|item| {
                            item.get("type").and_then(|t| t.as_str()) == Some("tool_use")
                                && item.get("id").and_then(|s| s.as_str()) == Some(tool_id.as_str())
                        });
                        if found {
                            kept.insert(0, original[i].clone());
                            break;
                        }
                    }
                }
            }
        }

        // Repair tool adjacency and then hard-enforce the budget.
        self.repair_claude_tool_integrity(&mut kept);
        self.repair_tool_call_integrity(&mut kept);

        while kept.len() > 1 && self.estimate_tokens(&kept) > budget {
            kept.remove(0);
        }

        if self.estimate_tokens(&kept) > budget {
            return minimal_stub(self, budget);
        }

        kept
    }

    /// Public wrapper for deterministic budget enforcement.
    pub fn enforce_budget(&self, messages: Vec<Value>, budget: usize) -> Vec<Value> {
        self.pack_to_budget(messages, budget)
    }

    pub async fn compress_hard_cap(
        &self,
        messages: Vec<Value>,
        api_client: &dyn LlmClient,
    ) -> Result<Vec<Value>> {
        // SAFETY MARGIN: Model-specific (MiniMax: 85%, Grok: 90%, Others: 95%)
        let budget = (self.max_tokens as f32 * self.safety_margin) as usize;

        let start_tokens = self.estimate_tokens(&messages);

        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            eprintln!("\n=== HARD CAP START ===");
            eprintln!("  start_tokens: {}", start_tokens);
            eprintln!(
                "  budget: {} ({}% of {})",
                budget,
                (self.safety_margin * 100.0) as u8,
                self.max_tokens
            );
        }

        if start_tokens <= budget {
            return Ok(messages);
        }

        // STEP 1: Partition by Lost-in-the-Middle zones (MESSAGE COUNT, not tokens!)
        // Paper: Primacy (front 3) + Recency (last 5) have best performance
        let primacy_end = self.primacy_size.min(messages.len());
        let recency_start = messages.len().saturating_sub(self.recency_size);

        let primacy: Vec<Value> = messages[..primacy_end].to_vec();
        let middle: Vec<Value> = if primacy_end < recency_start {
            messages[primacy_end..recency_start].to_vec()
        } else {
            Vec::new()
        };
        let recency: Vec<Value> = messages[recency_start..].to_vec();

        let primacy_tokens = self.estimate_tokens(&primacy);
        let recency_tokens = self.estimate_tokens(&recency);

        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            eprintln!(
                "  primacy: {} messages, {} tokens",
                primacy.len(),
                primacy_tokens
            );
            eprintln!("  middle: {} messages", middle.len());
            eprintln!(
                "  recency: {} messages, {} tokens",
                recency.len(),
                recency_tokens
            );
        }

        // STEP 2: Summarize middle (not primacy/recency!)
        // Lost-in-the-Middle: Middle has worst performance, so we compress it
        let reserved_for_zones = primacy_tokens + recency_tokens;
        let middle_budget = budget
            .saturating_sub(reserved_for_zones)
            .saturating_sub(1000);
        let middle_budget = middle_budget.max(500);

        let mut middle_summary = String::new();
        if !middle.is_empty() {
            middle_summary = self
                .summarize_with_tool_schema(&middle, middle_budget, api_client)
                .await
                .unwrap_or_else(|_| String::from("(Summary unavailable)"));
        }

        // STEP 3: Assemble - Head-Tail placement (Lost-in-the-Middle!)
        // [Primacy] + [Middle Summary] + [Recency]
        let mut result = Vec::new();
        let primacy_count = primacy.len();

        // Add primacy (verbatim)
        result.extend(primacy.clone());

        // Add middle summary (compressed)
        if !middle_summary.trim().is_empty() {
            result.push(json!({
                "role": "user",
                "content": format!(
                    "## Middle Context Summary\n\n{}\n\n\
                    **Note**: Logs compressed. Re-run tools if needed.",
                    middle_summary
                )
            }));
        }

        // Add recency (verbatim)
        result.extend(recency.clone());

        // STEP 4: WHILE LOOP - iterative trimming until budget satisfied
        // PROTECT: Primacy + Recency (Lost-in-the-Middle zones!)
        // TRIM: Only middle summary
        let mut iterations = 0;
        let max_iterations = 10;

        while iterations < max_iterations {
            let current_tokens = self.estimate_tokens(&result);

            if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                eprintln!("  iteration {}: {} tokens", iterations, current_tokens);
            }

            if current_tokens <= budget {
                break;
            }

            iterations += 1;
            let mut trimmed = false;

            // Find middle summary (it's after primacy messages)
            let summary_idx = primacy_count;

            // Trim 1: Shorten middle summary (NEVER touch primacy/recency!)
            if summary_idx < result.len() {
                if let Some(summary_msg) = result.get_mut(summary_idx) {
                    if let Some(content) = summary_msg.get_mut("content").and_then(|c| c.as_str()) {
                        let target_len = (content.len() as f32 * 0.7) as usize; // Reduce by 30%
                        if target_len > 100 && target_len < content.len() {
                            let truncated: String = content.chars().take(target_len).collect();
                            summary_msg["content"] = json!(truncated + "...");
                            trimmed = true;
                            continue;
                        }
                    }
                }
            }

            // Trim 2: Reduce primacy if absolutely necessary (violates principle, but emergency)
            if primacy_count > 1 && result.len() > primacy_count + 1 {
                if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                    eprintln!("  WARNING: Trimming primacy (violates Lost-in-the-Middle!)");
                }
                result.remove(0);
                trimmed = true;
                continue;
            }

            // Trim 3: Minimal summary
            if summary_idx < result.len() {
                result[summary_idx]["content"] = json!("(Context compressed)");
                trimmed = true;
                continue;
            }

            if !trimmed {
                break; // Can't trim anymore
            }
        }

        // Repair Claude tool_use/tool_result integrity first (if Claude format detected)
        self.repair_claude_tool_integrity(&mut result);
        // Then repair tool-call integrity before final measurement
        self.repair_tool_call_integrity(&mut result);

        let final_tokens = self.estimate_tokens(&result);

        if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
            eprintln!("=== HARD CAP END ===");
            eprintln!("  final_tokens: {} (<= {})", final_tokens, budget);
            eprintln!("  iterations: {}", iterations);
            eprintln!(
                "  compression: {:.1}x\n",
                start_tokens as f32 / final_tokens.max(1) as f32
            );
        }

        // FINAL ASSERTION: deterministically enforce the hard cap (must never fail).
        if final_tokens > budget {
            if std::env::var("LLM_TOKEN_SAVER_DEBUG").is_ok() {
                eprintln!(
                    "WARNING: Hard cap over budget after trimming ({} > {}). Enforcing budget.",
                    final_tokens, budget
                );
            }
            result = self.pack_to_budget(result, budget);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_max_tokens() {
        let mgr = UnifiedContextManager::new("claude-sonnet-4-5");
        assert_eq!(mgr.max_tokens, 200_000);

        let mgr = UnifiedContextManager::new("gpt-4-turbo");
        assert_eq!(mgr.max_tokens, 128_000);

        let mgr = UnifiedContextManager::new("gpt-3.5-turbo");
        assert_eq!(mgr.max_tokens, 4_000);
    }

    #[test]
    fn test_estimate_tokens() {
        let mgr = UnifiedContextManager::new("claude-sonnet-4-5");

        let messages = vec![
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "assistant", "content": "Hi there!"}),
        ];

        let tokens = mgr.estimate_tokens(&messages);
        assert!(tokens > 0);
        assert!(tokens < 100);
    }

    #[test]
    fn test_enforce_budget_never_exceeds() {
        let mgr = UnifiedContextManager::new("claude-4.5");
        let mut messages = Vec::new();
        for i in 0..200 {
            messages.push(json!({
                "role": "user",
                "content": format!("message {} {}", i, "x".repeat(200)),
            }));
        }

        let budget = 200; // intentionally small vs the input
        let packed = mgr.enforce_budget(messages, budget);
        let tokens = mgr.estimate_tokens(&packed);
        assert!(
            tokens <= budget,
            "expected packed tokens <= budget, got {} > {}",
            tokens,
            budget
        );
        assert!(!packed.is_empty());
    }

    #[test]
    fn test_select_tier() {
        let mgr = UnifiedContextManager::new("gpt-3.5-turbo"); // 4K limit

        // < 85% → No compression
        assert_eq!(mgr.select_tier(3000), 0);

        // 85-95% → Tier 1
        assert_eq!(mgr.select_tier(3500), 1);

        // 95-100% → Tier 2
        assert_eq!(mgr.select_tier(3900), 2);

        // > 100% → Tier 3
        assert_eq!(mgr.select_tier(4200), 3);
    }

    #[test]
    fn test_classify_importance() {
        // User message → Critical
        let user_msg = json!({"role": "user", "content": "Fix bug"});
        assert_eq!(
            UnifiedContextManager::classify_importance(&user_msg),
            MessageImportance::Critical
        );

        // Tool call → High
        let tool_msg = json!({
            "role": "assistant",
            "tool_calls": [{"name": "read_file"}]
        });
        assert_eq!(
            UnifiedContextManager::classify_importance(&tool_msg),
            MessageImportance::High
        );

        // Planning → Medium
        let plan_msg = json!({
            "role": "assistant",
            "content": "Here's my plan to implement..."
        });
        assert_eq!(
            UnifiedContextManager::classify_importance(&plan_msg),
            MessageImportance::Medium
        );

        // Reasoning → Low
        let reason_msg = json!({
            "role": "assistant",
            "content": "Let me think..."
        });
        assert_eq!(
            UnifiedContextManager::classify_importance(&reason_msg),
            MessageImportance::Low
        );
    }

    #[test]
    fn test_partition_by_position() {
        let mgr = UnifiedContextManager::new("claude-sonnet-4-5");

        let messages = vec![
            json!({"role": "user", "content": "msg 0"}), // Primacy
            json!({"role": "assistant", "content": "msg 1"}), // Primacy
            json!({"role": "user", "content": "msg 2"}), // Primacy
            json!({"role": "assistant", "content": "msg 3"}), // Middle
            json!({"role": "user", "content": "msg 4"}), // Middle
            json!({"role": "assistant", "content": "msg 5"}), // Recency
            json!({"role": "user", "content": "msg 6"}), // Recency
            json!({"role": "assistant", "content": "msg 7"}), // Recency
            json!({"role": "user", "content": "msg 8"}), // Recency
            json!({"role": "assistant", "content": "msg 9"}), // Recency
        ];

        let classified = mgr.classify_messages(&messages);
        let (primacy, middle, recency) = mgr.partition_by_position(classified);

        assert_eq!(primacy.len(), 3); // primacy_size = 3
        assert_eq!(middle.len(), 2); // 10 - 3 - 5 = 2
        assert_eq!(recency.len(), 5); // recency_size = 5
    }
}
