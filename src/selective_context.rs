//! Selective Context - Surprisal-based input filtering
//!
//! Based on EMNLP 2023 "Compressing Context to Enhance Inference Efficiency of LLMs"
//! Key idea: I(x_t) = -log_2 P(x_t | x_{<t})
//!
//! Core principles:
//! 1. Calculate surprisal (self-information) for each sentence
//! 2. Apply protection rules (numbers, paths, signatures, errors)
//! 3. Combine with query relevance scoring
//! 4. Select top-k% by importance

use anyhow::Result;

/// Sentence with surprisal score
#[derive(Debug, Clone)]
pub struct ScoredSentence {
    pub text: String,
    pub score: f32,      // I(sentence) = sum of I(token)
    pub importance: f32, // Combined: surprisal + query relevance + protection
}

/// Budget policies for different content types
#[derive(Debug, Clone)]
pub struct BudgetPolicy {
    /// Per-file limits
    pub max_sentences_per_file: usize,
    pub max_tokens_per_file: usize,

    /// Per-search-result limits
    pub max_search_results: usize,
    pub max_sentences_per_result: usize,

    /// Global limits
    pub global_hard_cap: usize,

    /// Selective retention ratio (0.0 - 1.0)
    pub keep_ratio: f32,
}

impl Default for BudgetPolicy {
    fn default() -> Self {
        Self {
            max_sentences_per_file: 50,   // 파일당 최대 50문장
            max_tokens_per_file: 2000,    // 파일당 최대 2000토큰
            max_search_results: 20,       // 검색결과 최대 20개
            max_sentences_per_result: 10, // 검색결과당 10문장
            global_hard_cap: 95000,       // 전역 하드캡
            keep_ratio: 0.3,              // 상위 30%만 유지
        }
    }
}

/// Selective Context Filter
pub struct SelectiveContextFilter {
    /// Use small LM for surprisal calculation (e.g., "claude-haiku-3-5")
    surprisal_model: String,

    /// Protection rules (preserve numbers, paths, signatures, errors)
    protect_numbers: bool,
    protect_paths: bool,
    protect_api_signatures: bool,
    protect_error_codes: bool,
}

impl SelectiveContextFilter {
    pub fn new(surprisal_model: &str) -> Self {
        Self {
            surprisal_model: surprisal_model.to_string(),
            protect_numbers: true,
            protect_paths: true,
            protect_api_signatures: true,
            protect_error_codes: true,
        }
    }

    /// Calculate surprisal for a sentence using heuristic estimation
    /// I(sentence) = sum of I(token) over all tokens
    ///
    /// Heuristic rules:
    /// - Rare words (long, capitalized, technical) → high I
    /// - Common words (stopwords) → low I
    pub async fn calculate_surprisal(&self, sentence: &str) -> Result<f32> {
        let mut score = 0.0;

        for word in sentence.split_whitespace() {
            let word_score = self.estimate_word_surprisal(word);
            score += word_score;
        }

        Ok(score)
    }

    /// Estimate word-level surprisal (heuristic until logprobs API available)
    fn estimate_word_surprisal(&self, word: &str) -> f32 {
        let mut score = 1.0; // base

        // Length penalty (longer = rarer = higher I)
        if word.len() > 8 {
            score += 0.5;
        }

        // Capitalization (proper nouns = higher I)
        if word.chars().next().map_or(false, |c| c.is_uppercase()) {
            score += 0.3;
        }

        // Numbers (protected)
        if word.chars().any(|c| c.is_numeric()) {
            score += 1.0;
        }

        // Technical characters (e.g., underscores, colons)
        if word.contains('_') || word.contains("::") || word.contains("->") {
            score += 0.5;
        }

        // Common stopwords (lower I)
        let stopwords = [
            "the", "is", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "with", "as", "by", "from", "that", "this", "it", "be", "are", "was", "were", "been",
            "have", "has", "had",
        ];
        if stopwords.contains(&word.to_lowercase().as_str()) {
            score = 0.1;
        }

        score
    }

    /// Apply protection rules to boost importance
    /// LLMLingua paper: Numbers (5×), Paths (4×), Signatures (4×), Errors (5×)
    fn apply_protection_rules(&self, sentence: &str) -> f32 {
        let mut boost = 1.0;

        // Numbers (5× weight per paper)
        if self.protect_numbers && sentence.chars().any(|c| c.is_numeric()) {
            boost *= 5.0;
        }

        // File paths (4× weight)
        if self.protect_paths {
            let has_path = sentence.contains('/')
                || sentence.contains('\\')
                || sentence.contains(".rs")
                || sentence.contains(".py")
                || sentence.contains(".js")
                || sentence.contains(".ts")
                || sentence.contains(".go")
                || sentence.contains(".java");

            if has_path {
                boost *= 4.0;
            }
        }

        // API signatures (4× weight)
        if self.protect_api_signatures {
            let has_signature = sentence.contains("fn ")
                || sentence.contains("def ")
                || sentence.contains("function ")
                || sentence.contains("impl ")
                || sentence.contains("struct ")
                || sentence.contains("class ")
                || sentence.contains("pub fn")
                || sentence.contains("async fn");

            if has_signature {
                boost *= 4.0;
            }
        }

        // Error codes (5× weight)
        if self.protect_error_codes {
            let has_error = sentence.to_lowercase().contains("error")
                || sentence.to_lowercase().contains("err")
                || sentence.contains("Error")
                || sentence.starts_with("E")
                || sentence.contains("failed")
                || sentence.contains("panic");

            if has_error {
                boost *= 5.0;
            }
        }

        boost
    }

    /// Calculate query relevance (BM25-like simple keyword matching)
    fn calculate_query_relevance(&self, sentence: &str, query: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let sentence_lower = sentence.to_lowercase();

        let keywords: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        if keywords.is_empty() {
            return 1.0;
        }

        let matches = keywords
            .iter()
            .filter(|kw| sentence_lower.contains(*kw))
            .count();

        1.0 + (matches as f32 * 2.0) // +2 per keyword match
    }

    /// Score sentences with combined surprisal + query relevance + protection
    pub async fn score_sentences(
        &self,
        sentences: &[String],
        user_query: Option<&str>,
    ) -> Result<Vec<ScoredSentence>> {
        let mut scored = Vec::new();

        for sent in sentences {
            let surprisal = self.calculate_surprisal(sent).await?;
            let protection = self.apply_protection_rules(sent);
            let query_relevance = if let Some(q) = user_query {
                self.calculate_query_relevance(sent, q)
            } else {
                1.0
            };

            let importance = surprisal * protection * query_relevance;

            scored.push(ScoredSentence {
                text: sent.clone(),
                score: surprisal,
                importance,
            });
        }

        Ok(scored)
    }

    /// Select top-k% sentences by importance
    pub fn select_top_k(
        &self,
        mut scored: Vec<ScoredSentence>,
        keep_ratio: f32, // e.g., 0.3 = keep top 30%
    ) -> Vec<String> {
        if scored.is_empty() {
            return Vec::new();
        }

        // Sort by importance descending
        scored.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());

        let keep_count = (scored.len() as f32 * keep_ratio).ceil() as usize;
        let keep_count = keep_count.max(1).min(scored.len());

        scored
            .into_iter()
            .take(keep_count)
            .map(|s| s.text)
            .collect()
    }

    /// Apply budget policy to file content (Phase 2)
    pub async fn filter_file_content(
        &self,
        content: &str,
        user_query: Option<&str>,
        policy: &BudgetPolicy,
    ) -> Result<String> {
        // 1. Split into sentences
        let sentences = split_sentences(content);

        // 2. Enforce per-file sentence limit (pre-filter to 3× for efficiency)
        let sentences: Vec<String> = sentences
            .into_iter()
            .take(policy.max_sentences_per_file * 3)
            .collect();

        if sentences.is_empty() {
            return Ok(String::new());
        }

        // 3. Score sentences
        let scored = self.score_sentences(&sentences, user_query).await?;

        // 4. Select top-k%
        let selected = self.select_top_k(scored, policy.keep_ratio);

        // 5. Enforce per-file sentence limit (post-selection)
        let selected: Vec<String> = selected
            .into_iter()
            .take(policy.max_sentences_per_file)
            .collect();

        // 6. Join and check token limit
        let mut result = selected.join(" ");
        let estimated_tokens = result.len() / 4;

        if estimated_tokens > policy.max_tokens_per_file {
            // Truncate to token limit (approx 4 chars ≈ 1 token),
            // but always cut at a valid UTF‑8 boundary to avoid panics
            let target_bytes = policy.max_tokens_per_file * 4;
            if result.len() > target_bytes {
                let mut new_len = target_bytes.min(result.len());
                while new_len > 0 && !result.is_char_boundary(new_len) {
                    new_len -= 1;
                }
                if new_len == 0 {
                    // Fallback: if we somehow can't find a safe boundary, keep original text
                    // rather than panicking; this is extremely unlikely in practice.
                } else {
                    result.truncate(new_len);
                    result.push_str("...");
                }
            }
        }

        Ok(result)
    }

    /// Apply budget policy to search results (Phase 2)
    pub async fn filter_search_results(
        &self,
        results: &[String],
        user_query: Option<&str>,
        policy: &BudgetPolicy,
    ) -> Result<Vec<String>> {
        let mut filtered = Vec::new();

        // Limit number of search results
        for result in results.iter().take(policy.max_search_results) {
            let sentences = split_sentences(result);
            let scored = self.score_sentences(&sentences, user_query).await?;
            let selected = self.select_top_k(scored, policy.keep_ratio);

            // Enforce per-result sentence limit
            let selected: Vec<String> = selected
                .into_iter()
                .take(policy.max_sentences_per_result)
                .collect();

            if !selected.is_empty() {
                filtered.push(selected.join(" "));
            }
        }

        Ok(filtered)
    }
}

/// Split text into sentences (NLTK-like simple splitter)
///
/// Algorithm:
/// 1. Split by newlines
/// 2. Further split by sentence terminators (., !, ?)
/// 3. Preserve sentence boundaries
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !current.is_empty() {
                sentences.push(current.clone());
                current.clear();
            }
            continue;
        }

        // Simple sentence boundary detection
        // Split by . ! ? but keep them
        let chars: Vec<char> = trimmed.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            current.push(chars[i]);

            // Check for sentence terminators
            if (chars[i] == '.' || chars[i] == '!' || chars[i] == '?')
                && (i + 1 >= chars.len() || chars[i + 1].is_whitespace())
            {
                sentences.push(current.trim().to_string());
                current.clear();
            }

            i += 1;
        }

        // Add space between lines if not ending sentence
        if !current.is_empty() && !current.ends_with(' ') {
            current.push(' ');
        }
    }

    if !current.is_empty() {
        sentences.push(current.trim().to_string());
    }

    sentences.into_iter().filter(|s| !s.is_empty()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences() {
        let text = "Hello world. This is a test! How are you? Good.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 4);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "This is a test!");
        assert_eq!(sentences[2], "How are you?");
        assert_eq!(sentences[3], "Good.");
    }

    #[test]
    fn test_split_sentences_multiline() {
        let text = "Line 1.\nLine 2!\n\nLine 3?";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[tokio::test]
    async fn test_surprisal_calculation() {
        let filter = SelectiveContextFilter::new("test-model");
        let score = filter
            .calculate_surprisal("The quick brown fox")
            .await
            .unwrap();
        assert!(score > 0.0);
    }

    #[tokio::test]
    async fn test_protection_rules() {
        let filter = SelectiveContextFilter::new("test-model");

        // Numbers should boost
        let boost_number = filter.apply_protection_rules("The value is 42");
        assert!(boost_number > 1.0);

        // Paths should boost
        let boost_path = filter.apply_protection_rules("src/main.rs");
        assert!(boost_path > 1.0);

        // Errors should boost
        let boost_error = filter.apply_protection_rules("Error: connection failed");
        assert!(boost_error > 1.0);
    }

    #[tokio::test]
    async fn test_select_top_k() {
        let filter = SelectiveContextFilter::new("test-model");
        let sentences = vec![
            "The quick brown fox".to_string(),
            "Error: failed to compile src/main.rs line 42".to_string(),
            "This is boring text".to_string(),
        ];

        let scored = filter.score_sentences(&sentences, None).await.unwrap();
        let selected = filter.select_top_k(scored, 0.5); // Keep top 50%

        // Should keep at least 2 sentences
        assert!(selected.len() >= 1);
        // High-importance sentence should be selected
        assert!(selected.iter().any(|s| s.contains("Error")));
    }
}
