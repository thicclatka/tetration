//! Shared ASCII helpers for CLI tables (history, info, …).

/// Truncate `s` to at most `max` bytes (UTF-8 safe) with an ellipsis suffix.
#[must_use]
pub fn truncate_field(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_owned();
    }
    let mut end = max.saturating_sub(1);
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

/// Case-insensitive substring match (ASCII lowercasing).
#[must_use]
pub fn contains_ascii_case_insensitive(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}
