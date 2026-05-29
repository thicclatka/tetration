//! Div-by-zero (and similar) warnings recorded during transform pass 2.

/// Cap of logical indices listed in the execution response (total count is unbounded).
pub(crate) const MAX_LISTED_DIV_BY_ZERO_INDICES: usize = 256;

/// Warnings emitted while applying a transform (pass 2).
#[derive(Debug, Clone, Default)]
pub(crate) struct TransformWarnings {
    pub div_by_zero_indices: Vec<u64>,
    pub div_by_zero_count: u64,
}

impl TransformWarnings {
    pub(crate) fn record_div_by_zero(&mut self, logical_index: u64) {
        self.div_by_zero_count += 1;
        if self.div_by_zero_indices.len() < MAX_LISTED_DIV_BY_ZERO_INDICES {
            self.div_by_zero_indices.push(logical_index);
        }
    }
}
