//! Shared reduction kinds and single-pass scalar accumulators.

use crate::query::types::{Operation, OperationPreviewFields};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReductionKind {
    Sum,
    Mean,
    Min,
    Max,
    Count,
}

impl From<&Operation> for ReductionKind {
    fn from(op: &Operation) -> Self {
        match op {
            Operation::Sum { .. } => Self::Sum,
            Operation::Mean { .. } => Self::Mean,
            Operation::Min { .. } => Self::Min,
            Operation::Max { .. } => Self::Max,
            Operation::Count { .. } => Self::Count,
        }
    }
}

/// Scalar reduction outputs (one field set per kind).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScalarReductionResult {
    pub element_count: usize,
    pub sum_scalar: Option<f64>,
    pub mean_scalar: Option<f64>,
    pub min_scalar: Option<f64>,
    pub max_scalar: Option<f64>,
}

#[derive(Debug, Default)]
pub(crate) struct ScalarAccum {
    count: usize,
    sum: f64,
    mean: f64,
    mean_k: f64,
    min: f64,
    max: f64,
    have_min_max: bool,
}

impl ScalarAccum {
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn push(&mut self, v: f32) {
        self.count += 1;
        let x = f64::from(v);
        self.sum += x;
        self.mean_k += 1.0;
        self.mean += (x - self.mean) / self.mean_k;
        if self.have_min_max {
            self.min = self.min.min(x);
            self.max = self.max.max(x);
        } else {
            self.min = x;
            self.max = x;
            self.have_min_max = true;
        }
    }

    pub fn finish(self, kind: ReductionKind) -> ScalarReductionResult {
        let mut sum_scalar = None;
        let mut mean_scalar = None;
        let mut min_scalar = None;
        let mut max_scalar = None;
        match kind {
            ReductionKind::Sum => sum_scalar = Some(self.sum),
            ReductionKind::Mean => mean_scalar = Some(self.mean),
            ReductionKind::Min => min_scalar = Some(self.min),
            ReductionKind::Max => max_scalar = Some(self.max),
            ReductionKind::Count => {}
        }
        ScalarReductionResult {
            element_count: self.count,
            sum_scalar,
            mean_scalar,
            min_scalar,
            max_scalar,
        }
    }
}

impl From<ScalarReductionResult> for OperationPreviewFields {
    fn from(r: ScalarReductionResult) -> Self {
        Self {
            element_count: Some(r.element_count),
            sum: r.sum_scalar,
            mean: r.mean_scalar,
            min: r.min_scalar,
            max: r.max_scalar,
            ..Self::default()
        }
    }
}
