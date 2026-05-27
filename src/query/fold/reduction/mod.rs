//! Shared reduction kinds and single-pass accumulators.

use crate::query::types::Operation;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ReductionKind {
    Sum,
    Mean,
    Min,
    Max,
    Count,
    Var,
    Std,
    Product,
    NormL1,
    NormL2,
    AllFinite,
    AnyNan,
    NanCount,
    InfCount,
    NullCount { fill: f64 },
    ArgMin,
    ArgMax,
}

impl From<&Operation> for ReductionKind {
    fn from(op: &Operation) -> Self {
        debug_assert!(!op.requires_materialize());
        match op {
            Operation::Sum { .. } => Self::Sum,
            Operation::Mean { .. } => Self::Mean,
            Operation::Min { .. } => Self::Min,
            Operation::Max { .. } => Self::Max,
            Operation::Count { .. } => Self::Count,
            Operation::Var { .. } => Self::Var,
            Operation::Std { .. } => Self::Std,
            Operation::Product { .. } => Self::Product,
            Operation::NormL1 { .. } => Self::NormL1,
            Operation::NormL2 { .. } => Self::NormL2,
            Operation::AllFinite { .. } => Self::AllFinite,
            Operation::AnyNan { .. } => Self::AnyNan,
            Operation::NanCount { .. } => Self::NanCount,
            Operation::InfCount { .. } => Self::InfCount,
            Operation::NullCount { fill: Some(f), .. } => Self::NullCount { fill: *f },
            Operation::NullCount { fill: None, .. } => {
                unreachable!("null_count fill must be resolved before fold")
            }
            Operation::ArgMin { .. } => Self::ArgMin,
            Operation::ArgMax { .. } => Self::ArgMax,
            Operation::Median { .. }
            | Operation::Quantile { .. }
            | Operation::Histogram { .. }
            | Operation::Covariance { .. }
            | Operation::Correlation { .. } => {
                unreachable!("tier-C stats use materialize-required execution path")
            }
        }
    }
}

mod scalar;
mod value_accum;
mod welford;

#[allow(unused_imports)]
// `ScalarReductionResult` / `WelfordAccum` mainly used under `#[cfg(test)]` or GPU features
pub(crate) use scalar::{ArgIndexAccum, ScalarReductionResult};
pub(crate) use value_accum::ValueAccum;
pub(crate) use welford::WelfordAccum;
