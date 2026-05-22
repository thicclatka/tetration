//! Shared reduction kinds and single-pass accumulators.

use crate::query::types::{Operation, OperationPreviewFields};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

impl From<&Operation> for ReductionKind {
    fn from(op: &Operation) -> Self {
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
        }
    }
}

/// Online mean / variance (population, `ddof = 0`).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct WelfordAccum {
    count_f: f64,
    mean: f64,
    m2: f64,
}

impl WelfordAccum {
    pub fn push(&mut self, x: f64) {
        self.count_f += 1.0;
        let delta = x - self.mean;
        self.mean += delta / self.count_f;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Population variance (`ddof = 0`); `0.0` for a single sample.
    #[must_use]
    pub fn population_variance(&self) -> f64 {
        if self.count_f <= 1.0 {
            0.0
        } else {
            self.m2 / self.count_f
        }
    }

    #[must_use]
    pub fn population_std(&self) -> f64 {
        self.population_variance().sqrt()
    }
}

/// Single-pass accumulator for one scalar or partial-reduction cell.
#[derive(Debug, Clone)]
pub(crate) struct ValueAccum {
    count: usize,
    sum: f64,
    welford: WelfordAccum,
    product: f64,
    norm_l1: f64,
    norm_l2_sq: f64,
    all_finite: bool,
    any_nan: bool,
    min: f64,
    max: f64,
    have_min_max: bool,
}

impl Default for ValueAccum {
    fn default() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            welford: WelfordAccum::default(),
            product: 1.0,
            norm_l1: 0.0,
            norm_l2_sq: 0.0,
            all_finite: true,
            any_nan: false,
            min: 0.0,
            max: 0.0,
            have_min_max: false,
        }
    }
}

impl ValueAccum {
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn push(&mut self, v: f32) {
        self.count += 1;
        let x = f64::from(v);
        self.sum += x;
        self.welford.push(x);
        self.product *= x;
        self.norm_l1 += x.abs();
        self.norm_l2_sq += x * x;
        self.all_finite &= v.is_finite();
        self.any_nan |= v.is_nan();
        if self.have_min_max {
            self.min = self.min.min(x);
            self.max = self.max.max(x);
        } else {
            self.min = x;
            self.max = x;
            self.have_min_max = true;
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn finish_f64(&self, kind: ReductionKind) -> f64 {
        match kind {
            ReductionKind::Sum => self.sum,
            ReductionKind::Mean => self.welford.mean,
            ReductionKind::Min => self.min,
            ReductionKind::Max => self.max,
            ReductionKind::Count => self.count as f64,
            ReductionKind::Var => self.welford.population_variance(),
            ReductionKind::Std => self.welford.population_std(),
            ReductionKind::Product => self.product,
            ReductionKind::NormL1 => self.norm_l1,
            ReductionKind::NormL2 => self.norm_l2_sq.sqrt(),
            ReductionKind::AllFinite => f64::from(u8::from(self.all_finite)),
            ReductionKind::AnyNan => f64::from(u8::from(self.any_nan)),
        }
    }

    #[must_use]
    pub fn finish_bool(&self, kind: ReductionKind) -> bool {
        match kind {
            ReductionKind::AllFinite => self.all_finite,
            ReductionKind::AnyNan => self.any_nan,
            _ => self.finish_f64(kind) > 0.5,
        }
    }

    pub fn finish_scalar(self, kind: ReductionKind) -> ScalarReductionResult {
        let mut sum_scalar = None;
        let mut mean_scalar = None;
        let mut min_scalar = None;
        let mut max_scalar = None;
        let mut var_scalar = None;
        let mut std_scalar = None;
        let mut product_scalar = None;
        let mut norm_l1_scalar = None;
        let mut norm_l2_scalar = None;
        let mut all_finite_scalar = None;
        let mut any_nan_scalar = None;
        match kind {
            ReductionKind::Sum => sum_scalar = Some(self.sum),
            ReductionKind::Mean => mean_scalar = Some(self.welford.mean),
            ReductionKind::Min => min_scalar = Some(self.min),
            ReductionKind::Max => max_scalar = Some(self.max),
            ReductionKind::Count => {}
            ReductionKind::Var => var_scalar = Some(self.welford.population_variance()),
            ReductionKind::Std => std_scalar = Some(self.welford.population_std()),
            ReductionKind::Product => product_scalar = Some(self.product),
            ReductionKind::NormL1 => norm_l1_scalar = Some(self.norm_l1),
            ReductionKind::NormL2 => norm_l2_scalar = Some(self.norm_l2_sq.sqrt()),
            ReductionKind::AllFinite => all_finite_scalar = Some(self.all_finite),
            ReductionKind::AnyNan => any_nan_scalar = Some(self.any_nan),
        }
        ScalarReductionResult {
            element_count: self.count,
            sum_scalar,
            mean_scalar,
            min_scalar,
            max_scalar,
            var_scalar,
            std_scalar,
            product_scalar,
            norm_l1_scalar,
            norm_l2_scalar,
            all_finite_scalar,
            any_nan_scalar,
        }
    }
}

/// Scalar reduction outputs (one field set per kind).
#[derive(Debug, Clone)]
pub(crate) struct ScalarReductionResult {
    pub element_count: usize,
    pub sum_scalar: Option<f64>,
    pub mean_scalar: Option<f64>,
    pub min_scalar: Option<f64>,
    pub max_scalar: Option<f64>,
    pub var_scalar: Option<f64>,
    pub std_scalar: Option<f64>,
    pub product_scalar: Option<f64>,
    pub norm_l1_scalar: Option<f64>,
    pub norm_l2_scalar: Option<f64>,
    pub all_finite_scalar: Option<bool>,
    pub any_nan_scalar: Option<bool>,
}

impl From<ScalarReductionResult> for OperationPreviewFields {
    fn from(r: ScalarReductionResult) -> Self {
        Self {
            element_count: Some(r.element_count),
            sum: r.sum_scalar,
            mean: r.mean_scalar,
            min: r.min_scalar,
            max: r.max_scalar,
            var: r.var_scalar,
            std: r.std_scalar,
            product: r.product_scalar,
            norm_l1: r.norm_l1_scalar,
            norm_l2: r.norm_l2_scalar,
            all_finite: r.all_finite_scalar,
            any_nan: r.any_nan_scalar,
            ..Self::default()
        }
    }
}
