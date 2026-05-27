//! Shared reduction kinds and single-pass accumulators.

use crate::query::fold::variance_simd;
use crate::query::types::{Operation, OperationPreviewFields};

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

    /// Merge another online accumulator (Chan parallel algorithm).
    pub fn merge_from(&mut self, other: &Self) {
        if other.count_f == 0.0 {
            return;
        }
        if self.count_f == 0.0 {
            *self = *other;
            return;
        }
        let n_a = self.count_f;
        let n_b = other.count_f;
        let n = n_a + n_b;
        let delta = other.mean - self.mean;
        self.mean += delta * (n_b / n);
        self.m2 += other.m2 + delta * delta * n_a * n_b / n;
        self.count_f = n;
    }

    /// Merge population variance stats from sum and sum-of-squares over `count` values.
    pub(crate) fn merge_sum_sumsq(&mut self, count: f64, sum: f64, sumsq: f64) {
        if count == 0.0 {
            return;
        }
        let mean = sum / count;
        let m2 = sumsq - count * mean * mean;
        self.merge_from(&Self {
            count_f: count,
            mean,
            m2,
        });
    }

    /// Merge population variance stats for a contiguous `f32` slice (slab / chunk bulk path).
    #[allow(dead_code)]
    pub fn merge_f32_slice(&mut self, vals: &[f32]) {
        if vals.is_empty() {
            return;
        }
        let (slab_sum, slab_sumsq) = variance_simd::f32_sum_sumsq(vals);
        self.merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
    }

    /// Like [`Self::merge_f32_slice`] for an `f64` slice.
    #[allow(dead_code)]
    pub fn merge_f64_slice(&mut self, vals: &[f64]) {
        if vals.is_empty() {
            return;
        }
        let (slab_sum, slab_sumsq) = variance_simd::f64_sum_sumsq(vals);
        self.merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
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
    /// NaN / inf / null-match tally for [`ReductionKind::NanCount`], [`ReductionKind::InfCount`], and [`ReductionKind::NullCount`].
    match_count: usize,
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
            match_count: 0,
        }
    }
}

impl ValueAccum {
    fn values_equal_fill(v: f64, fill: f64) -> bool {
        if fill.is_nan() {
            v.is_nan()
        } else {
            v.to_bits() == fill.to_bits()
        }
    }

    fn merge_slab_min_max(&mut self, slab_min: f64, slab_max: f64) {
        if self.have_min_max {
            self.min = self.min.min(slab_min);
            self.max = self.max.max(slab_max);
        } else {
            self.min = slab_min;
            self.max = slab_max;
            self.have_min_max = true;
        }
    }

    fn push_null_count_values(
        &mut self,
        len: usize,
        values: impl IntoIterator<Item = f64>,
        fill: f64,
    ) {
        self.count += len;
        for v in values {
            if Self::values_equal_fill(v, fill) {
                self.match_count += 1;
            }
        }
    }

    fn push_match_count_slice(&mut self, len: usize, matches: usize) {
        self.count += len;
        self.match_count += matches;
    }

    pub(crate) fn push_nan_f64(&mut self, v: f64) {
        self.count += 1;
        if v.is_nan() {
            self.match_count += 1;
        }
    }

    pub(crate) fn push_null_f64(&mut self, v: f64, fill: f64) {
        self.count += 1;
        if Self::values_equal_fill(v, fill) {
            self.match_count += 1;
        }
    }

    pub(crate) fn push_inf_f64(&mut self, v: f64) {
        self.count += 1;
        if v.is_infinite() {
            self.match_count += 1;
        }
    }
}

impl ValueAccum {
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn push(&mut self, v: f32) {
        self.push_f64(f64::from(v));
    }

    pub fn push_f64(&mut self, v: f64) {
        self.count += 1;
        self.sum += v;
        self.welford.push(v);
        self.product *= v;
        self.norm_l1 += v.abs();
        self.norm_l2_sq += v * v;
        self.all_finite &= v.is_finite();
        self.any_nan |= v.is_nan();
        if self.have_min_max {
            self.min = self.min.min(v);
            self.max = self.max.max(v);
        } else {
            self.min = v;
            self.max = v;
            self.have_min_max = true;
        }
    }

    /// Accumulate every little-endian `f32` in `raw` for a scalar fold (no per-element callbacks).
    pub fn push_f32_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 4, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[f32] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::f32_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::f32_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::f32_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::Product => {
                self.count += vals.len();
                for &v in vals {
                    self.product *= f64::from(v);
                }
            }
            ReductionKind::NormL1 => {
                self.count += vals.len();
                for &v in vals {
                    self.norm_l1 += f64::from(v).abs();
                }
            }
            ReductionKind::NormL2 => {
                self.count += vals.len();
                for &v in vals {
                    let vd = f64::from(v);
                    self.norm_l2_sq += vd * vd;
                }
            }
            ReductionKind::AllFinite => {
                self.count += vals.len();
                if !self.all_finite {
                    return;
                }
                for &v in vals {
                    if !v.is_finite() {
                        self.all_finite = false;
                        return;
                    }
                }
            }
            ReductionKind::AnyNan => {
                if self.any_nan {
                    self.count += vals.len();
                    return;
                }
                for &v in vals {
                    self.count += 1;
                    if v.is_nan() {
                        self.any_nan = true;
                        return;
                    }
                }
            }
            ReductionKind::NanCount => {
                self.push_match_count_slice(vals.len(), vals.iter().filter(|v| v.is_nan()).count());
            }
            ReductionKind::InfCount => self.push_match_count_slice(
                vals.len(),
                vals.iter().filter(|v| v.is_infinite()).count(),
            ),
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| f64::from(v)), fill);
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Like [`Self::push_f32_le_bytes`] but promotes each value to `f64` first.
    pub fn push_f64_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 8, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[f64] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::f64_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::f64_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                for &v in vals {
                    if self.have_min_max {
                        self.min = self.min.min(v);
                        self.max = self.max.max(v);
                    } else {
                        self.min = v;
                        self.max = v;
                        self.have_min_max = true;
                    }
                }
            }
            ReductionKind::Product => {
                self.count += vals.len();
                for &v in vals {
                    self.product *= v;
                }
            }
            ReductionKind::NormL1 => {
                self.count += vals.len();
                for &v in vals {
                    self.norm_l1 += v.abs();
                }
            }
            ReductionKind::NormL2 => {
                self.count += vals.len();
                for &v in vals {
                    self.norm_l2_sq += v * v;
                }
            }
            ReductionKind::AllFinite => {
                self.count += vals.len();
                if !self.all_finite {
                    return;
                }
                for &v in vals {
                    if !v.is_finite() {
                        self.all_finite = false;
                        return;
                    }
                }
            }
            ReductionKind::AnyNan => {
                if self.any_nan {
                    self.count += vals.len();
                    return;
                }
                for &v in vals {
                    self.count += 1;
                    if v.is_nan() {
                        self.any_nan = true;
                        return;
                    }
                }
            }
            ReductionKind::NanCount => {
                self.push_match_count_slice(vals.len(), vals.iter().filter(|v| v.is_nan()).count());
            }
            ReductionKind::InfCount => self.push_match_count_slice(
                vals.len(),
                vals.iter().filter(|v| v.is_infinite()).count(),
            ),
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().copied(), fill);
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `i32` LE slab fold (values promoted to `f64`).
    pub fn push_i32_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 4, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[i32] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count | ReductionKind::NanCount | ReductionKind::InfCount => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::i32_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::i32_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::i32_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| f64::from(v)), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan => {
                for &v in vals {
                    self.push_f64(f64::from(v));
                }
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `u8` LE slab fold (values promoted to `f64`).
    pub fn push_u8_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        if raw.is_empty() {
            return;
        }
        let vals = raw;
        match kind {
            ReductionKind::Count | ReductionKind::NanCount | ReductionKind::InfCount => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::u8_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::u8_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::u8_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| f64::from(v)), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan => {
                for &v in vals {
                    self.push_f64(f64::from(v));
                }
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `i64` LE slab fold (values promoted to `f64`).
    pub fn push_i64_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 8, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[i64] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count | ReductionKind::NanCount | ReductionKind::InfCount => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::i64_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::i64_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::i64_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| v as f64), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan => {
                for &v in vals {
                    self.push_f64(v as f64);
                }
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `u32` LE slab fold (values promoted to `f64`).
    pub fn push_u32_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 4, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[u32] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count | ReductionKind::NanCount | ReductionKind::InfCount => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::u32_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::u32_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::u32_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| f64::from(v)), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan => {
                for &v in vals {
                    self.push_f64(f64::from(v));
                }
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `u64` LE slab fold (values promoted to `f64`).
    pub fn push_u64_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 8, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[u64] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count | ReductionKind::NanCount | ReductionKind::InfCount => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::u64_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::u64_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::u64_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| v as f64), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan => {
                for &v in vals {
                    self.push_f64(v as f64);
                }
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `i16` LE slab fold (values promoted to `f64`).
    pub fn push_i16_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 2, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[i16] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count | ReductionKind::NanCount | ReductionKind::InfCount => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::i16_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::i16_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::i16_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| f64::from(v)), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan => {
                for &v in vals {
                    self.push_f64(f64::from(v));
                }
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `u16` LE slab fold (values promoted to `f64`).
    pub fn push_u16_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 2, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[u16] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count | ReductionKind::NanCount | ReductionKind::InfCount => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::u16_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::u16_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::u16_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| f64::from(v)), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan => {
                for &v in vals {
                    self.push_f64(f64::from(v));
                }
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `f16` LE slab fold (values promoted to `f64`).
    pub fn push_f16_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 2, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[half::f16] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::f16_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::f16_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::f16_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NanCount => self.push_match_count_slice(
                vals.len(),
                vals.iter().filter(|v| f64::from(**v).is_nan()).count(),
            ),
            ReductionKind::InfCount => self.push_match_count_slice(
                vals.len(),
                vals.iter().filter(|v| f64::from(**v).is_infinite()).count(),
            ),
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|v| f64::from(*v)), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan => {
                for &v in vals {
                    self.push_f64(f64::from(v));
                }
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn finish_f64(&self, kind: ReductionKind) -> f64 {
        match kind {
            ReductionKind::Sum => self.sum,
            ReductionKind::Mean => {
                if self.count == 0 {
                    0.0
                } else {
                    self.sum / self.count as f64
                }
            }
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
            ReductionKind::NanCount | ReductionKind::InfCount | ReductionKind::NullCount { .. } => {
                self.match_count as f64
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    #[must_use]
    pub fn finish_bool(&self, kind: ReductionKind) -> bool {
        match kind {
            ReductionKind::AllFinite => self.all_finite,
            ReductionKind::AnyNan => self.any_nan,
            ReductionKind::NanCount | ReductionKind::InfCount | ReductionKind::NullCount { .. } => {
                self.match_count > 0
            }
            _ => self.finish_f64(kind) > 0.5,
        }
    }

    /// Combine partial accumulators from disjoint chunk visits.
    pub fn merge_from(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }
        self.count += other.count;
        self.sum += other.sum;
        self.welford.merge_from(&other.welford);
        self.product *= other.product;
        self.norm_l1 += other.norm_l1;
        self.norm_l2_sq += other.norm_l2_sq;
        self.all_finite &= other.all_finite;
        self.any_nan |= other.any_nan;
        self.match_count += other.match_count;
        if other.have_min_max {
            if self.have_min_max {
                self.min = self.min.min(other.min);
                self.max = self.max.max(other.max);
            } else {
                self.min = other.min;
                self.max = other.max;
                self.have_min_max = true;
            }
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
        let mut nan_count_scalar = None;
        let mut inf_count_scalar = None;
        let mut null_count_scalar = None;
        match kind {
            ReductionKind::Sum => sum_scalar = Some(self.sum),
            ReductionKind::Mean => {
                mean_scalar = Some(if self.count == 0 {
                    0.0
                } else {
                    self.sum / self.count as f64
                });
            }
            ReductionKind::Min => min_scalar = Some(self.min),
            ReductionKind::Max => max_scalar = Some(self.max),
            ReductionKind::Var => var_scalar = Some(self.welford.population_variance()),
            ReductionKind::Std => std_scalar = Some(self.welford.population_std()),
            ReductionKind::Product => product_scalar = Some(self.product),
            ReductionKind::NormL1 => norm_l1_scalar = Some(self.norm_l1),
            ReductionKind::NormL2 => norm_l2_scalar = Some(self.norm_l2_sq.sqrt()),
            ReductionKind::AllFinite => all_finite_scalar = Some(self.all_finite),
            ReductionKind::AnyNan => any_nan_scalar = Some(self.any_nan),
            ReductionKind::NanCount => nan_count_scalar = Some(self.match_count as f64),
            ReductionKind::InfCount => inf_count_scalar = Some(self.match_count as f64),
            ReductionKind::NullCount { .. } => null_count_scalar = Some(self.match_count as f64),
            ReductionKind::Count | ReductionKind::ArgMin | ReductionKind::ArgMax => {}
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
            nan_count_scalar,
            inf_count_scalar,
            null_count_scalar,
            argmin_index: None,
            argmax_index: None,
        }
    }
}

/// Index of min/max in logical row-major order (scalar) or within reduced axes (partial).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ArgIndexAccum {
    have: bool,
    best_val: f64,
    best_idx: u64,
}

impl ArgIndexAccum {
    pub fn is_empty(&self) -> bool {
        !self.have
    }

    pub fn push(&mut self, idx: u64, v: f32, kind: ReductionKind) {
        self.push_f64(idx, f64::from(v), kind);
    }

    pub fn push_f64(&mut self, idx: u64, v: f64, kind: ReductionKind) {
        if !self.have {
            self.best_val = v;
            self.best_idx = idx;
            self.have = true;
            return;
        }
        match kind {
            ReductionKind::ArgMin if v < self.best_val => {
                self.best_val = v;
                self.best_idx = idx;
            }
            ReductionKind::ArgMax if v > self.best_val => {
                self.best_val = v;
                self.best_idx = idx;
            }
            _ => {}
        }
    }

    pub fn index(&self) -> u64 {
        self.best_idx
    }

    /// Combine partial argmin/argmax state from disjoint chunk visits.
    pub fn merge_from(&mut self, other: &Self, kind: ReductionKind) {
        if !other.have {
            return;
        }
        if !self.have {
            *self = *other;
            return;
        }
        match kind {
            ReductionKind::ArgMin if other.best_val < self.best_val => *self = *other,
            ReductionKind::ArgMax if other.best_val > self.best_val => *self = *other,
            _ => {}
        }
    }

    pub fn finish_scalar(self, kind: ReductionKind, element_count: usize) -> ScalarReductionResult {
        let idx = self.best_idx;
        let mut argmin_index = None;
        let mut argmax_index = None;
        match kind {
            ReductionKind::ArgMin => argmin_index = Some(idx),
            ReductionKind::ArgMax => argmax_index = Some(idx),
            _ => {}
        }
        ScalarReductionResult {
            element_count,
            argmin_index,
            argmax_index,
            ..ScalarReductionResult::default_fields(element_count)
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
    pub nan_count_scalar: Option<f64>,
    pub inf_count_scalar: Option<f64>,
    pub null_count_scalar: Option<f64>,
    pub argmin_index: Option<u64>,
    pub argmax_index: Option<u64>,
}

impl ScalarReductionResult {
    /// Combine partial results from disjoint chunk device (or host) reduces.
    #[allow(dead_code)]
    pub(crate) fn merge_partial(&mut self, part: &Self, kind: ReductionKind) {
        if part.element_count == 0 {
            return;
        }
        self.element_count += part.element_count;
        match kind {
            ReductionKind::Sum | ReductionKind::Mean => {
                let acc = self.sum_scalar.get_or_insert(0.0);
                *acc += part.sum_scalar.unwrap_or(0.0);
            }
            ReductionKind::Min => {
                let p = part.min_scalar.expect("partial min");
                match self.min_scalar {
                    Some(m) => self.min_scalar = Some(m.min(p)),
                    None => self.min_scalar = Some(p),
                }
            }
            ReductionKind::Max => {
                let p = part.max_scalar.expect("partial max");
                match self.max_scalar {
                    Some(m) => self.max_scalar = Some(m.max(p)),
                    None => self.max_scalar = Some(p),
                }
            }
            ReductionKind::Var | ReductionKind::Std => {
                unreachable!("var/std use ValueAccum streaming, not ScalarReductionResult merge")
            }
            _ => {}
        }
    }

    /// Set derived fields after all chunk partials are merged.
    #[allow(dead_code)]
    pub(crate) fn finalize_merged(self, kind: ReductionKind) -> Self {
        let mut out = self;
        match kind {
            ReductionKind::Mean if out.element_count > 0 => {
                let sum = out.sum_scalar.unwrap_or(0.0);
                out.mean_scalar = Some(sum / out.element_count as f64);
            }
            ReductionKind::Count => {
                out.sum_scalar = None;
                out.mean_scalar = None;
            }
            _ => {}
        }
        out
    }

    pub(crate) fn default_fields(element_count: usize) -> Self {
        Self {
            element_count,
            sum_scalar: None,
            mean_scalar: None,
            min_scalar: None,
            max_scalar: None,
            var_scalar: None,
            std_scalar: None,
            product_scalar: None,
            norm_l1_scalar: None,
            norm_l2_scalar: None,
            all_finite_scalar: None,
            any_nan_scalar: None,
            nan_count_scalar: None,
            inf_count_scalar: None,
            null_count_scalar: None,
            argmin_index: None,
            argmax_index: None,
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
            var: r.var_scalar,
            std: r.std_scalar,
            product: r.product_scalar,
            norm_l1: r.norm_l1_scalar,
            norm_l2: r.norm_l2_scalar,
            all_finite: r.all_finite_scalar,
            any_nan: r.any_nan_scalar,
            nan_count: r.nan_count_scalar,
            inf_count: r.inf_count_scalar,
            null_count: r.null_count_scalar,
            argmin_index: r.argmin_index,
            argmax_index: r.argmax_index,
            ..Self::default()
        }
    }
}
