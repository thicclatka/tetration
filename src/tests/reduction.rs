//! Bulk variance accumulators vs elementwise Welford reference.

use crate::query::fold::reduction::{
    ReductionKind, ScalarReductionResult, ValueAccum, WelfordAccum,
};
use crate::query::fold::variance_simd;

fn var_from(vals: &[f32]) -> f64 {
    let mut acc = ValueAccum::default();
    acc.push_f32_le_bytes(bytemuck::cast_slice(vals), ReductionKind::Var);
    acc.finish_f64(ReductionKind::Var)
}

fn var_elementwise(vals: &[f32]) -> f64 {
    let mut w = WelfordAccum::default();
    for &v in vals {
        w.push(f64::from(v));
    }
    w.population_variance()
}

#[test]
fn f32_population_variance_large_slice_matches_welford() {
    // Bench-scale element count; f32 GPU tree reduce was wrong here, f64 SIMD must hold.
    let n = 2_000_000usize;
    let vals: Vec<f32> = (0..n).map(|i| (i % 1000) as f32 / 999.0).collect();
    let (sum, sumsq) = variance_simd::f32_sum_sumsq(&vals);
    let nf = n as f64;
    let mean = sum / nf;
    let simd_var = (sumsq / nf - mean * mean).max(0.0);
    let welford_var = var_elementwise(&vals);
    assert!(
        (simd_var - welford_var).abs() < 1e-9,
        "simd={simd_var} welford={welford_var}"
    );
    // Uniform on [0, 1] → population variance 1/12.
    assert!((simd_var - 1.0 / 12.0).abs() < 0.01, "simd={simd_var}");
}

#[test]
fn scalar_reduction_merge_partial_sum_and_mean() {
    let mut acc = ScalarReductionResult::default_fields(0);
    acc.merge_partial(
        &ScalarReductionResult {
            element_count: 3,
            sum_scalar: Some(6.0),
            ..ScalarReductionResult::default_fields(3)
        },
        ReductionKind::Sum,
    );
    acc.merge_partial(
        &ScalarReductionResult {
            element_count: 3,
            sum_scalar: Some(15.0),
            ..ScalarReductionResult::default_fields(3)
        },
        ReductionKind::Sum,
    );
    let done = acc.finalize_merged(ReductionKind::Sum);
    assert_eq!(done.element_count, 6);
    assert!((done.sum_scalar.unwrap() - 21.0).abs() < 1e-9);

    let mut mean_acc = ScalarReductionResult::default_fields(0);
    mean_acc.merge_partial(
        &ScalarReductionResult {
            element_count: 2,
            sum_scalar: Some(3.0),
            ..ScalarReductionResult::default_fields(2)
        },
        ReductionKind::Mean,
    );
    mean_acc.merge_partial(
        &ScalarReductionResult {
            element_count: 2,
            sum_scalar: Some(7.0),
            ..ScalarReductionResult::default_fields(2)
        },
        ReductionKind::Mean,
    );
    let mean_done = mean_acc.finalize_merged(ReductionKind::Mean);
    assert_eq!(mean_done.element_count, 4);
    assert!((mean_done.mean_scalar.unwrap() - 2.5).abs() < 1e-9);
}

#[test]
fn bulk_f32_var_matches_elementwise_welford() {
    let vals: Vec<f32> = (0..10_000).map(|i| (i as f32) * 0.001).collect();
    let bulk = var_from(&vals);
    let elem = var_elementwise(&vals);
    assert!((bulk - elem).abs() < 1e-6, "bulk={bulk} elem={elem}");
}

#[test]
fn bulk_f64_var_matches_elementwise_welford() {
    let vals: Vec<f64> = (0..10_000).map(|i| i as f64 * 0.001).collect();
    let mut bulk = ValueAccum::default();
    bulk.push_f64_le_bytes(bytemuck::cast_slice(&vals), ReductionKind::Var);
    let bulk_v = bulk.finish_f64(ReductionKind::Var);

    let mut w = WelfordAccum::default();
    for &v in &vals {
        w.push(v);
    }
    let elem = w.population_variance();
    assert!((bulk_v - elem).abs() < 1e-9, "bulk={bulk_v} elem={elem}");
}

#[test]
fn bulk_i32_var_matches_elementwise_welford() {
    let vals: Vec<i32> = (0..10_000).map(|i| i - 5_000).collect();
    let mut bulk = ValueAccum::default();
    bulk.push_i32_le_bytes(bytemuck::cast_slice(&vals), ReductionKind::Var);
    let bulk_v = bulk.finish_f64(ReductionKind::Var);

    let mut w = WelfordAccum::default();
    for &v in &vals {
        w.push(f64::from(v));
    }
    let elem = w.population_variance();
    assert!((bulk_v - elem).abs() < 1e-6, "bulk={bulk_v} elem={elem}");
}

#[test]
fn bulk_u8_var_matches_elementwise_welford() {
    let vals: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
    let mut bulk = ValueAccum::default();
    bulk.push_u8_le_bytes(&vals, ReductionKind::Var);
    let bulk_v = bulk.finish_f64(ReductionKind::Var);

    let mut w = WelfordAccum::default();
    for &v in &vals {
        w.push(f64::from(v));
    }
    let elem = w.population_variance();
    assert!((bulk_v - elem).abs() < 1e-6, "bulk={bulk_v} elem={elem}");
}
