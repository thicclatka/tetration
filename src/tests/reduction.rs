//! Bulk variance accumulators vs elementwise Welford reference.

use crate::query::fold::reduction::{ReductionKind, ValueAccum, WelfordAccum};

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
