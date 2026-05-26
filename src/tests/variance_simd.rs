//! `f32` SIMD sum/sumsq and min/max vs scalar reference.

use crate::query::fold::variance_simd::{
    f32_min_max, f32_sum_sumsq, i32_sum_sumsq, scalar_f32_min_max, scalar_f32_sum_sumsq,
    scalar_i32_sum_sumsq,
};

#[test]
fn empty_slice() {
    assert_eq!(f32_sum_sumsq(&[]), (0.0, 0.0));
}

#[test]
fn matches_scalar_reference() {
    let vals: Vec<f32> = (0..10_000).map(|i| (i as f32) * 0.001 - 5.0).collect();
    let scalar = scalar_f32_sum_sumsq(&vals);
    let fast = f32_sum_sumsq(&vals);
    assert!((scalar.0 - fast.0).abs() < 1e-6);
    assert!((scalar.1 - fast.1).abs() < 1e-3);
}

#[test]
fn min_max_matches_scalar_reference() {
    let vals: Vec<f32> = (0..10_000).map(|i| (i as f32) * 0.001 - 5.0).collect();
    let scalar = scalar_f32_min_max(&vals);
    let fast = f32_min_max(&vals);
    assert_eq!(scalar.0, fast.0);
    assert_eq!(scalar.1, fast.1);
}

#[test]
fn i32_sum_sumsq_matches_scalar_reference() {
    let vals: Vec<i32> = (0..10_000).map(|i| i - 5_000).collect();
    let scalar = scalar_i32_sum_sumsq(&vals);
    let fast = i32_sum_sumsq(&vals);
    assert!((scalar.0 - fast.0).abs() < 1e-6);
    assert!((scalar.1 - fast.1).abs() < 1e-3);
}
