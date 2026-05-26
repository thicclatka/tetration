//! `f32` SIMD sum/sumsq and min/max vs scalar reference.

use crate::query::fold::variance_simd::{
    f16_min_max, f16_sum_sumsq, f32_min_max, f32_sum_sumsq, i32_sum_sumsq, i64_sum_sumsq,
    scalar_f32_min_max, scalar_f32_sum_sumsq, scalar_i32_sum_sumsq, scalar_i64_sum_sumsq,
    scalar_u8_sum_sumsq, scalar_u16_sum_sumsq, scalar_u32_sum_sumsq, scalar_u64_sum_sumsq,
    u8_sum_sumsq, u16_sum_sumsq, u32_sum_sumsq, u64_sum_sumsq,
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

#[test]
fn u8_sum_sumsq_matches_scalar_reference() {
    let vals: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
    let scalar = scalar_u8_sum_sumsq(&vals);
    let fast = u8_sum_sumsq(&vals);
    assert!((scalar.0 - fast.0).abs() < 1e-6);
    assert!((scalar.1 - fast.1).abs() < 1e-3);
}

#[test]
fn u16_sum_sumsq_matches_scalar_reference() {
    let vals: Vec<u16> = (0..10_000).map(|i| (i % 40_000) as u16).collect();
    let scalar = scalar_u16_sum_sumsq(&vals);
    let fast = u16_sum_sumsq(&vals);
    assert!((scalar.0 - fast.0).abs() < 1e-6);
    assert!((scalar.1 - fast.1).abs() < 1e-3);
}

#[test]
fn i64_sum_sumsq_matches_scalar_reference() {
    let vals: Vec<i64> = (0..10_000).map(|i| i64::from(i - 5_000)).collect();
    let scalar = scalar_i64_sum_sumsq(&vals);
    let fast = i64_sum_sumsq(&vals);
    assert!((scalar.0 - fast.0).abs() < 1e-6);
    assert!((scalar.1 - fast.1).abs() < 1e-3);
}

#[test]
fn u32_sum_sumsq_matches_scalar_reference() {
    let vals: Vec<u32> = (0..10_000).map(|i| i % 100_000).collect();
    let scalar = scalar_u32_sum_sumsq(&vals);
    let fast = u32_sum_sumsq(&vals);
    assert!((scalar.0 - fast.0).abs() < 1e-6);
    assert!((scalar.1 - fast.1).abs() < 1e-3);
}

#[test]
fn u64_sum_sumsq_matches_scalar_reference() {
    let vals: Vec<u64> = (0..10_000).map(|i| i as u64).collect();
    let scalar = scalar_u64_sum_sumsq(&vals);
    let fast = u64_sum_sumsq(&vals);
    assert!((scalar.0 - fast.0).abs() < 1e-6);
    assert!((scalar.1 - fast.1).abs() < 1e-3);
}

#[test]
fn f16_sum_sumsq_matches_scalar_chunks() {
    let vals: Vec<half::f16> = (0..10_000)
        .map(|i| half::f16::from_f32((i as f32) * 0.01 - 50.0))
        .collect();
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for v in &vals {
        let x = f64::from(f32::from(*v));
        sum += x;
        sumsq += x * x;
    }
    let fast = f16_sum_sumsq(&vals);
    assert!((sum - fast.0).abs() < 1e-3);
    assert!((sumsq - fast.1).abs() < 1e-2);
    let (min, max) = f16_min_max(&vals);
    assert!(min <= max);
}
