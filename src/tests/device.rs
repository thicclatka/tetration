//! Phase 10 device routing (CPU scaffold until GPU kernels ship).

use super::fixture;
use crate::layout::mmap_file_read;
use crate::query::{
    ExecutionDeviceHint, gpu_backend_available, plan_query_with_tet_mmap, resolve_device_route,
    validate_query,
};
use crate::utils::dtype::ElementDtype;

#[test]
fn execution_device_hint_parse_tokens() {
    assert_eq!(
        ExecutionDeviceHint::parse("cpu").unwrap(),
        ExecutionDeviceHint::Cpu
    );
    assert_eq!(
        ExecutionDeviceHint::parse("auto").unwrap(),
        ExecutionDeviceHint::Auto
    );
    assert_eq!(
        ExecutionDeviceHint::parse("cuda").unwrap(),
        ExecutionDeviceHint::Cuda(0)
    );
    assert_eq!(
        ExecutionDeviceHint::parse("cuda:2").unwrap(),
        ExecutionDeviceHint::Cuda(2)
    );
    assert!(ExecutionDeviceHint::parse("metal").is_err());
}

#[test]
fn parse_query_json_execution_device() {
    let doc = crate::query::parse_query_json(
        r#"{"dataset":"a","mean":[],"execution":{"device":"auto"}}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    assert_eq!(
        doc.execution.as_ref().unwrap().device,
        Some(ExecutionDeviceHint::Auto)
    );
}

#[test]
fn resolve_device_auto_small_selection_stays_cpu() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("small.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = fixture::query_files::json("mean_a");
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    let plan = response.read_plan.as_ref().unwrap();
    let route = resolve_device_route(
        Some(&crate::query::ExecutionHints {
            device: Some(ExecutionDeviceHint::Auto),
            ..Default::default()
        }),
        plan,
        ElementDtype::F32,
        doc.operation.as_ref(),
    );
    assert_eq!(route.used, "cpu");
    assert_eq!(route.fallback_reason, Some("auto_below_size_threshold"));
}

#[test]
fn resolve_device_cuda_falls_back_without_gpu_feature() {
    if gpu_backend_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.tet");
    fixture::write_verify_large_f32(&path, "a");
    let doc = crate::query::parse_query_json(
        r#"{"dataset":"a","mean":[],"execution":{"device":"cuda:0"}}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    let ex = response.execution.as_ref().unwrap();
    assert_eq!(ex.device_requested.as_deref(), Some("cuda"));
    assert_eq!(ex.device_used, Some("cpu"));
    assert_eq!(
        ex.device_fallback_reason.as_deref(),
        Some("gpu_feature_disabled")
    );
    assert_eq!(ex.device_gpu_reduce, Some(false));
}
