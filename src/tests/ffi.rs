//! C ABI smoke tests (`cargo test --lib --features tetration-ffi --no-default-features ffi`).

use std::ffi::{CStr, CString};
use std::ptr;

use crate::ffi::{
    TET_ABI_VERSION, tet_abi_version, tet_close, tet_last_error, tet_open, tet_query_json,
    tet_string_free, tet_summary_json, tet_verify_json,
};

use super::fixture::tracked_small_tet_dir;

fn sample_path() -> CString {
    let p = tracked_small_tet_dir().join("sample.tet");
    assert!(
        p.is_file(),
        "missing {}; commit fixtures/small/tet/ or regenerate",
        p.display()
    );
    CString::new(p.to_str().expect("fixture path utf-8")).expect("no nul in path")
}

#[test]
fn ffi_abi_version() {
    assert_eq!(unsafe { tet_abi_version() }, TET_ABI_VERSION);
}

#[test]
fn ffi_open_summary_query_close() {
    let path = sample_path();
    unsafe {
        let h = tet_open(path.as_ptr());
        assert!(
            !h.is_null(),
            "{}",
            CStr::from_ptr(tet_last_error()).to_string_lossy()
        );

        let summary = tet_summary_json(h);
        assert!(
            !summary.is_null(),
            "{}",
            CStr::from_ptr(tet_last_error()).to_string_lossy()
        );
        let summary_s = CStr::from_ptr(summary).to_str().unwrap();
        assert!(summary_s.contains("temperature"));
        tet_string_free(summary);

        let q = CString::new(r#"{"dataset":"temperature","mean":[]}"#).unwrap();
        let out = tet_query_json(h, q.as_ptr());
        assert!(
            !out.is_null(),
            "{}",
            CStr::from_ptr(tet_last_error()).to_string_lossy()
        );
        let out_s = CStr::from_ptr(out).to_str().unwrap();
        assert!(out_s.contains("operation_mean"));
        tet_string_free(out);

        tet_close(h);
        tet_close(ptr::null_mut());
    }
}

#[test]
fn ffi_verify_json() {
    let path = sample_path();
    unsafe {
        let out = tet_verify_json(path.as_ptr());
        assert!(
            !out.is_null(),
            "{}",
            CStr::from_ptr(tet_last_error()).to_string_lossy()
        );
        let s = CStr::from_ptr(out).to_str().unwrap();
        assert!(s.contains("\"ok\":true"));
        tet_string_free(out);
    }
}

#[test]
fn ffi_open_invalid_path() {
    let bad = CString::new("/nonexistent/tetration/sample.tet").unwrap();
    unsafe {
        let h = tet_open(bad.as_ptr());
        assert!(h.is_null());
        assert!(!CStr::from_ptr(tet_last_error()).to_bytes().is_empty());
    }
}
