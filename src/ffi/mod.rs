//! Stable C ABI for embedders (Phase 11). Enable with feature **`tetration-ffi`**.
//!
//! Symbols and `include/tetration.h` must stay in sync.

mod error;

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr;

use crate::catalog::TetFile;
use crate::query::{ExecuteQueryOptions, execute_query_json};
use crate::verify::{VerifyOptions, verify_tet_bytes};

pub use error::clear_last_error;

/// ABI version (`include/tetration.h` → `TET_ABI_VERSION`). Bump on breaking C layout/symbol changes.
pub const TET_ABI_VERSION: u32 = 1;

/// Opaque `.tet` handle (`Box<TetFile>`).
#[repr(C)]
pub struct TetHandle {
    _private: [u8; 0],
}

fn handle_mut<'a>(ptr: *mut TetHandle) -> Option<&'a mut TetFile> {
    if ptr.is_null() {
        error::set_last_error("handle: null pointer");
        return None;
    }
    // SAFETY: exclusive handle; caller must not use after `tet_close`.
    Some(unsafe { &mut *ptr.cast::<TetFile>() })
}

fn ffi_guard<F, T>(f: F) -> Option<T>
where
    F: FnOnce() -> Option<T>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => {
            error::set_last_error("internal panic (aborting cdylib build is recommended for FFI)");
            None
        }
    }
}

fn cstr_input<'a>(ptr: *const c_char, field: &'static str) -> Option<&'a CStr> {
    if ptr.is_null() {
        error::set_last_error(format!("{field}: null pointer"));
        return None;
    }
    // SAFETY: caller must pass a valid NUL-terminated C string for the call duration.
    let s = unsafe { CStr::from_ptr(ptr) };
    if s.to_bytes().is_empty() {
        error::set_last_error(format!("{field}: empty path"));
        return None;
    }
    Some(s)
}

fn return_json(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => {
            error::set_last_error("internal: JSON contained interior NUL");
            ptr::null_mut()
        }
    }
}

fn json_output<E: std::fmt::Display>(r: Result<String, E>) -> *mut c_char {
    match r {
        Ok(s) => return_json(s),
        Err(e) => {
            error::set_last_error(e.to_string());
            ptr::null_mut()
        }
    }
}

/// Return compile-time ABI version.
///
/// # Safety
///
/// Always safe to call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tet_abi_version() -> u32 {
    TET_ABI_VERSION
}

/// Open a `.tet` file read-only. Returns null on error (`tet_last_error`).
///
/// # Safety
///
/// `path` must be a valid NUL-terminated UTF-8 path for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tet_open(path: *const c_char) -> *mut TetHandle {
    ffi_guard(|| {
        let cstr = cstr_input(path, "path")?;
        let path_str = match cstr.to_str() {
            Ok(p) => p,
            Err(_) => {
                error::set_last_error("path: invalid UTF-8");
                return None;
            }
        };
        match TetFile::open(path_str) {
            Ok(file) => Some(Box::into_raw(Box::new(file)).cast::<TetHandle>()),
            Err(e) => {
                error::set_last_error(e.to_string());
                None
            }
        }
    })
    .unwrap_or(ptr::null_mut())
}

/// Close a handle from [`tet_open`]. Null is a no-op.
///
/// # Safety
///
/// `handle` must be null or a pointer returned by `tet_open` and not previously closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tet_close(handle: *mut TetHandle) {
    if handle.is_null() {
        return;
    }
    // SAFETY: reclaim box from `tet_open`.
    drop(unsafe { Box::from_raw(handle.cast::<TetFile>()) });
}

/// UTF-8 error from the last failed FFI call on this thread, or empty string.
///
/// # Safety
///
/// Pointer is valid until the next FFI call on this thread; do not free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tet_last_error() -> *const c_char {
    error::last_error_cstr()
}

/// Clear the thread-local error string.
///
/// # Safety
///
/// Always safe to call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tet_clear_error() {
    clear_last_error();
}

/// Catalog + superblock summary JSON (`TetFileSummaryV1`). Caller frees with [`tet_string_free`].
///
/// # Safety
///
/// `handle` must be a valid open handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tet_summary_json(handle: *mut TetHandle) -> *mut c_char {
    ffi_guard(|| {
        let file = handle_mut(handle)?;
        Some(json_output(
            file.summary()
                .map_err(|e| e.to_string())
                .and_then(|s| serde_json::to_string(&s).map_err(|e| e.to_string())),
        ))
    })
    .unwrap_or(ptr::null_mut())
}

/// Execute a query document JSON; returns `QueryResponse` JSON. Caller frees with [`tet_string_free`].
///
/// # Safety
///
/// `handle` must be valid; `query_json` a NUL-terminated UTF-8 JSON document for the call duration.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tet_query_json(
    handle: *mut TetHandle,
    query_json: *const c_char,
) -> *mut c_char {
    ffi_guard(|| {
        let file = handle_mut(handle)?;
        let q = cstr_input(query_json, "query_json")?;
        let query = match q.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("query_json: invalid UTF-8");
                return None;
            }
        };
        let path = file.path();
        let mmap = file.mmap();
        let resp = match execute_query_json(
            query,
            path,
            mmap,
            ExecuteQueryOptions::execute_no_preview(),
            None,
        ) {
            Ok(r) => r,
            Err(e) => {
                error::set_last_error(e.to_string());
                return None;
            }
        };
        Some(json_output(
            serde_json::to_string(&resp).map_err(|e| e.to_string()),
        ))
    })
    .unwrap_or(ptr::null_mut())
}

/// Quick verify report JSON for `path` (no handle). Caller frees with [`tet_string_free`].
///
/// # Safety
///
/// `path` must be a valid NUL-terminated UTF-8 path for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tet_verify_json(path: *const c_char) -> *mut c_char {
    ffi_guard(|| {
        let cstr = cstr_input(path, "path")?;
        let path_str = match cstr.to_str() {
            Ok(p) => p,
            Err(_) => {
                error::set_last_error("path: invalid UTF-8");
                return None;
            }
        };
        let data = match std::fs::read(path_str) {
            Ok(d) => d,
            Err(e) => {
                error::set_last_error(e.to_string());
                return None;
            }
        };
        let report = verify_tet_bytes(&data, Some(Path::new(path_str)), VerifyOptions::default());
        Some(json_output(
            serde_json::to_string(&report).map_err(|e| e.to_string()),
        ))
    })
    .unwrap_or(ptr::null_mut())
}

/// Free a string returned by `tet_*_json`. Null is a no-op.
///
/// # Safety
///
/// `s` must be null or a pointer previously returned by this library and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tet_string_free(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    // SAFETY: reclaim CString from `return_json`.
    drop(unsafe { CString::from_raw(s) });
}
