//! Thread-local last error for FFI callers.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Store a UTF-8 error message for [`super::tet_last_error`].
pub fn set_last_error(msg: impl Into<String>) {
    let msg = msg.into();
    let _ = LAST_ERROR.try_with(|cell| {
        *cell.borrow_mut() = CString::new(msg).ok();
    });
}

/// Clear the thread-local error.
pub fn clear_last_error() {
    let _ = LAST_ERROR.try_with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Pointer to NUL-terminated error text, or empty C string.
pub fn last_error_cstr() -> *const c_char {
    static EMPTY: &[u8] = b"\0";
    LAST_ERROR
        .try_with(|cell| {
            cell.borrow()
                .as_ref()
                .map_or(EMPTY.as_ptr().cast(), |c| c.as_ptr())
        })
        .unwrap_or(EMPTY.as_ptr().cast())
}
