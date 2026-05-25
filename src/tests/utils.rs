//! Utils integration tests.

use crate::available_memory_bytes;

#[test]
fn host_memory_probe_does_not_panic() {
    let _ = available_memory_bytes();
}
