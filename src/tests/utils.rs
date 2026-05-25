//! Utils integration tests.

use crate::utils::host_memory::available_memory_bytes;

#[test]
fn host_memory_probe_does_not_panic() {
    let _ = available_memory_bytes();
}
