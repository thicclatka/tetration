//! Best-effort host RAM available for new allocations.

use std::process::Command;

/// Returns an estimate of bytes the OS considers available for new allocations, if detectable.
#[must_use]
pub fn available_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        read_linux_mem_available()
    }
    #[cfg(target_os = "macos")]
    {
        read_macos_available()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn read_linux_mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return kb.checked_mul(1024);
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn read_macos_available() -> Option<u64> {
    let page_size = sysctl_u64("hw.pagesize")?;
    let free = sysctl_u64("vm.page_free_count")?;
    let inactive = sysctl_u64("vm.page_inactive_count").unwrap_or(0);
    Some(free.saturating_add(inactive).saturating_mul(page_size))
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn sysctl_u64(name: &str) -> Option<u64> {
    let out = Command::new("sysctl").args(["-n", name]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    std::str::from_utf8(&out.stdout).ok()?.trim().parse().ok()
}
