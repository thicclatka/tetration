//! Filesystem / volume identity helpers for atomic publish (rename vs copy).
//!
//! Sidecar `.tet` writes may build in platform cache then land beside the source file.
//! [`same_filesystem`] picks [`publish_file`]'s strategy without relying on a failed `rename`.
//!
//! Production call sites (transform sidecar publish) are not wired yet; integration tests
//! in `src/tests/fs_device.rs` exercise this module until then.

#![allow(dead_code)]

use std::io;
use std::path::Path;

/// Stable device / volume id for the filesystem hosting `path` (parent used when `path` is a file).
///
/// # Errors
///
/// I/O errors from [`std::fs::metadata`] (missing path, permissions).
pub fn device_id(path: &Path) -> io::Result<u64> {
    let probe = path_for_device_probe(path);
    device_id_metadata(&probe)
}

/// `true` when `a` and `b` reside on the same mounted filesystem / volume.
///
/// Compares [`device_id`] of each path (files probe their parent directory). Bind mounts
/// that share a device id are treated as same-filesystem.
///
/// # Errors
///
/// Propagates metadata errors from either path.
pub fn same_filesystem(a: &Path, b: &Path) -> io::Result<bool> {
    Ok(device_id(a)? == device_id(b)?)
}

/// Move `temp` to `dest`, using `rename` on one filesystem or `copy` + `remove_file` across volumes.
///
/// Creates `dest`'s parent directories when missing. On success `temp` no longer exists.
///
/// # Errors
///
/// I/O errors from probe, directory creation, rename, copy, or remove.
pub fn publish_file(temp: &Path, dest: &Path) -> io::Result<()> {
    if temp == dest {
        return Ok(());
    }
    if let Some(parent) = dest.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    if same_filesystem(temp, dest)? {
        std::fs::rename(temp, dest)?;
    } else {
        std::fs::copy(temp, dest)?;
        std::fs::remove_file(temp)?;
    }
    Ok(())
}

fn path_for_device_probe(path: &Path) -> std::path::PathBuf {
    if let Ok(meta) = path.metadata() {
        if meta.is_file() {
            return path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .map_or_else(|| path.to_path_buf(), std::path::Path::to_path_buf);
        }
        return path.to_path_buf();
    }
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map_or_else(|| path.to_path_buf(), std::path::Path::to_path_buf)
}

fn device_id_metadata(path: &Path) -> io::Result<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(std::fs::metadata(path)?.dev())
    }
    #[cfg(windows)]
    {
        volume_serial_number(path)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = path;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "device_id is only supported on Unix and Windows",
        ))
    }
}

#[cfg(windows)]
fn volume_serial_number(path: &Path) -> io::Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    use std::ptr;

    fn wide(path: &Path) -> Vec<u16> {
        path.as_os_str().encode_wide().chain([0]).collect()
    }

    const MAX_PATH: usize = 260;

    let input = wide(path);
    let mut volume_path = vec![0u16; MAX_PATH];
    let volume_len = unsafe {
        GetVolumePathNameW(
            input.as_ptr(),
            volume_path.as_mut_ptr(),
            u32::try_from(MAX_PATH).expect("MAX_PATH fits in u32"),
        )
    };
    if volume_len == 0 {
        return Err(io::Error::last_os_error());
    }
    volume_path.truncate(volume_len as usize);

    let mut serial = 0u32;
    let ok = unsafe {
        GetVolumeInformationW(
            volume_path.as_ptr(),
            ptr::null_mut(),
            0,
            &mut serial,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            0,
        )
    };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(u64::from(serial))
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn GetVolumePathNameW(
        lpsz_file_name: *const u16,
        lpsz_volume_path_name: *mut u16,
        cch_buffer_length: u32,
    ) -> u32;

    fn GetVolumeInformationW(
        lp_root_path_name: *const u16,
        lp_volume_name_buffer: *mut u16,
        n_volume_name_size: u32,
        lp_volume_serial_number: *mut u32,
        lp_maximum_component_length: *mut u32,
        lp_file_system_flags: *mut u32,
        lp_file_system_name_buffer: *mut u16,
        n_file_system_name_size: u32,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn same_directory_is_same_filesystem() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.bin");
        let b = dir.path().join("b.bin");
        fs::write(&a, b"x").unwrap();
        assert!(same_filesystem(&a, &b).unwrap());
    }

    #[test]
    fn publish_file_rename_within_tempdir() {
        let dir = tempfile::tempdir().unwrap();
        let temp = dir.path().join("temp.bin");
        let dest = dir.path().join("dest.bin");
        fs::write(&temp, b"payload").unwrap();
        publish_file(&temp, &dest).unwrap();
        assert!(!temp.exists());
        assert_eq!(fs::read(&dest).unwrap(), b"payload");
    }
}
