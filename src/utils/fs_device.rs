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
    Ok(device_id_from_metadata(&std::fs::metadata(path)?))
}

#[cfg(unix)]
fn device_id_from_metadata(meta: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    meta.dev()
}

#[cfg(windows)]
fn device_id_from_metadata(meta: &std::fs::Metadata) -> io::Result<u64> {
    use std::os::windows::fs::MetadataExt;
    Ok(u64::from(meta.volume_serial_number()))
}

#[cfg(not(any(unix, windows)))]
fn device_id_from_metadata(_meta: &std::fs::Metadata) -> io::Result<u64> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "device_id is only supported on Unix and Windows",
    ))
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
