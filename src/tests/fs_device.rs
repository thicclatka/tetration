//! Integration tests for [`crate::utils::fs_device`] and spill publish helper.

use std::fs;
use std::path::Path;

use crate::query::engine::spill_policy::publish_output_file;

#[test]
fn publish_output_file_same_as_publish_file_in_tempdir() {
    let dir = tempfile::tempdir().unwrap();
    let temp = dir.path().join("temp.bin");
    let dest = dir.path().join("dest.bin");
    fs::write(&temp, b"ok").unwrap();
    publish_output_file(&temp, &dest).unwrap();
    assert!(!temp.exists());
    assert_eq!(fs::read(&dest).unwrap(), b"ok");
}

#[test]
fn same_filesystem_across_missing_dest_path() {
    let dir = tempfile::tempdir().unwrap();
    let existing = dir.path().join("a.bin");
    let future = dir.path().join("not_created_yet.bin");
    fs::write(&existing, b"x").unwrap();
    assert!(crate::utils::fs_device::same_filesystem(&existing, &future).unwrap());
}

#[test]
fn device_id_reports_parent_for_files() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("x.bin");
    fs::write(&file, b"").unwrap();
    let dir_dev = crate::utils::fs_device::device_id(dir.path()).unwrap();
    let file_dev = crate::utils::fs_device::device_id(&file).unwrap();
    assert_eq!(dir_dev, file_dev);
}

#[test]
#[cfg(unix)]
fn publish_picks_rename_on_same_device() {
    let dir = tempfile::tempdir().unwrap();
    let temp = dir.path().join("draft.tet");
    let dest = dir.path().join("volume_zscore.tet");
    fs::write(&temp, b"tet").unwrap();
    assert!(
        crate::utils::fs_device::same_filesystem(&temp, &dest).unwrap(),
        "tempdir paths should share a device"
    );
    publish_output_file(&temp, &dest).unwrap();
    assert!(dest.exists());
}

#[allow(dead_code)]
fn _cross_device_copy_path_documentation_only() -> Option<()> {
    // When cache and data paths differ in st_dev, `publish_file` copies then unlinks temp.
    // Run `scripts/fs-device-info.sh ~/.cache/tetration /path/to/data.tet` to compare DEV.
    let _cache: &Path = Path::new("/tmp");
    let _data: &Path = Path::new("/Volumes/hdd/data.tet");
    None
}
