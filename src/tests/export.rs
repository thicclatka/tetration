//! `.tet` → Zarr v3 export roundtrip (convert fixtures → export → compare chunk bytes).

use super::convert::{materialize_dataset_le_bytes, small_zarr, zarr_array_le_bytes};
use super::verify::assert_tet_verify_ok;
use crate::catalog::read_tet_summary_v1;
use crate::convert::convert_zarr_to_tet_with_progress;
use crate::export::export_tet_to_zarr;
use crate::layout::mmap_file_read;

const FIXTURE_DTYPES: [&str; 4] = ["f32", "f64", "i32", "i64"];

#[test]
fn export_zarr_tensor_3d_roundtrip_matches_fixture_bytes() {
    let input = small_zarr("tensor_3d");
    let dir = tempfile::tempdir().unwrap();
    let tet = dir.path().join("tensor_3d.tet");
    let zarr_out = dir.path().join("exported");
    convert_zarr_to_tet_with_progress(&input, &tet, 1, None::<fn(_)>).unwrap();
    export_tet_to_zarr(&tet, &zarr_out).unwrap();
    assert_tet_verify_ok(&tet);

    for name in FIXTURE_DTYPES {
        let want = zarr_array_le_bytes(&input.join(name));
        let got = zarr_array_le_bytes(&zarr_out.join(name));
        assert_eq!(
            got, want,
            "dataset {name} zarr bytes differ after export roundtrip"
        );
    }
}

#[test]
fn export_zarr_groups_3d_nested_paths() {
    let input = small_zarr("groups_3d");
    let dir = tempfile::tempdir().unwrap();
    let tet = dir.path().join("groups.tet");
    let zarr_out = dir.path().join("exported");
    convert_zarr_to_tet_with_progress(&input, &tet, 1, None::<fn(_)>).unwrap();
    export_tet_to_zarr(&tet, &zarr_out).unwrap();

    for dtype in FIXTURE_DTYPES {
        let rel = format!("primary/{dtype}");
        let want = zarr_array_le_bytes(&input.join(&rel));
        let got = zarr_array_le_bytes(&zarr_out.join(&rel));
        assert_eq!(got, want, "dataset {rel}");
        assert!(zarr_out.join("primary/zarr.json").is_file());
    }
}

#[test]
fn export_preserves_logical_f32_after_convert() {
    let input = small_zarr("tensor_3d");
    let dir = tempfile::tempdir().unwrap();
    let tet = dir.path().join("t.tet");
    let zarr_out = dir.path().join("out");
    convert_zarr_to_tet_with_progress(&input, &tet, 1, None::<fn(_)>).unwrap();
    export_tet_to_zarr(&tet, &zarr_out).unwrap();
    let mmap = mmap_file_read(&tet).unwrap();
    let want = materialize_dataset_le_bytes(&mmap, "f32");
    let got = zarr_array_le_bytes(&zarr_out.join("f32"));
    assert_eq!(got, want);
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets.len(), 4);
}
