//! Footer `metadata` JSON roundtrip.

use super::fixture::write_multichunk_2x3_tiles;
use crate::catalog::{
    FooterBlobV1, TetMetadataV1, append_convert_history, read_footer_blob, read_tet_summary_v1,
    write_footer_blob,
};
use crate::layout::mmap_file_read;

#[test]
fn convert_history_preserves_existing_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("meta_convert.tet");
    write_multichunk_2x3_tiles(&path, "a");

    let meta = TetMetadataV1 {
        file: None,
        datasets: [(
            "a".to_owned(),
            crate::catalog::DatasetMetadataV1 {
                attrs: [("units".to_owned(), "K".to_owned())].into_iter().collect(),
                dim_names: Some(vec!["y".to_owned(), "x".to_owned()]),
                coords: None,
            },
        )]
        .into_iter()
        .collect(),
    };
    write_footer_blob(
        &path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata: Some(meta),
        },
    )
    .unwrap();

    append_convert_history(&path, "h5").unwrap();

    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.history.len(), 1);
    assert_eq!(summary.history[0].0, "convert");
    assert_eq!(
        summary.metadata.datasets["a"]
            .attrs
            .get("units")
            .map(String::as_str),
        Some("K")
    );

    let blob = read_footer_blob(&mmap, summary.superblock.flags).unwrap();
    assert!(blob.metadata.is_some());
    assert_eq!(blob.history.len(), 1);
}
