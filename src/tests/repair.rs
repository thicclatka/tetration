//! `tet repair` / [`repair_tet_file`] integration tests.

use crate::catalog::{FooterBlobV1, TetMetadataV1, write_footer_blob};
use crate::repair::{RepairOptions, repair_command_for_code, repair_tet_file};
use crate::verify::verify_tet_file;

use super::fixture::write_multichunk_2x3_tiles;

fn corrupt_footer_json(path: &std::path::Path) {
    let mut data = std::fs::read(path).unwrap();
    const TAIL: usize = 16;
    let json_len = u64::from_le_bytes(
        data[data.len() - TAIL..data.len() - TAIL + 8]
            .try_into()
            .unwrap(),
    );
    let json_end = data.len() - TAIL;
    let json_start = json_end - usize::try_from(json_len).unwrap();
    for b in &mut data[json_start..json_end] {
        *b = b'X';
    }
    std::fs::write(path, &data).unwrap();
}

#[test]
fn verify_recommendation_includes_repair_command() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_footer.tet");
    write_multichunk_2x3_tiles(&path, "a");
    write_footer_blob(
        &path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata: Some(TetMetadataV1::default()),
            metadata_ref: None,
        },
    )
    .unwrap();
    corrupt_footer_json(&path);

    let report = verify_tet_file(&path).unwrap();
    assert!(!report.ok);
    let rec = report
        .recommendations
        .iter()
        .find(|r| r.code == "footer_invalid")
        .expect("footer recommendation");
    let cmd = rec
        .fix
        .as_ref()
        .and_then(|f| f.command.as_deref())
        .expect("repair command");
    assert!(cmd.contains("tet repair"));
    assert!(cmd.contains("footer_invalid"));
    assert!(cmd.contains(path.to_str().unwrap()));

    let expected = repair_command_for_code(&path, "footer_invalid").unwrap();
    assert_eq!(cmd, expected);
}

#[test]
fn repair_strip_invalid_footer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fix.tet");
    write_multichunk_2x3_tiles(&path, "a");
    write_footer_blob(
        &path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata: Some(TetMetadataV1::default()),
            metadata_ref: None,
        },
    )
    .unwrap();
    let len_before = std::fs::metadata(&path).unwrap().len();
    corrupt_footer_json(&path);
    assert!(std::fs::metadata(&path).unwrap().len() >= len_before);

    assert!(!verify_tet_file(&path).unwrap().ok);

    let options = RepairOptions {
        dry_run: false,
        apply: vec!["footer_invalid".to_owned()],
        plan_codes: Vec::new(),
    };
    let report = repair_tet_file(&path, &options).unwrap();
    assert!(report.actions.iter().any(|a| a.applied));

    let after = verify_tet_file(&path).unwrap();
    assert!(after.ok, "{:?}", after.findings);
    assert!(after.summary.as_ref().is_some_and(|s| !s.history_footer));
    assert!(std::fs::metadata(&path).unwrap().len() < len_before);
}

#[test]
fn repair_dry_run_does_not_truncate() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dry.tet");
    write_multichunk_2x3_tiles(&path, "a");
    write_footer_blob(
        &path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata: Some(TetMetadataV1::default()),
            metadata_ref: None,
        },
    )
    .unwrap();
    corrupt_footer_json(&path);
    let len = std::fs::metadata(&path).unwrap().len();

    let options = RepairOptions {
        dry_run: true,
        apply: vec!["footer_invalid".to_owned()],
        plan_codes: Vec::new(),
    };
    repair_tet_file(&path, &options).unwrap();

    assert_eq!(std::fs::metadata(&path).unwrap().len(), len);
}
