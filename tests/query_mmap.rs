mod fixture;

use tetration::{
    CHUNK_TOUCH_POLICY, DTYPE_F32, OneChunkRawWrite, create_empty_v1_file,
    materialize_read_plan_f32_le, mmap_file_read, parse_query_json, plan_query_with_tet_mmap,
    validate_query, write_one_chunk_raw_file,
};

use fixture::{CHUNK_2X2, SHAPE_2X3, write_multichunk_2x3_tiles};

#[test]
fn plan_query_resolves_dataset_in_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q.tet");
    write_multichunk_2x3_tiles(&path, "temperature");

    let doc = parse_query_json(r#"{"dataset":"temperature","layout_version":1}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, Some("q.tet"), &mmap, None).unwrap();
    assert_eq!(plan.tet_file.as_deref(), Some("q.tet"));
    let cat = plan.catalog.as_ref().unwrap();
    assert!(cat.matched);
    assert_eq!(cat.dataset_index, Some(0));
    assert_eq!(cat.shape.as_ref().unwrap(), &SHAPE_2X3);
    assert_eq!(cat.chunk_shape.as_ref().unwrap(), &CHUNK_2X2);
    assert_eq!(cat.chunk_index_rows, Some(2));
    let rp = plan.read_plan.as_ref().unwrap();
    assert_eq!(rp.chunk_count, 2);
    assert_eq!(
        rp.chunk_touch_policy,
        CHUNK_TOUCH_POLICY.dense_half_open_unit_step
    );
    assert_eq!(rp.total_stored_bytes, 24);
    assert_eq!(rp.chunks[0].chunk_index, vec![0, 0]);
    assert_eq!(rp.chunks[1].chunk_index, vec![0, 1]);
}

#[test]
fn plan_query_unknown_dataset_lists_available() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("one.tet");
    let shape = [1u64, 1];
    let mut payload = vec![0u8; 4];
    payload.copy_from_slice(&1.0f32.to_le_bytes());
    write_one_chunk_raw_file(
        &path,
        &OneChunkRawWrite {
            name: "only_me",
            dtype: DTYPE_F32,
            shape: &shape,
            chunk_shape: &shape,
            payload: &payload,
        },
    )
    .unwrap();

    let doc = parse_query_json(r#"{"dataset":"missing"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let cat = plan.catalog.as_ref().unwrap();
    assert!(!cat.matched);
    assert_eq!(
        cat.available_datasets.as_ref().unwrap(),
        &vec!["only_me".to_string()]
    );
    assert!(plan.read_plan.is_none());
}

#[test]
fn plan_query_empty_file_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.tet");
    create_empty_v1_file(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"x"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let cat = plan.catalog.as_ref().unwrap();
    assert!(!cat.matched);
    assert_eq!(
        cat.available_datasets.as_ref().unwrap(),
        &Vec::<String>::new()
    );
    assert!(plan.read_plan.is_none());
}

#[test]
fn plan_query_read_plan_respects_narrow_selection() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("narrow.tet");
    write_multichunk_2x3_tiles(&path, "a");

    let doc = parse_query_json(
        r#"{"dataset":"a","selection":[{"start":0,"stop":2},{"start":2,"stop":3}]}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    assert_eq!(rp.chunk_count, 1);
    assert_eq!(rp.chunks[0].chunk_index, vec![0, 1]);
}

#[test]
fn plan_query_selection_wrong_rank_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rank.tet");
    write_multichunk_2x3_tiles(&path, "a");

    let doc = parse_query_json(r#"{"dataset":"a","selection":[{"start":0,"stop":1}]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let err = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("exactly 2 axes"), "{msg}");
}

#[test]
fn plan_query_f32_preview_full_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prev.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(32)).unwrap();
    let ex = plan.execution.as_ref().expect("preview requested");
    assert_eq!(ex.total_bytes_read_from_disk, 24);
    assert!(!ex.f32_preview_truncated);
    let want = [1f32, 2.0, 4.0, 5.0, 3.0, 6.0];
    assert_eq!(ex.f32_preview.len(), want.len());
    for (a, b) in ex.f32_preview.iter().zip(want.iter()) {
        assert!((a - b).abs() < 1e-5, "got {a} want {b}");
    }
}

#[test]
fn plan_query_f32_preview_narrow_selection() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prev2.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(
        r#"{"dataset":"a","selection":[{"start":0,"stop":2},{"start":2,"stop":3}]}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(8)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.total_bytes_read_from_disk, 8);
    assert_eq!(ex.f32_preview.len(), 2);
    assert!((ex.f32_preview[0] - 3.0).abs() < 1e-5);
    assert!((ex.f32_preview[1] - 6.0).abs() < 1e-5);
}

#[test]
fn materialize_read_plan_f32_decodes_full_planned_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mat.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let (vals, truncated, bytes) = materialize_read_plan_f32_le(&mmap, rp, None).unwrap();
    assert_eq!(bytes, 24);
    assert!(!truncated);
    assert_eq!(vals.len(), 6);
    let want = [1f32, 2.0, 4.0, 5.0, 3.0, 6.0];
    for (a, b) in vals.iter().zip(want.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn materialize_read_plan_f32_cap_truncates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mat2.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let (vals, truncated, _) = materialize_read_plan_f32_le(&mmap, rp, Some(4)).unwrap();
    assert!(truncated);
    assert_eq!(vals.len(), 4);
}
