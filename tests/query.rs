//! Query engine integration tests: JSON validation, mmap planning, materialize, operations.

mod fixture;

use std::path::{Path, PathBuf};

use fixture::{
    CHUNK_2X2, SHAPE_2X3, write_multichunk_2x3_f64_tiles, write_multichunk_2x3_tiles,
    write_multichunk_2x3_zero_zstd,
};
use tetration::{
    CHUNK_PAYLOAD_CODEC_V1, CHUNK_TOUCH_POLICY, DATASET_DTYPE_TAG_V1, OneChunkRawWrite,
    RawArrayWrite, SpillPathAllowlist, TempSpillFile, create_empty_v1_file,
    materialize_read_plan_f32_le, materialize_read_plan_f32_le_into,
    materialize_read_plan_f32_le_into_parallel, materialize_read_plan_f32_le_parallel,
    mmap_file_read, parse_query_json, plan_query_empty, plan_query_with_tet_mmap,
    plan_query_with_tet_mmap_ex, read_tet_summary_v1, validate_query, write_one_chunk_raw_file,
    write_raw_array_file,
};

/// JSON string literal for a filesystem path embedded in test query fixtures.
fn json_path_handle(path: &Path) -> String {
    let s = path.display().to_string();
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

// --- JSON / plan-only ---

#[test]
fn sample_query_parses_and_plans() {
    let json = r#"{
        "layout_version": 1,
        "dataset": "temperature",
        "selection": [
            { "start": 0, "stop": 100, "step": 2 },
            { "start": null, "stop": null, "step": 1 }
        ],
        "mean":[]
    }"#;
    let doc = parse_query_json(json).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_empty(&doc);
    assert!(plan.accepted);
    assert_eq!(plan.dataset, "temperature");
    assert_eq!(plan.selection_axes, Some(2));
}

#[test]
fn rejects_nested_operation_object() {
    let json = r#"{"dataset":"a","operation":{"mean":{"axes":[]}}}"#;
    let err = parse_query_json(json).unwrap_err();
    assert!(err.to_string().contains("operation"), "{err}");
}

#[test]
fn rejects_nested_output_object() {
    let json = r#"{"dataset":"a","output":{"preferred":{"spill_array":{"handle":"out.bin"}}}}"#;
    let err = parse_query_json(json).unwrap_err();
    assert!(err.to_string().contains("output"), "{err}");
}

#[test]
fn parses_flat_spill_roundtrip() {
    let json = r#"{"dataset":"a","spill":"slice.bin"}"#;
    let doc = parse_query_json(json).unwrap();
    validate_query(&doc).unwrap();
    let out = doc.output.as_ref().unwrap();
    assert!(matches!(
        out.preferred,
        Some(tetration::OutputHint::SpillArray { ref handle }) if handle == "slice.bin"
    ));
    let roundtrip = serde_json::to_string(&doc).unwrap();
    assert!(roundtrip.contains(r#""spill":"slice.bin""#));
    assert!(!roundtrip.contains("output"));
}

#[test]
fn parses_flat_mean_on_axis_zero() {
    let json = r#"{
        "dataset": "temperature",
        "mean": 0,
        "execution": { "memory_budget_percent": 40 }
    }"#;
    let doc = parse_query_json(json).unwrap();
    validate_query(&doc).unwrap();
    let op = doc.operation.as_ref().unwrap();
    assert!(matches!(op, tetration::Operation::Mean { axes } if axes.as_slice() == ["0"]));
    assert_eq!(
        doc.execution.as_ref().unwrap().memory_budget_percent_bps,
        Some(4000)
    );
    let roundtrip = serde_json::to_string_pretty(&doc).unwrap();
    assert!(roundtrip.contains(r#""mean": 0"#));
    assert!(roundtrip.contains(r#""memory_budget_percent": 40"#));
    assert!(!roundtrip.contains("operation"));
}

#[test]
fn rejects_invalid_operation_axis_token() {
    let json = r#"{"dataset":"a","sum":"x"}"#;
    let err = parse_query_json(json).unwrap_err();
    assert!(err.to_string().contains("decimal"), "{err}");
}

#[test]
fn accepts_decimal_operation_axis_indices() {
    let json = r#"{"dataset":"a","sum":0}"#;
    let doc = parse_query_json(json).unwrap();
    validate_query(&doc).unwrap();
}

#[test]
fn accepts_min_max_count_operations() {
    for json in [
        r#"{"dataset":"a","min":[]}"#,
        r#"{"dataset":"a","max":1}"#,
        r#"{"dataset":"a","count":[]}"#,
        r#"{"dataset":"a","var":[]}"#,
        r#"{"dataset":"a","std":0}"#,
        r#"{"dataset":"a","product":[]}"#,
        r#"{"dataset":"a","median":[]}"#,
        r#"{"dataset":"a","quantile":{"q":0.5}}"#,
        r#"{"dataset":"a","histogram":{"bins":4}}"#,
    ] {
        let doc = parse_query_json(json).unwrap();
        validate_query(&doc).unwrap();
    }
}

#[test]
fn rejects_empty_dataset() {
    let json = r#"{"dataset": "   "}"#;
    let doc = parse_query_json(json).unwrap();
    assert!(validate_query(&doc).is_err());
}

#[test]
fn rejects_unknown_query_fields() {
    let json = r#"{"dataset":"a","extra":1}"#;
    let err = parse_query_json(json).unwrap_err();
    assert!(err.to_string().contains("unknown"), "{err}");
}

#[test]
fn rejects_oversized_query_json() {
    let limits = tetration::QueryLimits::DEFAULT;
    let pad = "x".repeat(limits.max_json_bytes);
    let json = format!(r#"{{"dataset":"{pad}"}}"#);
    let err = parse_query_json(&json).unwrap_err();
    assert!(err.to_string().contains("maximum size"), "{err}");
}

#[test]
fn rejects_oversized_dataset_name() {
    let limits = tetration::QueryLimits::DEFAULT;
    let name = "a".repeat(limits.max_dataset_name_len + 1);
    let json = format!(r#"{{"dataset":"{name}"}}"#);
    let doc = parse_query_json(&json).unwrap();
    let err = validate_query(&doc).unwrap_err();
    assert!(err.to_string().contains("dataset"), "{err}");
}

#[test]
fn rejects_selection_rank_above_max_ndim() {
    use tetration::MAX_NDIM;
    let slices = (0..=MAX_NDIM)
        .map(|_| r#"{ "start": 0, "stop": 1 }"#)
        .collect::<Vec<_>>()
        .join(",");
    let json = format!(r#"{{"dataset":"a","selection":[{slices}]}}"#);
    let doc = parse_query_json(&json).unwrap();
    let err = validate_query(&doc).unwrap_err();
    assert!(err.to_string().contains("selection rank"), "{err}");
}

#[test]
fn rejects_deeply_nested_query_json() {
    let limits = tetration::QueryLimits::DEFAULT;
    let mut inner = "null".to_string();
    for _ in 0..=limits.max_json_depth {
        inner = format!(r#"{{"x":{inner}}}"#);
    }
    let json = format!(r#"{{"dataset":"a","junk":{inner}}}"#);
    let err = parse_query_json(&json).unwrap_err();
    assert!(err.to_string().contains("nesting depth"), "{err}");
}

// --- strided read plan ---

#[test]
fn read_plan_strided_step_touches_fewer_chunks_than_dense() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strided.tet");
    let shape = [4u64, 3];
    let chunk_shape = [2u64, 3];
    let mut data = vec![0u8; 4 * 12];
    for (i, slot) in data.chunks_exact_mut(4).enumerate() {
        let v = (i + 1) as f32;
        slot.copy_from_slice(&v.to_le_bytes());
    }
    write_raw_array_file(
        &path,
        &RawArrayWrite {
            name: "a",
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape: &shape,
            chunk_shape: &chunk_shape,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: &data,
            file_execution: None,
        },
    )
    .unwrap();

    let mmap = mmap_file_read(&path).unwrap();

    let doc_dense = parse_query_json(
        r#"{"dataset":"a","selection":[{"start":1,"stop":3},{"start":0,"stop":3}]}"#,
    )
    .unwrap();
    validate_query(&doc_dense).unwrap();
    let dense = plan_query_with_tet_mmap(&doc_dense, None, &mmap, None).unwrap();
    assert_eq!(
        dense.read_plan.as_ref().unwrap().chunk_touch_policy,
        CHUNK_TOUCH_POLICY.dense_half_open_unit_step
    );
    assert_eq!(dense.read_plan.as_ref().unwrap().chunk_count, 2);

    let doc_strided = parse_query_json(
        r#"{"dataset":"a","selection":[{"start":1,"stop":3,"step":2},{"start":0,"stop":3}]}"#,
    )
    .unwrap();
    validate_query(&doc_strided).unwrap();
    let strided = plan_query_with_tet_mmap(&doc_strided, None, &mmap, None).unwrap();
    assert_eq!(
        strided.read_plan.as_ref().unwrap().chunk_touch_policy,
        CHUNK_TOUCH_POLICY.strided_half_open
    );
    assert_eq!(strided.read_plan.as_ref().unwrap().chunk_count, 1);
    assert_eq!(
        strided.read_plan.as_ref().unwrap().chunks[0].chunk_index,
        vec![0, 0]
    );
}

// --- mmap planning, materialize, operations ---

#[test]
fn plan_query_f32_preview_zstd_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zstd_q.tet");
    write_multichunk_2x3_zero_zstd(&path, "temperature");
    let doc = parse_query_json(r#"{"dataset":"temperature","layout_version":1}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(32)).unwrap();
    let ex = plan.execution.as_ref().expect("preview requested");
    let s = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(
        ex.total_bytes_read_from_disk,
        s.chunks[0].stored_byte_len + s.chunks[1].stored_byte_len
    );
    assert!(!ex.f32_preview_truncated);
    assert!(ex.f32_preview.iter().all(|&x| x == 0.0));
}

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
            dtype: DATASET_DTYPE_TAG_V1.f32,
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
    let want = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0];
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
    let want = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0];
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
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn plan_query_operation_requires_explicit_preview_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("op_req.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    assert!(plan_query_with_tet_mmap(&doc, None, &mmap, None).is_err());
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert!(ex.f32_preview.is_empty());
    assert_eq!(ex.operation_element_count, Some(6));
    assert!((ex.operation_sum.unwrap() - 21.0).abs() < 1e-5);
}

#[test]
fn plan_query_preview_cap_zero_without_operation_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prev0.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    assert!(plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).is_err());
}

#[test]
fn plan_query_operation_min_max_count_scalar() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("minmax.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();

    let min_doc = parse_query_json(r#"{"dataset":"a","min":[]}"#).unwrap();
    validate_query(&min_doc).unwrap();
    let min_plan = plan_query_with_tet_mmap(&min_doc, None, &mmap, Some(8)).unwrap();
    assert!((min_plan.execution.as_ref().unwrap().operation_min.unwrap() - 1.0).abs() < 1e-5);

    let max_doc = parse_query_json(r#"{"dataset":"a","max":[]}"#).unwrap();
    validate_query(&max_doc).unwrap();
    let max_plan = plan_query_with_tet_mmap(&max_doc, None, &mmap, Some(8)).unwrap();
    assert!((max_plan.execution.as_ref().unwrap().operation_max.unwrap() - 6.0).abs() < 1e-5);

    let count_doc = parse_query_json(r#"{"dataset":"a","count":[]}"#).unwrap();
    validate_query(&count_doc).unwrap();
    let count_plan = plan_query_with_tet_mmap(&count_doc, None, &mmap, Some(0)).unwrap();
    let ex = count_plan.execution.as_ref().unwrap();
    assert_eq!(ex.operation_element_count, Some(6));
    assert!(ex.f32_preview.is_empty());
}

#[test]
fn plan_query_operation_var_std_scalar_and_partial() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("varstd.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();

    let var_doc = parse_query_json(r#"{"dataset":"a","var":[]}"#).unwrap();
    validate_query(&var_doc).unwrap();
    let var_plan = plan_query_with_tet_mmap(&var_doc, None, &mmap, Some(0)).unwrap();
    let var_ex = var_plan.execution.as_ref().unwrap();
    // population var of 1..6
    assert!((var_ex.operation_var.unwrap() - 2.916_666_7).abs() < 1e-5);

    let std_doc = parse_query_json(r#"{"dataset":"a","std":[]}"#).unwrap();
    validate_query(&std_doc).unwrap();
    let std_plan = plan_query_with_tet_mmap(&std_doc, None, &mmap, Some(0)).unwrap();
    let std_ex = std_plan.execution.as_ref().unwrap();
    assert!((std_ex.operation_std.unwrap() - 1.707_825_1).abs() < 1e-5);

    let partial_doc = parse_query_json(r#"{"dataset":"a","var":0}"#).unwrap();
    validate_query(&partial_doc).unwrap();
    let partial_plan = plan_query_with_tet_mmap(&partial_doc, None, &mmap, Some(4)).unwrap();
    let partial_ex = partial_plan.execution.as_ref().unwrap();
    assert_eq!(
        partial_ex.operation_reduced_shape.as_deref(),
        Some(&[3u64][..])
    );
    let vars = partial_ex.operation_reduced_var.as_ref().unwrap();
    assert_eq!(vars.len(), 3);
    for v in vars {
        assert!((*v - 2.25).abs() < 1e-5, "expected 2.25, got {v}");
    }
}

#[test]
fn plan_query_operation_product_scalar_and_partial() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("product.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();

    let doc = parse_query_json(r#"{"dataset":"a","product":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    assert!((plan.execution.as_ref().unwrap().operation_product.unwrap() - 720.0).abs() < 1e-5);

    let partial_doc = parse_query_json(r#"{"dataset":"a","product":0}"#).unwrap();
    validate_query(&partial_doc).unwrap();
    let partial_plan = plan_query_with_tet_mmap(&partial_doc, None, &mmap, Some(4)).unwrap();
    let ex = partial_plan.execution.as_ref().unwrap();
    assert_eq!(ex.operation_reduced_shape.as_deref(), Some(&[3u64][..]));
    let products = ex.operation_reduced_product.as_ref().unwrap();
    assert_eq!(products.len(), 3);
    assert!((products[0] - 4.0).abs() < 1e-5);
    assert!((products[1] - 10.0).abs() < 1e-5);
    assert!((products[2] - 18.0).abs() < 1e-5);
}

#[test]
fn plan_query_operation_min_along_axis_zero() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("min_axis0.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","min":0}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(4)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.operation_reduced_shape.as_deref(), Some(&[3u64][..]));
    let mins = ex.operation_reduced_min.as_ref().unwrap();
    assert_eq!(mins.len(), 3);
    assert!((mins[0] - 1.0).abs() < 1e-5);
    assert!((mins[1] - 2.0).abs() < 1e-5);
    assert!((mins[2] - 3.0).abs() < 1e-5);
}

#[test]
fn plan_query_scalar_sum_fold_matches_materialize_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fold_sum.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(8)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let (full, _, _) = materialize_read_plan_f32_le(&mmap, rp, None).unwrap();
    let manual_sum: f64 = full.iter().map(|&x| f64::from(x)).sum();
    assert_eq!(ex.operation_element_count, Some(6));
    assert!((ex.operation_sum.unwrap() - manual_sum).abs() < 1e-5);
    assert_eq!(ex.f32_preview.len(), 6);
    assert!(!ex.f32_preview_truncated);
}

#[test]
fn plan_query_operation_sum_full_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("op_sum.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(64)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.operation_element_count, Some(6));
    assert!((ex.operation_sum.unwrap() - 21.0).abs() < 1e-5);
    assert!(ex.operation_mean.is_none());
    assert!(!ex.f32_preview_truncated);
}

#[test]
fn plan_query_operation_mean_narrow_selection() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("op_mean.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(
        r#"{"dataset":"a","mean":[],"selection":[{"start":0,"stop":2},{"start":2,"stop":3}]}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(8)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.operation_element_count, Some(2));
    assert!((ex.operation_mean.unwrap() - 4.5).abs() < 1e-5);
    assert!(ex.operation_sum.is_none());
}

#[test]
fn plan_query_operation_sum_preview_truncated_but_aggregate_full() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("op_trunc.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(2)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert!(ex.f32_preview_truncated);
    assert_eq!(ex.f32_preview.len(), 2);
    assert_eq!(ex.operation_element_count, Some(6));
    assert!((ex.operation_sum.unwrap() - 21.0).abs() < 1e-5);
}

#[test]
fn plan_query_operation_sum_along_axis_zero() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("op_axis0.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","sum":0}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(4)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.operation_reduced_shape.as_deref(), Some(&[3u64][..]));
    let sums = ex.operation_reduced_sum.as_ref().unwrap();
    assert_eq!(sums.len(), 3);
    assert!((sums[0] - 5.0).abs() < 1e-5);
    assert!((sums[1] - 7.0).abs() < 1e-5);
    assert!((sums[2] - 9.0).abs() < 1e-5);
    assert!(ex.operation_sum.is_none());
}

#[test]
fn plan_query_execute_multi_chunk_matches_parallel_materialize() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("exec_par.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(64)).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    assert!(rp.chunk_count > 1, "fixture must touch multiple chunks");
    let ex = plan.execution.as_ref().unwrap();
    let (par, par_trunc, par_bytes) =
        materialize_read_plan_f32_le_parallel(&mmap, rp, Some(64)).unwrap();
    assert_eq!(ex.f32_preview, par);
    assert_eq!(ex.f32_preview_truncated, par_trunc);
    assert_eq!(ex.total_bytes_read_from_disk, par_bytes);
}

#[test]
fn materialize_read_plan_f32_parallel_matches_sequential() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("par.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let (seq, seq_trunc, seq_bytes) = materialize_read_plan_f32_le(&mmap, rp, None).unwrap();
    let (par, par_trunc, par_bytes) =
        materialize_read_plan_f32_le_parallel(&mmap, rp, None).unwrap();
    assert_eq!(seq, par);
    assert_eq!(seq_trunc, par_trunc);
    assert_eq!(seq_bytes, par_bytes);

    let mut seq_buf = vec![0.0f32; 6];
    let mut par_buf = vec![0.0f32; 6];
    let seq_into = materialize_read_plan_f32_le_into(&mmap, rp, None, &mut seq_buf).unwrap();
    let par_into =
        materialize_read_plan_f32_le_into_parallel(&mmap, rp, None, &mut par_buf).unwrap();
    assert_eq!(
        seq_into.logical_element_count,
        par_into.logical_element_count
    );
    assert_eq!(seq_into.elements_written, par_into.elements_written);
    assert_eq!(seq_into.truncated, par_into.truncated);
    assert_eq!(
        seq_into.total_bytes_read_from_disk,
        par_into.total_bytes_read_from_disk
    );
    assert_eq!(seq_buf, par_buf);
}

#[test]
fn materialize_read_plan_f32_parallel_zstd_matches_sequential() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("par_zstd.tet");
    write_multichunk_2x3_zero_zstd(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let (seq, _, seq_bytes) = materialize_read_plan_f32_le(&mmap, rp, None).unwrap();
    let (par, _, par_bytes) = materialize_read_plan_f32_le_parallel(&mmap, rp, None).unwrap();
    assert_eq!(seq, par);
    assert_eq!(seq_bytes, par_bytes);
    assert!(seq.iter().all(|&x| x == 0.0));
}

#[test]
fn materialize_read_plan_f32_le_into_matches_vec() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("into.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let (want, _, _) = materialize_read_plan_f32_le(&mmap, rp, None).unwrap();
    let mut buf = vec![0.0f32; 6];
    let out = materialize_read_plan_f32_le_into(&mmap, rp, None, &mut buf).unwrap();
    assert_eq!(out.logical_element_count, 6);
    assert_eq!(out.elements_written, 6);
    assert!(!out.truncated);
    assert_eq!(buf, want);
}

// --- JSON hardening property tests ---

use proptest::prelude::*;

// --- file execution settings / memory budget ---

#[test]
fn file_execution_settings_roundtrip_in_index_header() {
    use tetration::{FileExecutionSettingsV1, read_tet_summary_v1};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("exec.tet");
    let settings = FileExecutionSettingsV1 {
        memory_budget_percent_bps: 5000,
        memory_budget_bytes: 64 * 1024 * 1024,
    };
    fixture::write_multichunk_2x3_with_execution(&path, "t", settings);
    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.file_execution, settings);
}

#[test]
fn query_execution_reports_dataset_and_budget_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "temperature");
    let mmap = mmap_file_read(&path).unwrap();
    let json = r#"{"dataset":"temperature","execution":{"memory_budget_bytes":999999}}"#;
    let doc = parse_query_json(json).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(4)).unwrap();
    let catalog = resp.catalog.as_ref().unwrap();
    assert_eq!(catalog.dataset_f32_bytes, Some(24));
    assert_eq!(
        catalog.file_execution,
        Some(tetration::FileExecutionSettingsV1::default_engine())
    );
    let exec = resp.execution.as_ref().unwrap();
    assert_eq!(exec.memory_budget_bytes, Some(999_999));
    assert_eq!(exec.logical_selection_f32_bytes, Some(24));
}

#[test]
fn plan_query_operation_argmin_argmax_scalar_and_partial() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a","arg_min":[]}"#).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(64)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.operation_argmin_index, Some(0));
    assert_eq!(ex.memory_strategy, Some("streaming_fold"));

    let doc = parse_query_json(r#"{"dataset":"a","arg_max":0}"#).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(64)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.operation_reduced_argmax, Some(vec![1, 1, 1]));
}

#[test]
fn spill_path_allowlist_rejects_outside_root() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let outside_dir = tempfile::tempdir().unwrap();
    let policy = SpillPathAllowlist::default_for_tet(&path, std::iter::empty::<PathBuf>()).unwrap();
    let spill_out = outside_dir.path().join("out.bin");
    let json = format!(
        r#"{{"dataset":"a","spill":{}}}"#,
        json_path_handle(&spill_out)
    );
    let doc = parse_query_json(&json).unwrap();
    let err = plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(2), Some(&policy)).unwrap_err();
    assert!(err.to_string().contains("allowed root"), "{err}");

    let json = r#"{"dataset":"a","spill":"out.bin"}"#;
    let doc = parse_query_json(json).unwrap();
    let resp = plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(2), Some(&policy)).unwrap();
    let resp_zero_preview =
        plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(0), Some(&policy)).unwrap();
    assert!(
        resp_zero_preview
            .execution
            .as_ref()
            .unwrap()
            .spill_f32_path
            .is_some()
    );
    assert!(
        resp.execution
            .as_ref()
            .unwrap()
            .spill_f32_path
            .as_ref()
            .unwrap()
            .ends_with("out.bin")
    );

    let allow = dir.path().join("allowed");
    std::fs::create_dir_all(&allow).unwrap();
    let policy = SpillPathAllowlist::default_for_tet(&path, [allow.clone()]).unwrap();
    let spill_ok = allow.join("out.bin");
    let json = format!(
        r#"{{"dataset":"a","spill":{}}}"#,
        json_path_handle(&spill_ok)
    );
    let doc = parse_query_json(&json).unwrap();
    let resp = plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(2), Some(&policy)).unwrap();
    let spilled = resp
        .execution
        .as_ref()
        .unwrap()
        .spill_f32_path
        .as_ref()
        .unwrap();
    assert!(
        spilled.ends_with("allowed/out.bin") || spilled.ends_with("allowed\\out.bin"),
        "{spilled}"
    );
}

#[test]
fn capped_preview_materialize_allocates_only_cap_not_full_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(2)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.f32_preview.len(), 2);
    assert!(ex.f32_preview_truncated);
    assert_eq!(ex.memory_strategy, Some("capped_in_memory"));
}

#[test]
fn plan_query_operation_median_scalar_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json = r#"{"dataset":"a","median":[],"execution":{"memory_budget_bytes":999999}}"#;
    let doc = parse_query_json(json).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(64)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert!((ex.operation_median.unwrap() - 3.5).abs() < 1e-5);
    assert_eq!(ex.operation_element_count, Some(6));
    assert_eq!(ex.memory_strategy, Some("in_memory_materialize"));
}

#[test]
fn plan_query_operation_median_scalar_temp_spill_when_over_budget() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    // 8 bytes = 2 f32; logical selection is 6 f32 (24 bytes).
    let json = r#"{"dataset":"a","median":[],"execution":{"memory_budget_bytes":8}}"#;
    let doc = parse_query_json(json).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(2)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert!((ex.operation_median.unwrap() - 3.5).abs() < 1e-5);
    assert_eq!(ex.memory_strategy, Some("temp_spill_materialize"));
    assert!(ex.spill_f32_path.is_none());
}

#[test]
fn plan_query_i32_preview_and_sum() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i32.tet");
    fixture::write_multichunk_2x3_i32_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json = r#"{"dataset":"a","sum":[]}"#;
    let doc = parse_query_json(json).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(4)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.i32_preview.len(), 4);
    assert!(ex.i32_preview_truncated);
    assert_eq!(ex.f32_preview.len(), 0);
    assert_eq!(ex.operation_sum.unwrap(), 21.0);
    assert_eq!(resp.catalog.as_ref().unwrap().dataset_i32_bytes, Some(24));
}

#[test]
fn plan_query_i64_preview_materialize() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64.tet");
    fixture::write_multichunk_2x3_i64_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(6)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.i64_preview, vec![1, 2, 3, 4, 5, 6]);
    assert!(!ex.i64_preview_truncated);
    assert_eq!(resp.catalog.as_ref().unwrap().dataset_i64_bytes, Some(48));
}

#[test]
fn plan_query_f64_preview_and_sum() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f64.tet");
    write_multichunk_2x3_f64_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json = r#"{"dataset":"a","sum":[]}"#;
    let doc = parse_query_json(json).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(4)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.f64_preview.len(), 4);
    assert!(ex.f64_preview_truncated);
    assert_eq!(ex.f32_preview.len(), 0);
    assert!((ex.operation_sum.unwrap() - 21.0).abs() < 1e-9);
    assert_eq!(resp.catalog.as_ref().unwrap().dataset_f64_bytes, Some(48));
}

#[test]
fn plan_query_operation_median_partial_axis_0() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json = r#"{"dataset":"a","median":0,"execution":{"memory_budget_bytes":999999}}"#;
    let doc = parse_query_json(json).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(0)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.operation_reduced_shape, Some(vec![3]));
    let medians = ex.operation_reduced_median.as_ref().unwrap();
    assert_eq!(medians.len(), 3);
    assert!((medians[0] - 2.5).abs() < 1e-5);
    assert!((medians[1] - 3.5).abs() < 1e-5);
    assert!((medians[2] - 4.5).abs() < 1e-5);
}

#[test]
fn plan_query_operation_quantile_scalar() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json =
        r#"{"dataset":"a","quantile":{"q":0.95},"execution":{"memory_budget_bytes":999999}}"#;
    let doc = parse_query_json(json).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(0)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert!((ex.operation_quantile.unwrap() - 5.75).abs() < 1e-5);
}

#[test]
fn plan_query_operation_histogram_scalar() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json =
        r#"{"dataset":"a","histogram":{"bins":3},"execution":{"memory_budget_bytes":999999}}"#;
    let doc = parse_query_json(json).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(0)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    let counts = ex.operation_histogram_counts.as_ref().unwrap();
    let edges = ex.operation_histogram_edges.as_ref().unwrap();
    assert_eq!(counts.len(), 3);
    assert_eq!(edges.len(), 4);
    assert_eq!(counts.iter().sum::<f64>() as u32, 6);
}

#[test]
fn plan_query_f64_median_temp_spill_when_over_budget() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f64.tet");
    write_multichunk_2x3_f64_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json = r#"{"dataset":"a","median":[],"execution":{"memory_budget_bytes":16}}"#;
    let doc = parse_query_json(json).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(2)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert!((ex.operation_median.unwrap() - 3.5).abs() < 1e-5);
    assert_eq!(ex.memory_strategy, Some("temp_spill_materialize"));
}

// --- spill policy (from src/query/engine/spill_policy.rs) ---

#[test]
fn spill_path_allowlist_default_for_tet_includes_parent() {
    let dir = tempfile::tempdir().unwrap();
    let tet = dir.path().join("data.tet");
    std::fs::write(&tet, b"x").unwrap();
    let policy = SpillPathAllowlist::default_for_tet(&tet, std::iter::empty::<PathBuf>()).unwrap();
    let resolved = policy.validate(Path::new("spill.bin")).unwrap();
    assert!(resolved.starts_with(std::fs::canonicalize(dir.path()).unwrap()));
}

#[test]
fn temp_spill_file_deleted_on_drop() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("spill-test.bin");
    std::fs::write(&file_path, b"x").unwrap();
    {
        let guard = TempSpillFile::with_path_for_test(file_path.clone());
        assert!(file_path.exists());
        drop(guard);
    }
    assert!(!file_path.exists());
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        ..ProptestConfig::default()
    })]

    #[test]
    fn parse_query_json_never_panics(raw in "\\PC{0,4096}") {
        let _ = parse_query_json(&raw);
    }

    #[test]
    fn validate_query_never_panics_after_parse(raw in "\\PC{0,4096}") {
        if let Ok(doc) = parse_query_json(&raw) {
            let _ = validate_query(&doc);
        }
    }
}
