use tetration::{
    CHUNK_PAYLOAD_CODEC_V1, CHUNK_TOUCH_POLICY, DTYPE_F32, RawArrayWrite, mmap_file_read,
    parse_query_json, plan_query_with_tet_mmap, validate_query, write_raw_array_file,
};

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
            dtype: DTYPE_F32,
            shape: &shape,
            chunk_shape: &chunk_shape,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: &data,
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
