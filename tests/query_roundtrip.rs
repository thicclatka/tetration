use tetration::{parse_query_json, plan_query_empty, validate_query};

#[test]
fn sample_query_parses_and_plans() {
    let json = r#"{
        "layout_version": 1,
        "dataset": "temperature",
        "selection": [
            { "start": 0, "stop": 100, "step": 2 },
            { "start": null, "stop": null, "step": 1 }
        ],
        "operation": { "mean": { "axes": [] } },
        "output": { "preferred": { "inline_json": null } }
    }"#;
    let doc = parse_query_json(json).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_empty(&doc);
    assert!(plan.accepted);
    assert_eq!(plan.dataset, "temperature");
    assert_eq!(plan.selection_axes, Some(2));
}

#[test]
fn rejects_invalid_operation_axis_token() {
    let json = r#"{"dataset":"a","operation":{"sum":{"axes":["x"]}}}"#;
    let doc = parse_query_json(json).unwrap();
    let err = validate_query(&doc).unwrap_err();
    assert!(err.to_string().contains("decimal"), "{err}");
}

#[test]
fn accepts_decimal_operation_axis_indices() {
    let json = r#"{"dataset":"a","operation":{"sum":{"axes":["0"]}}}"#;
    let doc = parse_query_json(json).unwrap();
    validate_query(&doc).unwrap();
}

#[test]
fn accepts_min_max_count_operations() {
    for json in [
        r#"{"dataset":"a","operation":{"min":{"axes":[]}}}"#,
        r#"{"dataset":"a","operation":{"max":{"axes":["1"]}}}"#,
        r#"{"dataset":"a","operation":{"count":{"axes":[]}}}"#,
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
