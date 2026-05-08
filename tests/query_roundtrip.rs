use tetration::{parse_query_json, plan_query, validate_query};

#[test]
fn sample_query_parses_and_plans() {
    let json = r#"{
        "layout_version": 1,
        "dataset": "temperature",
        "selection": [
            { "start": 0, "stop": 100, "step": 2 },
            { "start": null, "stop": null, "step": 1 }
        ],
        "operation": { "mean": { "axes": ["time"] } },
        "output": { "preferred": { "inline_json": null } }
    }"#;
    let doc = parse_query_json(json).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query(&doc);
    assert!(plan.accepted);
    assert_eq!(plan.dataset, "temperature");
    assert_eq!(plan.selection_axes, Some(2));
}

#[test]
fn rejects_empty_dataset() {
    let json = r#"{"dataset": "   "}"#;
    let doc = parse_query_json(json).unwrap();
    assert!(validate_query(&doc).is_err());
}
