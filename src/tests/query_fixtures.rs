//! `fixtures/queries/` JSON/TOML pairs parse to equivalent documents.

use super::fixture::query_files;
use crate::query::{Operation, validate_query};

#[test]
fn query_fixture_pairs_json_and_toml_match() {
    for stem in [
        "mean_temperature",
        "mean_strided_temperature",
        "slice_full_temperature",
        "slice_2x2_temperature",
        "mean_a",
        "sum_a",
        "sum_axis0_a",
        "var_a",
        "quantile_axis0_a",
    ] {
        let json = query_files::json(stem);
        let toml = query_files::toml(stem);
        validate_query(&json).unwrap();
        validate_query(&toml).unwrap();
        assert_eq!(json.dataset, toml.dataset, "{stem}");
        assert_eq!(json.layout_version, toml.layout_version, "{stem}");
        assert_eq!(
            json.selection.as_ref().map(|v| v.len()),
            toml.selection.as_ref().map(|v| v.len()),
            "{stem}"
        );
        if let (Some(ja), Some(ta)) = (&json.selection, &toml.selection) {
            assert_eq!(ja.len(), ta.len(), "{stem}");
            for (j, t) in ja.iter().zip(ta) {
                assert_eq!(j.start, t.start, "{stem} start");
                assert_eq!(j.stop, t.stop, "{stem} stop");
                assert_eq!(j.step, t.step, "{stem} step");
            }
        }
        assert_eq!(
            json.operation.as_ref().map(op_tag),
            toml.operation.as_ref().map(op_tag),
            "{stem}"
        );
    }
}

fn op_tag(op: &Operation) -> &'static str {
    op.wire_key()
}
