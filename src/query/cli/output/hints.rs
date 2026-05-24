//! stderr hints after successful `tet query` (catalog miss, etc.).

use std::fmt::Write as _;

use crate::query::types::QueryResponse;

/// Human-oriented stderr when the dataset name is missing from the `.tet` catalog.
#[must_use]
pub fn format_catalog_miss_hint(response: &QueryResponse) -> Option<String> {
    let catalog = response.catalog.as_ref()?;
    if catalog.matched {
        return None;
    }
    let mut out = String::new();
    let _ = writeln!(
        out,
        "hint: dataset {:?} not found in this .tet",
        response.dataset
    );
    if let Some(names) = catalog.available_datasets.as_ref() {
        out.push_str("available datasets:\n");
        for name in names {
            let _ = writeln!(out, "  {name}");
        }
        if let Some(tet) = response.tet_file.as_deref() {
            let _ = writeln!(out, "tip: run `tet info {tet}` to list catalog datasets");
        } else {
            let _ = writeln!(
                out,
                "tip: run `tet info <path.tet>` to list catalog datasets"
            );
        }
    }
    Some(out)
}
