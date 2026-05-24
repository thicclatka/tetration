//! Slim JSON `plan` format: catalog + `read_plan` summary without per-chunk rows or execution.

use serde::Serialize;

use crate::query::types::{Operation, QueryResponse};

use super::stats::{StatsCatalog, StatsReadPlan, stats_catalog, stats_read_plan};

#[derive(Serialize)]
struct PlanResponse<'a> {
    status: &'static str,
    accepted: bool,
    dataset: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    layout_version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    selection_axes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    operation: Option<&'a Operation>,
    message: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tet_file: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    catalog: Option<StatsCatalog<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    read_plan: Option<StatsReadPlan<'a>>,
}

pub(super) fn format_plan_json(response: &QueryResponse) -> Result<String, String> {
    serde_json::to_string_pretty(&plan_view(response)).map_err(|e| e.to_string())
}

fn plan_view(response: &QueryResponse) -> PlanResponse<'_> {
    let catalog = response.catalog.as_ref().map(stats_catalog);
    let read_plan = response.read_plan.as_ref().map(stats_read_plan);
    PlanResponse {
        status: response.status,
        accepted: response.accepted,
        dataset: &response.dataset,
        layout_version: response.layout_version,
        selection_axes: response.selection_axes,
        operation: response.operation.as_ref(),
        message: &response.message,
        tet_file: response.tet_file.as_deref(),
        catalog,
        read_plan,
    }
}
