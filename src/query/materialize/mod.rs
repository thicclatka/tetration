//! Decode planned chunk payloads into logical row-major tensors.

pub(crate) mod covariance;
mod f16;
mod f32;
mod f64;
pub mod int;
mod logical;
pub mod parallel;
mod selection;
pub(crate) mod shared;
pub mod stats;
mod types;
mod validate;

pub use f16::{materialize_read_plan_f16_le, spill_read_plan_f16_le};
pub use f32::{
    MaterializeReadPlanF32IntoOutcome, materialize_read_plan_f32_le,
    materialize_read_plan_f32_le_into, spill_read_plan_f32_le,
};
pub use f64::{materialize_read_plan_f64_le, spill_read_plan_f64_le};

pub(crate) use f16::fold_read_plan_scalar_operation_f16;
pub(crate) use f32::fold_read_plan_scalar_operation;
pub(crate) use f64::fold_read_plan_scalar_operation_f64;
pub(crate) use logical::{MaterializedLogical, materialized_logical_as_f64};
pub(crate) use selection::{
    materialize_logical_selection, preview_from_materialized, preview_from_spill_export_file,
};
pub(crate) use types::{DecodePreviewBundle, LogicalF32Backing, LogicalF64Backing};

pub(crate) use f32::{materialize_read_plan_f32_le_core, materialize_read_plan_f32_le_into_core};
pub(crate) use f64::materialize_read_plan_f64_le_core;
pub(crate) use validate::validate_read_plan_geometry;
