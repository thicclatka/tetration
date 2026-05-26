//! Chunk-streaming partial-axis reductions (no full logical tensor allocation).

#![allow(clippy::too_many_arguments)]

mod fields;
mod float;
mod int;

pub(crate) use fields::{partial_arg_fields, partial_fields};
pub(crate) use float::{fold_read_plan_partial_operation, fold_read_plan_partial_operation_f64};
pub(crate) use int::fold_read_plan_partial_operation_int;
