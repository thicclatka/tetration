//! Parallel chunk-streaming scalar and partial-axis folds (Rayon).

mod merge;
mod partial;
mod preview;
mod scalar;
mod workers;

pub(crate) use partial::{
    fold_read_plan_partial_operation_f64_parallel, fold_read_plan_partial_operation_parallel,
};
pub(crate) use scalar::{
    fold_read_plan_scalar_operation_f16_parallel, fold_read_plan_scalar_operation_f64_parallel,
    fold_read_plan_scalar_operation_int_parallel, fold_read_plan_scalar_operation_parallel,
};
pub(crate) use workers::use_parallel_fold;
