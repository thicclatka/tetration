//! Integer logical materialize (`i32` / `i64` / `u8` / `u16` / `i16`).

mod core;
mod fold;
#[macro_use]
mod macros;

use std::path::Path;

use crate::query::decode::chunk_decode;
use crate::query::{
    dispatch::accumulate_chunk_read_bytes,
    engine::spill_policy::TempSpillFile,
    types::{ReadPlan, TetError},
};
use crate::utils::{dtype::ElementDtype, i16_le, i32_le, i64_le, u8_le, u16_le, u32_le, u64_le};

use super::parallel;
use super::validate::validate_read_plan_geometry;

use core::{materialize_read_plan_int_le_core, spill_read_plan_int_le_impl};

pub(crate) use fold::IntVisit;
pub(crate) use fold::fold_read_plan_scalar_operation_int;

define_int_materialize! {
    i32;
    backing LogicalI32Backing;
    le_mod i32_le;
    scatter chunk_decode::scatter_chunk_into_plan_i32;
    scatter_seq scatter_fill_sequential_i32;
    scatter_ty ScatterI32Fn;
    scatter_par parallel::materialize_scatter_fill_parallel_i32;
    core_fn materialize_read_plan_i32_le_core;
    read_fn materialize_read_plan_i32_le;
    spill_fn spill_read_plan_i32_le;
    type_label "i32";
    into_vec_fn materialize_into_vec_i32;
    spill_file_fn preview_from_spill_file_i32;
    preview_mat_fn preview_from_materialized_i32;
    as_f64_fn materialized_logical_as_f64_i32;
    promote_inmem |x| f64::from(x);
    promote_spill |x| f64::from(x);
}

define_int_materialize! {
    i64;
    backing LogicalI64Backing;
    le_mod i64_le;
    scatter chunk_decode::scatter_chunk_into_plan_i64;
    scatter_seq scatter_fill_sequential_i64;
    scatter_ty ScatterI64Fn;
    scatter_par parallel::materialize_scatter_fill_parallel_i64;
    core_fn materialize_read_plan_i64_le_core;
    read_fn materialize_read_plan_i64_le;
    spill_fn spill_read_plan_i64_le;
    type_label "i64";
    into_vec_fn materialize_into_vec_i64;
    spill_file_fn preview_from_spill_file_i64;
    preview_mat_fn preview_from_materialized_i64;
    as_f64_fn materialized_logical_as_f64_i64;
    promote_inmem |x| x as f64;
    promote_spill |x| x as f64;
}

define_int_materialize! {
    u8;
    backing LogicalU8Backing;
    le_mod u8_le;
    scatter chunk_decode::scatter_chunk_into_plan_u8;
    scatter_seq scatter_fill_sequential_u8;
    scatter_ty ScatterU8Fn;
    scatter_par parallel::materialize_scatter_fill_parallel_u8;
    core_fn materialize_read_plan_u8_le_core;
    read_fn materialize_read_plan_u8_le;
    spill_fn spill_read_plan_u8_le;
    type_label "u8";
    into_vec_fn materialize_into_vec_u8;
    spill_file_fn preview_from_spill_file_u8;
    preview_mat_fn preview_from_materialized_u8;
    as_f64_fn materialized_logical_as_f64_u8;
    promote_inmem |x| f64::from(x);
    promote_spill |x| f64::from(x);
}

define_int_materialize! {
    u16;
    backing LogicalU16Backing;
    le_mod u16_le;
    scatter chunk_decode::scatter_chunk_into_plan_u16;
    scatter_seq scatter_fill_sequential_u16;
    scatter_ty ScatterU16Fn;
    scatter_par parallel::materialize_scatter_fill_parallel_u16;
    core_fn materialize_read_plan_u16_le_core;
    read_fn materialize_read_plan_u16_le;
    spill_fn spill_read_plan_u16_le;
    type_label "u16";
    into_vec_fn materialize_into_vec_u16;
    spill_file_fn preview_from_spill_file_u16;
    preview_mat_fn preview_from_materialized_u16;
    as_f64_fn materialized_logical_as_f64_u16;
    promote_inmem |x| f64::from(x);
    promote_spill |x| f64::from(x);
}

define_int_materialize! {
    u32;
    backing LogicalU32Backing;
    le_mod u32_le;
    scatter chunk_decode::scatter_chunk_into_plan_u32;
    scatter_seq scatter_fill_sequential_u32;
    scatter_ty ScatterU32Fn;
    scatter_par parallel::materialize_scatter_fill_parallel_u32;
    core_fn materialize_read_plan_u32_le_core;
    read_fn materialize_read_plan_u32_le;
    spill_fn spill_read_plan_u32_le;
    type_label "u32";
    into_vec_fn materialize_into_vec_u32;
    spill_file_fn preview_from_spill_file_u32;
    preview_mat_fn preview_from_materialized_u32;
    as_f64_fn materialized_logical_as_f64_u32;
    promote_inmem |x| f64::from(x);
    promote_spill |x| f64::from(x);
}

define_int_materialize! {
    u64;
    backing LogicalU64Backing;
    le_mod u64_le;
    scatter chunk_decode::scatter_chunk_into_plan_u64;
    scatter_seq scatter_fill_sequential_u64;
    scatter_ty ScatterU64Fn;
    scatter_par parallel::materialize_scatter_fill_parallel_u64;
    core_fn materialize_read_plan_u64_le_core;
    read_fn materialize_read_plan_u64_le;
    spill_fn spill_read_plan_u64_le;
    type_label "u64";
    into_vec_fn materialize_into_vec_u64;
    spill_file_fn preview_from_spill_file_u64;
    preview_mat_fn preview_from_materialized_u64;
    as_f64_fn materialized_logical_as_f64_u64;
    promote_inmem |x| x as f64;
    promote_spill |x| x as f64;
}

define_int_materialize! {
    i16;
    backing LogicalI16Backing;
    le_mod i16_le;
    scatter chunk_decode::scatter_chunk_into_plan_i16;
    scatter_seq scatter_fill_sequential_i16;
    scatter_ty ScatterI16Fn;
    scatter_par parallel::materialize_scatter_fill_parallel_i16;
    core_fn materialize_read_plan_i16_le_core;
    read_fn materialize_read_plan_i16_le;
    spill_fn spill_read_plan_i16_le;
    type_label "i16";
    into_vec_fn materialize_into_vec_i16;
    spill_file_fn preview_from_spill_file_i16;
    preview_mat_fn preview_from_materialized_i16;
    as_f64_fn materialized_logical_as_f64_i16;
    promote_inmem |x| f64::from(x);
    promote_spill |x| f64::from(x);
}

/// Spill a full logical integer selection to `path` (dispatches by `dtype`).
///
/// # Errors
///
/// Returns [`TetError::Validation`] when `dtype` is not an integer wire type, or on spill I/O failure.
pub fn spill_read_plan_int_le(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &Path,
    dtype: ElementDtype,
) -> Result<u64, TetError> {
    match dtype {
        ElementDtype::I32 => spill_read_plan_i32_le(mmap, plan, path),
        ElementDtype::I64 => spill_read_plan_i64_le(mmap, plan, path),
        ElementDtype::U8 => spill_read_plan_u8_le(mmap, plan, path),
        ElementDtype::U16 => spill_read_plan_u16_le(mmap, plan, path),
        ElementDtype::I16 => spill_read_plan_i16_le(mmap, plan, path),
        ElementDtype::U32 => spill_read_plan_u32_le(mmap, plan, path),
        ElementDtype::U64 => spill_read_plan_u64_le(mmap, plan, path),
        _ => Err(TetError::Validation(
            "spill_read_plan_int_le requires an integer dtype (i32/i64/u8/u16/i16/u32/u64)".into(),
        )),
    }
}
