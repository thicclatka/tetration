//! Shared integer logical materialize and spill helpers.

use std::path::Path;

use memmap2::{Mmap, MmapMut};

use crate::query::types::{ReadPlan, TetError};

use super::super::validate::validate_full_read_plan_buffer;

fn check_materialized_complete_option<T>(out: &[Option<T>]) -> Result<(), TetError> {
    if out.iter().any(Option::is_none) {
        return Err(TetError::Validation(
            "materialized selection has unset elements (chunk payloads vs selection mismatch)"
                .into(),
        ));
    }
    Ok(())
}

pub(crate) fn finalize_option<T: Copy + Default>(out: Vec<Option<T>>) -> Result<Vec<T>, TetError> {
    check_materialized_complete_option(&out)?;
    Ok(out.into_iter().map(Option::unwrap_or_default).collect())
}

pub(crate) fn materialize_read_plan_int_le_core<T, F>(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    scatter_fill: F,
) -> Result<(Vec<T>, bool, u64), TetError>
where
    T: Copy + Default,
    F: Fn(&[u8], &ReadPlan, &mut [Option<T>]) -> Result<u64, TetError>,
{
    if matches!(max_elements, Some(0)) {
        return Ok((Vec::new(), false, 0));
    }
    let n = plan.logical_f32_element_count;
    let cap = max_elements.map(|m| m.min(n));
    let (buf_len, truncated) = match cap {
        Some(c) if c < n => (c, true),
        _ => (n, max_elements.is_some_and(|m| m < n)),
    };
    let mut out = vec![None; buf_len];
    let total_bytes_read_from_disk = scatter_fill(mmap, plan, &mut out)?;
    let vec = finalize_option(out)?;
    Ok((vec, truncated, total_bytes_read_from_disk))
}

pub(crate) fn spill_read_plan_int_le_impl<T, F>(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &Path,
    byte_len: u64,
    scatter: F,
) -> Result<u64, TetError>
where
    T: bytemuck::Pod + Copy + Default,
    F: Fn(&[u8], &ReadPlan, &mut [Option<T>]) -> Result<u64, TetError>,
{
    use std::fs::OpenOptions;
    use std::io::Write;

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|e| TetError::Validation(format!("spill open failed: {e}")))?;
    file.set_len(byte_len)
        .map_err(|e| TetError::Validation(format!("spill set_len failed: {e}")))?;
    file.flush()
        .map_err(|e| TetError::Validation(format!("spill flush failed: {e}")))?;
    let mut out_mmap = unsafe {
        MmapMut::map_mut(&file)
            .map_err(|e| TetError::Validation(format!("spill mmap failed: {e}")))?
    };
    let out: &mut [T] = bytemuck::cast_slice_mut(out_mmap.as_mut());
    validate_full_read_plan_buffer(plan, out.len())?;
    let mut slot = vec![None; out.len()];
    let total = scatter(mmap, plan, &mut slot)?;
    out.copy_from_slice(bytemuck::cast_slice(&finalize_option(slot)?));
    out_mmap
        .flush()
        .map_err(|e| TetError::Validation(format!("spill mmap flush failed: {e}")))?;
    Ok(total)
}

pub(crate) fn preview_from_backing_in_memory<T: Copy>(
    v: &[T],
    logical_len: usize,
    max: usize,
) -> (Vec<T>, bool) {
    let cap = max.min(logical_len);
    if cap == 0 {
        return (Vec::new(), logical_len > 0);
    }
    (v[..cap].to_vec(), logical_len > max)
}

pub(crate) fn preview_from_spill_file_pod<T: bytemuck::Pod + Copy>(
    path: &Path,
    cap: usize,
    logical_len: usize,
) -> Result<(Vec<T>, bool), TetError> {
    let file = std::fs::File::open(path)
        .map_err(|e| TetError::Validation(format!("temp spill read failed: {e}")))?;
    let mmap = unsafe {
        Mmap::map(&file)
            .map_err(|e| TetError::Validation(format!("temp spill mmap failed: {e}")))?
    };
    let slice: &[T] = bytemuck::cast_slice(&mmap);
    if slice.len() < cap {
        return Err(TetError::Validation(
            "temp spill shorter than logical selection".into(),
        ));
    }
    Ok((slice[..cap].to_vec(), logical_len > cap))
}
