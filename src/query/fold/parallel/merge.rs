//! Merge per-chunk parallel fold accumulators.

use crate::query::fold::{partial_geometry, reduction};
use crate::query::types::{OperationPreviewFields, TetError};
use crate::query::{decode, dispatch};

pub(crate) struct ScalarChunkWork {
    pub bytes: u64,
    pub value: reduction::ValueAccum,
    pub arg: reduction::ArgIndexAccum,
}

pub(crate) struct PartialChunkValue {
    pub bytes: u64,
    pub cells: Vec<reduction::ValueAccum>,
    pub saw_any: bool,
}

pub(crate) struct PartialChunkArg {
    pub bytes: u64,
    pub cells: Vec<reduction::ArgIndexAccum>,
    pub saw_any: bool,
}

pub(crate) fn sum_chunk_bytes(bytes: impl IntoIterator<Item = u64>) -> Result<u64, TetError> {
    dispatch::sum_chunk_read_bytes(bytes)
}

pub(crate) fn merge_scalar_chunks(
    parts: &[ScalarChunkWork],
    kind: reduction::ReductionKind,
    n: usize,
) -> Result<OperationPreviewFields, TetError> {
    match kind {
        reduction::ReductionKind::ArgMin | reduction::ReductionKind::ArgMax => {
            let mut acc = reduction::ArgIndexAccum::default();
            for p in parts {
                acc.merge_from(&p.arg, kind);
            }
            if acc.is_empty() {
                return Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ));
            }
            Ok(acc.finish_scalar(kind, n).into())
        }
        _ => {
            let mut acc = reduction::ValueAccum::default();
            for p in parts {
                acc.merge_from(&p.value);
            }
            if acc.is_empty() {
                return Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ));
            }
            Ok(acc.finish_scalar(kind).into())
        }
    }
}

pub(crate) fn merge_partial_value_cells(
    dst: &mut [reduction::ValueAccum],
    src: &[reduction::ValueAccum],
) {
    for (d, s) in dst.iter_mut().zip(src) {
        d.merge_from(s);
    }
}

pub(crate) fn merge_partial_arg_cells(
    dst: &mut [reduction::ArgIndexAccum],
    src: &[reduction::ArgIndexAccum],
    kind: reduction::ReductionKind,
) {
    for (d, s) in dst.iter_mut().zip(src) {
        d.merge_from(s, kind);
    }
}

pub(crate) fn reduced_cell_index(
    li: usize,
    shape: &[u64],
    layout: &partial_geometry::PartialAxisLayout,
) -> Result<(usize, u64), TetError> {
    let coords = decode::indexing::coords_from_linear_row_major(li, shape)?;
    let oi = partial_geometry::reduced_index(&coords, &layout.axis_set, &layout.out_shape)?;
    let fi = partial_geometry::fiber_linear_index(&coords, &layout.axis_indices, shape)? as u64;
    Ok((oi, fi))
}
