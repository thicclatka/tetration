//! Shared partial-axis reduction geometry (output shape, index mapping).

use std::collections::BTreeSet;

use crate::query::decode::indexing::{coords_from_linear_row_major, linear_rm_index};
use crate::query::plan::read_plan::shape_product_usize;
use crate::query::types::{ReadPlan, TetError};

#[derive(Debug, Clone)]
pub(crate) struct PartialAxisLayout {
    pub axis_indices: Vec<usize>,
    pub axis_set: BTreeSet<usize>,
    pub out_shape: Vec<u64>,
    pub out_len: usize,
}

pub(crate) fn parse_axis_indices(labels: &[String], ndim: usize) -> Result<Vec<usize>, TetError> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for label in labels {
        let v: usize = label
            .parse()
            .map_err(|_| TetError::Validation(format!("invalid operation axis index {label:?}")))?;
        if v >= ndim {
            return Err(TetError::Validation(format!(
                "operation axis index {v} out of range for rank {ndim}"
            )));
        }
        if !seen.insert(v) {
            return Err(TetError::Validation(format!(
                "duplicate operation axis index {v}"
            )));
        }
        out.push(v);
    }
    out.sort_unstable();
    Ok(out)
}

pub(crate) fn out_shape(shape: &[u64], axis_indices: &[usize]) -> Vec<u64> {
    let axis_set: BTreeSet<usize> = axis_indices.iter().copied().collect();
    shape
        .iter()
        .enumerate()
        .filter(|(d, _)| !axis_set.contains(d))
        .map(|(_, &e)| e)
        .collect()
}

pub(crate) fn partial_axis_layout(
    plan: &ReadPlan,
    axis_labels: &[String],
) -> Result<PartialAxisLayout, TetError> {
    let shape = &plan.logical_selection_shape;
    let ndim = shape.len();
    let axis_indices = parse_axis_indices(axis_labels, ndim)?;
    if axis_indices.is_empty() {
        return Err(TetError::Validation(
            "internal: partial layout requires non-empty axes".into(),
        ));
    }
    if axis_indices.len() == ndim {
        return Err(TetError::Validation(
            "operation axes list reduces every dimension; use \"axes\": [] for a scalar reduction"
                .into(),
        ));
    }
    let out_shape = out_shape(shape, &axis_indices);
    let out_len = shape_product_usize(&out_shape)?;
    Ok(PartialAxisLayout {
        axis_set: axis_indices.iter().copied().collect(),
        axis_indices,
        out_shape,
        out_len,
    })
}

pub(crate) fn reduced_index(
    coords: &[usize],
    axis_set: &BTreeSet<usize>,
    out_shape: &[u64],
) -> Result<usize, TetError> {
    let mut out_c = Vec::new();
    for (d, &c) in coords.iter().enumerate() {
        if !axis_set.contains(&d) {
            out_c.push(c);
        }
    }
    linear_rm_index(&out_c, out_shape)
}

pub(crate) fn fiber_linear_index(
    coords: &[usize],
    axis_indices: &[usize],
    shape: &[u64],
) -> Result<usize, TetError> {
    let rshape: Vec<u64> = axis_indices.iter().map(|&d| shape[d]).collect();
    let rc: Vec<usize> = axis_indices.iter().map(|&d| coords[d]).collect();
    linear_rm_index(&rc, &rshape)
}

/// Map a logical row-major index to reduced output cell and fiber index (partial-axis fold).
pub(crate) fn reduced_cell_index(
    li: usize,
    shape: &[u64],
    layout: &PartialAxisLayout,
) -> Result<(usize, u64), TetError> {
    let coords = coords_from_linear_row_major(li, shape)?;
    let oi = reduced_index(&coords, &layout.axis_set, &layout.out_shape)?;
    let fi = fiber_linear_index(&coords, &layout.axis_indices, shape)? as u64;
    Ok((oi, fi))
}
