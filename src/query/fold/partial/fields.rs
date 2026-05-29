//! Operation preview fields for partial-axis reductions.

use crate::query::fold::reduction;
use crate::query::types::OperationPreviewFields;

pub(crate) fn partial_arg_fields(
    kind: reduction::ReductionKind,
    element_count: usize,
    out_shape: &[u64],
    cells: &[reduction::ArgIndexAccum],
) -> OperationPreviewFields {
    let indices: Vec<u64> = cells.iter().map(reduction::ArgIndexAccum::index).collect();
    let mut fields = OperationPreviewFields {
        element_count: Some(element_count),
        reduced_shape: Some(out_shape.to_vec()),
        ..OperationPreviewFields::default()
    };
    match kind {
        reduction::ReductionKind::ArgMin => fields.reduced_argmin = Some(indices),
        reduction::ReductionKind::ArgMax => fields.reduced_argmax = Some(indices),
        _ => {}
    }
    fields
}

pub(crate) fn partial_fields(
    kind: reduction::ReductionKind,
    element_count: usize,
    out_shape: &[u64],
    reduced: &[f64],
    cells: &[reduction::ValueAccum],
) -> OperationPreviewFields {
    let mut fields = OperationPreviewFields {
        element_count: Some(element_count),
        reduced_shape: Some(out_shape.to_vec()),
        ..OperationPreviewFields::default()
    };
    match kind {
        reduction::ReductionKind::Sum => fields.reduced_sum = Some(reduced.to_vec()),
        reduction::ReductionKind::Mean => fields.reduced_mean = Some(reduced.to_vec()),
        reduction::ReductionKind::NanMean => fields.reduced_nan_mean = Some(reduced.to_vec()),
        reduction::ReductionKind::Min => fields.reduced_min = Some(reduced.to_vec()),
        reduction::ReductionKind::Max => fields.reduced_max = Some(reduced.to_vec()),
        reduction::ReductionKind::Count => fields.reduced_count = Some(reduced.to_vec()),
        reduction::ReductionKind::Var => fields.reduced_var = Some(reduced.to_vec()),
        reduction::ReductionKind::Std => fields.reduced_std = Some(reduced.to_vec()),
        reduction::ReductionKind::NanStd => fields.reduced_nan_std = Some(reduced.to_vec()),
        reduction::ReductionKind::Product => fields.reduced_product = Some(reduced.to_vec()),
        reduction::ReductionKind::NormL1 => fields.reduced_norm_l1 = Some(reduced.to_vec()),
        reduction::ReductionKind::NormL2 => fields.reduced_norm_l2 = Some(reduced.to_vec()),
        reduction::ReductionKind::AllFinite => {
            fields.reduced_all_finite = Some(cells.iter().map(|c| c.finish_bool(kind)).collect());
        }
        reduction::ReductionKind::AnyNan => {
            fields.reduced_any_nan = Some(cells.iter().map(|c| c.finish_bool(kind)).collect());
        }
        reduction::ReductionKind::AnyInf => {
            fields.reduced_any_inf = Some(cells.iter().map(|c| c.finish_bool(kind)).collect());
        }
        reduction::ReductionKind::NanCount => {
            fields.reduced_nan_count = Some(reduced.to_vec());
        }
        reduction::ReductionKind::InfCount => {
            fields.reduced_inf_count = Some(reduced.to_vec());
        }
        reduction::ReductionKind::NullCount { .. } => {
            fields.reduced_null_count = Some(reduced.to_vec());
        }
        reduction::ReductionKind::ArgMin | reduction::ReductionKind::ArgMax => {
            unreachable!("argmin/argmax use partial_arg_fields")
        }
    }
    fields
}
