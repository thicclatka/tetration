//! Resolve JSON `selection` to a dense global half-open box and per-axis steps.

use crate::query::document::validate_axis_slice_json;
use crate::query::types::{QueryDocument, TetError};

pub(crate) type ResolvedGlobalBox = (Vec<u64>, Vec<u64>, Vec<u64>);

pub(crate) fn resolved_dense_global_box(
    doc: &QueryDocument,
    shape: &[u64],
) -> Result<ResolvedGlobalBox, TetError> {
    let ndim = shape.len();
    if ndim == 0 {
        return Err(TetError::Validation(
            "dataset rank must be at least 1 for selection planning".into(),
        ));
    }
    match &doc.selection {
        None => Ok((vec![0u64; ndim], shape.to_vec(), vec![1u64; ndim])),
        Some(axes) => {
            if axes.len() != ndim {
                return Err(TetError::Validation(format!(
                    "selection must specify exactly {ndim} axes (one per dataset dimension), got {}",
                    axes.len()
                )));
            }
            let mut g0 = Vec::with_capacity(ndim);
            let mut g1 = Vec::with_capacity(ndim);
            let mut steps = Vec::with_capacity(ndim);
            for (d, sl) in axes.iter().enumerate() {
                validate_axis_slice_json(d, sl)?;
                let sd = shape[d];
                let start = sl.start.unwrap_or(0);
                let stop = sl.stop.unwrap_or(sd);
                if start >= sd {
                    return Err(TetError::Validation(format!(
                        "selection[{d}].start must be < shape[{d}] ({sd}), got {start}"
                    )));
                }
                if stop > sd {
                    return Err(TetError::Validation(format!(
                        "selection[{d}].stop must be <= shape[{d}] ({sd}), got {stop}"
                    )));
                }
                if start >= stop {
                    return Err(TetError::Validation(format!(
                        "selection[{d}]: require start < stop (got {start} >= {stop})"
                    )));
                }
                g0.push(start);
                g1.push(stop);
                steps.push(sl.step.unwrap_or(1));
            }
            Ok((g0, g1, steps))
        }
    }
}
