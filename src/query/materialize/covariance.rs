//! Population covariance / correlation along one observation axis (tier C).

use crate::query::types::{OperationPreviewFields, TetError};

/// Max variables (matrix side length) for covariance / correlation.
pub(crate) const MAX_COVARIANCE_VARS: u32 = 1024;

#[derive(Debug, Clone, Copy)]
struct Matrix2dLayout {
    n_obs: usize,
    n_var: usize,
    obs_axis: usize,
    stride: usize,
}

fn matrix_layout(shape: &[u64], obs_axis: usize) -> Result<Matrix2dLayout, TetError> {
    if shape.len() != 2 {
        return Err(TetError::Validation(format!(
            "covariance/correlation require a rank-2 logical selection (got rank {})",
            shape.len()
        )));
    }
    let s0 = usize::try_from(shape[0])
        .map_err(|_| TetError::Validation("shape axis 0 overflow".into()))?;
    let s1 = usize::try_from(shape[1])
        .map_err(|_| TetError::Validation("shape axis 1 overflow".into()))?;
    if obs_axis > 1 {
        return Err(TetError::Validation(format!(
            "observation axis index {obs_axis} out of range for rank 2"
        )));
    }
    let (n_obs, n_var) = if obs_axis == 0 { (s0, s1) } else { (s1, s0) };
    if n_obs == 0 || n_var == 0 {
        return Err(TetError::Validation(
            "covariance/correlation require non-empty observation and variable axes".into(),
        ));
    }
    let n_var_u32 = u32::try_from(n_var)
        .map_err(|_| TetError::Validation("variable axis length overflow".into()))?;
    if n_var_u32 > MAX_COVARIANCE_VARS {
        return Err(TetError::Validation(format!(
            "variable axis length {n_var} exceeds maximum {MAX_COVARIANCE_VARS}"
        )));
    }
    Ok(Matrix2dLayout {
        n_obs,
        n_var,
        obs_axis,
        stride: s1,
    })
}

#[inline]
fn value_at(values: &[f64], layout: Matrix2dLayout, sample: usize, var: usize) -> f64 {
    let li = if layout.obs_axis == 0 {
        sample * layout.stride + var
    } else {
        var * layout.stride + sample
    };
    values[li]
}

fn population_covariance_matrix(
    values: &[f64],
    layout: Matrix2dLayout,
) -> Result<Vec<f64>, TetError> {
    let Matrix2dLayout { n_obs, n_var, .. } = layout;
    let mut means = vec![0.0; n_var];
    for (j, mean) in means.iter_mut().enumerate() {
        let mut sum = 0.0;
        for i in 0..n_obs {
            let v = value_at(values, layout, i, j);
            if !v.is_finite() {
                return Err(TetError::Validation(
                    "covariance/correlation require finite values in the logical selection".into(),
                ));
            }
            sum += v;
        }
        *mean = sum / n_obs as f64;
    }
    let mut cov = vec![0.0; n_var * n_var];
    for j in 0..n_var {
        for k in 0..n_var {
            let mut acc = 0.0;
            for i in 0..n_obs {
                let dj = value_at(values, layout, i, j) - means[j];
                let dk = value_at(values, layout, i, k) - means[k];
                acc += dj * dk;
            }
            cov[j * n_var + k] = acc / n_obs as f64;
        }
    }
    Ok(cov)
}

fn correlation_from_covariance(cov: &[f64], n_var: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_var * n_var];
    let mut std = vec![0.0; n_var];
    for j in 0..n_var {
        let v = cov[j * n_var + j];
        std[j] = if v > 0.0 { v.sqrt() } else { 0.0 };
    }
    for j in 0..n_var {
        for k in 0..n_var {
            let idx = j * n_var + k;
            if j == k {
                out[idx] = 1.0;
            } else if std[j] == 0.0 || std[k] == 0.0 {
                out[idx] = 0.0;
            } else {
                out[idx] = cov[idx] / (std[j] * std[k]);
            }
        }
    }
    out
}

pub(crate) fn run_covariance_correlation(
    values: &[f64],
    shape: &[u64],
    obs_axis: usize,
    correlation: bool,
) -> Result<OperationPreviewFields, TetError> {
    let layout = matrix_layout(shape, obs_axis)?;
    let cov = population_covariance_matrix(values, layout)?;
    let order = u64::try_from(layout.n_var)
        .map_err(|_| TetError::Validation("covariance matrix order overflow".into()))?;
    if correlation {
        let matrix = correlation_from_covariance(&cov, layout.n_var);
        Ok(OperationPreviewFields {
            element_count: Some(values.len()),
            correlation_order: Some(order),
            correlation: Some(matrix),
            ..OperationPreviewFields::default()
        })
    } else {
        Ok(OperationPreviewFields {
            element_count: Some(values.len()),
            covariance_order: Some(order),
            covariance: Some(cov),
            ..OperationPreviewFields::default()
        })
    }
}
