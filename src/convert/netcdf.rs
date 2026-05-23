//! `NetCDF` → `.tet` conversion (streaming, one chunk at a time).

use std::path::Path;
use std::time::Instant;

use netcdf::Variable;
use netcdf::types::{FloatType, IntType, NcVariableType};

use crate::utils::dtype::ElementDtype;

use super::parallel::NetcdfParallelSource;
use super::shared::{ImportPlan, chunk_shape_for_import, ensure_non_empty, write_plans_streaming};
use super::{ConvertError, ConvertProgress, ConvertReport, report};
use crate::catalog::append_convert_history;

/// Import all supported root-level numeric variables from a `NetCDF` file into one `.tet`.
///
/// Supported element types: `f32`, `f64`, `i32`, `i64`. Peak host RAM is roughly
/// `parallel_jobs` chunk tiles (plus index metadata), not the full logical tensor size.
///
/// # Errors
///
/// Returns [`ConvertError`] when the file cannot be read or no supported variables are present.
pub fn convert_netcdf_to_tet(input: &Path, output: &Path) -> Result<ConvertReport, ConvertError> {
    convert_netcdf_to_tet_with_progress(input, output, 0, None::<fn(ConvertProgress)>)
}

/// Like [`convert_netcdf_to_tet`], invoking `progress` after each chunk payload is written.
///
/// `parallel_jobs`: chunk read workers (`0` = [`super::parallel::default_parallel_jobs`]).
///
/// # Errors
///
/// Returns [`ConvertError`] when the file cannot be read or no supported variables are present.
pub fn convert_netcdf_to_tet_with_progress(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    mut progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    let started = Instant::now();
    let file = netcdf::open(input).map_err(|e| ConvertError::Netcdf(e.to_string()))?;
    let mut vars: Vec<Variable<'_>> = file.variables().collect();
    vars.sort_by_key(Variable::name);

    let mut plans = Vec::new();
    for var in &vars {
        if var.dimensions().is_empty() {
            continue;
        }
        match plan_variable(var) {
            Ok(plan) => plans.push(plan),
            Err(ConvertError::UnsupportedDtype { .. }) => {}
            Err(e) => return Err(e),
        }
    }
    ensure_non_empty(
        input,
        &plans.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
    )?;

    let mut progress_bridge = |done: u64, total: u64, dataset: &str| {
        if let Some(ref mut cb) = progress {
            cb(ConvertProgress {
                chunks_done: done,
                chunks_total: total,
                dataset: dataset.to_owned(),
            });
        }
    };

    let parallel_jobs = super::parallel::resolve_parallel_jobs(parallel_jobs);
    let source = NetcdfParallelSource::new(input.to_path_buf(), plans.clone());
    write_plans_streaming(
        output,
        &plans,
        parallel_jobs,
        |job, buf| source.fill_tile(job, buf),
        Some(&mut progress_bridge as &mut dyn FnMut(u64, u64, &str)),
    )?;
    let history = append_convert_history(output, "nc")?;

    Ok(report(
        input,
        output,
        &plans,
        history,
        started.elapsed().as_secs_f64(),
    ))
}

fn plan_variable(var: &Variable<'_>) -> Result<ImportPlan, ConvertError> {
    let name = var.name();
    let dtype = map_netcdf_dtype(&name, var.vartype())?;
    let shape: Vec<u64> = var
        .dimensions()
        .iter()
        .map(|d| u64::try_from(d.len()).unwrap_or(0))
        .collect();
    if shape.contains(&0) {
        return Err(ConvertError::UnsupportedDtype {
            name,
            detail: "zero-length dimension".into(),
        });
    }
    let chunk_shape = chunk_shape_for_import(&shape, var.chunking().ok().flatten());
    Ok(ImportPlan {
        name,
        dtype,
        shape,
        chunk_shape,
    })
}

fn map_netcdf_dtype(name: &str, ty: NcVariableType) -> Result<ElementDtype, ConvertError> {
    match ty {
        NcVariableType::Float(FloatType::F32) => Ok(ElementDtype::F32),
        NcVariableType::Float(FloatType::F64) => Ok(ElementDtype::F64),
        NcVariableType::Int(IntType::I32) => Ok(ElementDtype::I32),
        NcVariableType::Int(IntType::I64) => Ok(ElementDtype::I64),
        other => Err(ConvertError::UnsupportedDtype {
            name: name.to_owned(),
            detail: format!("{other:?}"),
        }),
    }
}
