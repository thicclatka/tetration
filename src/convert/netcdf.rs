//! `NetCDF` → `.tet` conversion (streaming, one chunk at a time).

use std::path::Path;
use std::time::Instant;

use netcdf::Variable;
use netcdf::types::{FloatType, IntType, NcVariableType};

use crate::utils::dtype::ElementDtype;

use super::cf::cf_from_netcdf;
use super::parallel::NetcdfParallelSource;
use super::shared::{
    ImportPlan, chunk_shape_for_import, ensure_non_empty, join_catalog_path, write_plans_streaming,
};
use super::{ConvertError, ConvertProgress, ConvertReport, report};
use crate::catalog::append_convert_history;

/// Import all supported numeric variables from a `NetCDF` file into one `.tet`.
///
/// Walks nested groups (`group/var` → catalog name `group/var`). Supported element types:
/// `f32`, `f64`, `i32`, `i64`. CF `scale_factor` / `add_offset` / `_FillValue` are decoded at import.
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
    let mut plans = Vec::new();
    if let Some(root) = file.root() {
        collect_nc_plans(&root, "", &mut plans)?;
    } else {
        let mut vars: Vec<Variable<'_>> = file.variables().collect();
        vars.sort_by_key(Variable::name);
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

fn collect_nc_plans(
    grp: &netcdf::Group<'_>,
    prefix: &str,
    plans: &mut Vec<ImportPlan>,
) -> Result<(), ConvertError> {
    let mut vars: Vec<Variable<'_>> = grp.variables().collect();
    vars.sort_by_key(Variable::name);
    for var in vars {
        if var.dimensions().is_empty() {
            continue;
        }
        let stem = var.name();
        let path = join_catalog_path(prefix, &stem);
        match plan_variable_at(&path, &var) {
            Ok(plan) => plans.push(plan),
            Err(ConvertError::UnsupportedDtype { .. }) => {}
            Err(e) => return Err(e),
        }
    }
    let mut groups: Vec<netcdf::Group<'_>> = grp.groups().collect();
    groups.sort_by_key(netcdf::Group::name);
    for sub in groups {
        let sub_name = sub.name();
        collect_nc_plans(&sub, &join_catalog_path(prefix, &sub_name), plans)?;
    }
    Ok(())
}

fn plan_variable(var: &Variable<'_>) -> Result<ImportPlan, ConvertError> {
    plan_variable_at(&var.name(), var)
}

fn plan_variable_at(name: &str, var: &Variable<'_>) -> Result<ImportPlan, ConvertError> {
    let dtype = map_netcdf_dtype(name, var.vartype())?;
    let shape: Vec<u64> = var
        .dimensions()
        .iter()
        .map(|d| u64::try_from(d.len()).unwrap_or(0))
        .collect();
    if shape.contains(&0) {
        return Err(ConvertError::UnsupportedDtype {
            name: name.to_owned(),
            detail: "zero-length dimension".into(),
        });
    }
    let cf = cf_from_netcdf(var);
    let chunk_shape = if cf.is_some() {
        chunk_shape_for_import(&shape, None)
    } else {
        chunk_shape_for_import(&shape, var.chunking().ok().flatten())
    };
    Ok(ImportPlan {
        name: name.to_owned(),
        dtype,
        shape,
        chunk_shape,
        cf,
        zarr_array_rel: None,
        zarr_zstd: false,
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
