//! Shared helpers for foreign-format import.

use std::collections::BTreeMap;

use super::cf::CfTransform;

use crate::catalog::{
    ArrayWriteMeta, CHUNK_PAYLOAD_CODEC_V1, CatalogError, CoordAxisV1, StreamTileJob,
    StreamWriteProgress, write_multi_raw_array_streaming,
};
use crate::utils::dtype::ElementDtype;

use super::ConvertError;

/// Dataset geometry collected before a streaming write.
#[derive(Clone)]
pub(crate) struct ImportPlan {
    pub name: String,
    pub dtype: ElementDtype,
    pub shape: Vec<u64>,
    pub chunk_shape: Vec<u64>,
    pub cf: Option<CfTransform>,
    /// Zarr array path relative to store root (`primary/f32`); `None` for HDF5/NetCDF.
    pub zarr_array_rel: Option<String>,
    /// When importing Zarr: chunk files on disk are zstd-compressed (`bytes` + `zstd` codecs).
    pub zarr_zstd: bool,
    /// Dataset attributes copied into footer `metadata.datasets[name].attrs`.
    pub import_attrs: BTreeMap<String, String>,
    /// NetCDF dimension names → `metadata.datasets[name].dim_names`.
    pub import_dim_names: Option<Vec<String>>,
    /// Inline coordinate labels → `metadata.datasets[name].coords`.
    pub import_coords: Option<BTreeMap<String, CoordAxisV1>>,
}

pub(crate) fn join_catalog_path(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}/{name}")
    }
}

/// One tile read during import: geometry from the plan plus the chunk coordinate from the stream job.
#[derive(Clone, Copy)]
pub(crate) struct ImportTileRead<'a> {
    pub dtype: ElementDtype,
    pub shape: &'a [u64],
    pub chunk_shape: &'a [u64],
    pub chunk_coord: &'a [u64],
    pub ndim: usize,
    pub cf: Option<CfTransform>,
}

impl ImportPlan {
    pub(crate) fn tile_read<'a>(&'a self, job: &'a StreamTileJob<'_>) -> ImportTileRead<'a> {
        ImportTileRead {
            dtype: self.dtype,
            shape: &self.shape,
            chunk_shape: &self.chunk_shape,
            chunk_coord: &job.chunk_coord[..job.ndim],
            ndim: job.ndim,
            cf: self.cf,
        }
    }
}

/// Pick a v1 chunk grid: reuse source chunking when it matches rank, else one tile per array.
pub(crate) fn chunk_shape_for_import(shape: &[u64], source_chunks: Option<Vec<usize>>) -> Vec<u64> {
    if let Some(chunks) = source_chunks
        && chunks.len() == shape.len()
        && chunks.iter().all(|&c| c > 0)
    {
        let mut out = Vec::with_capacity(chunks.len());
        for (&dim, &chunk) in shape.iter().zip(chunks.iter()) {
            let c = u64::try_from(chunk).unwrap_or(dim);
            out.push(c.min(dim).max(1));
        }
        return out;
    }
    shape.to_vec()
}

pub(crate) fn write_plans_streaming(
    output: &std::path::Path,
    plans: &[ImportPlan],
    parallel_jobs: usize,
    fill_tile: impl Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), ConvertError> + Sync + Send,
    progress_hook: Option<&mut StreamWriteProgress<'_>>,
) -> Result<(), ConvertError> {
    let mut metas: Vec<ArrayWriteMeta<'_>> = Vec::with_capacity(plans.len());
    for plan in plans {
        metas.push(ArrayWriteMeta {
            name: &plan.name,
            dtype: plan.dtype.wire_tag(),
            shape: &plan.shape,
            chunk_shape: &plan.chunk_shape,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            file_execution: None,
        });
    }
    write_multi_raw_array_streaming(
        output,
        &metas,
        parallel_jobs,
        |job, buf| match fill_tile(job, buf) {
            Ok(()) => Ok(()),
            Err(ConvertError::Catalog(c)) => Err(c),
            Err(e) => Err(CatalogError::Io(std::io::Error::other(e.to_string()))),
        },
        progress_hook,
    )
    .map_err(ConvertError::from)
}

pub(crate) fn ensure_non_empty(
    path: &std::path::Path,
    names: &[String],
) -> Result<(), ConvertError> {
    if names.is_empty() {
        return Err(ConvertError::NoDatasets {
            path: path.display().to_string(),
        });
    }
    Ok(())
}
