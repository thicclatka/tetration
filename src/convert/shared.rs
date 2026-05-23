//! Shared helpers for foreign-format import.

use crate::catalog::{
    ArrayWriteMeta, CHUNK_PAYLOAD_CODEC_V1, CatalogError, StreamTileJob, StreamWriteProgress,
    write_multi_raw_array_streaming,
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
