//! Per-dataset chunk grid and logical tensor byte-length cross-checks.

use crate::catalog::tile::{chunk_grid_counts, tile_raw_byte_len, total_chunk_count};
use crate::catalog::{ChunkIndexEntryV1, DatasetRecordV1};
use crate::utils::dtype::ElementDtype;

use super::report::{VerifyFinding, err_finding, ok_finding};

pub(crate) fn check_dataset_tensor_bytes(
    datasets: &[DatasetRecordV1],
    chunks: &[ChunkIndexEntryV1],
) -> Vec<VerifyFinding> {
    let mut findings = Vec::new();
    let mut per_ds: Vec<Vec<usize>> = vec![Vec::new(); datasets.len()];

    for (i, ch) in chunks.iter().enumerate() {
        let id = usize::try_from(ch.dataset_id).unwrap_or(usize::MAX);
        if id >= datasets.len() {
            continue;
        }
        per_ds[id].push(i);
    }

    for (ds_id, ds) in datasets.iter().enumerate() {
        let ndim = ds.shape.len();
        let Some(elem) = ElementDtype::try_from_wire_tag(ds.dtype) else {
            findings.push(err_finding(
                "dataset_tensor_bytes",
                format!(
                    "dataset {ds_id} ({:?}): unsupported dtype tag {}",
                    ds.name, ds.dtype
                ),
            ));
            continue;
        };
        let Some(expected_total) = elem.tensor_bytes_for_shape(&ds.shape) else {
            findings.push(err_finding(
                "dataset_tensor_bytes",
                format!(
                    "dataset {ds_id} ({:?}): logical tensor byte length overflow",
                    ds.name
                ),
            ));
            continue;
        };
        let counts = chunk_grid_counts(&ds.shape, &ds.chunk_shape);
        let expected_chunks = match total_chunk_count(&counts) {
            Ok(n) => n,
            Err(e) => {
                findings.push(err_finding(
                    "dataset_tensor_bytes",
                    format!("dataset {ds_id} ({:?}): {e}", ds.name),
                ));
                continue;
            }
        };
        let indices = &per_ds[ds_id];
        if u64::try_from(indices.len()).ok() != Some(expected_chunks) {
            findings.push(err_finding(
                "dataset_tensor_bytes",
                format!(
                    "dataset {ds_id} ({:?}): chunk count {} != expected grid {}",
                    ds.name,
                    indices.len(),
                    expected_chunks
                ),
            ));
            continue;
        }

        let mut sum_raw: u64 = 0;
        let mut tile_mismatch = false;
        for &ci in indices {
            let ch = &chunks[ci];
            let coord = &ch.chunk_index[..ndim];
            let expected_tile = match tile_raw_byte_len(
                &ds.shape,
                &ds.chunk_shape,
                coord,
                ndim,
                elem.elem_size(),
            ) {
                Ok(n) => n,
                Err(e) => {
                    findings.push(err_finding(
                        "dataset_tensor_bytes",
                        format!("dataset {ds_id} chunk {ci}: {e}"),
                    ));
                    tile_mismatch = true;
                    break;
                }
            };
            if ch.raw_byte_len != expected_tile {
                findings.push(err_finding(
                    "dataset_tensor_bytes",
                    format!(
                        "dataset {ds_id} chunk {ci}: raw_byte_len {} != tile {}",
                        ch.raw_byte_len, expected_tile
                    ),
                ));
                tile_mismatch = true;
            }
            sum_raw = sum_raw.saturating_add(ch.raw_byte_len);
        }
        if tile_mismatch {
            continue;
        }
        if sum_raw != expected_total {
            findings.push(err_finding(
                "dataset_tensor_bytes",
                format!(
                    "dataset {ds_id} ({:?}): sum(raw_byte_len)={sum_raw} != logical tensor {expected_total}",
                    ds.name
                ),
            ));
        } else {
            findings.push(ok_finding(
                "dataset_tensor_bytes",
                Some(format!(
                    "dataset {ds_id} ({:?}): {expected_chunks} chunk(s), {expected_total} B",
                    ds.name
                )),
            ));
        }
    }

    findings
}
