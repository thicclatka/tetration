//! Orchestrate layout, chunk, and footer verification.

use std::path::Path;

use crate::catalog::{CatalogError, read_tet_summary_v1};

use super::chunks::{
    check_chunk_dataset_ids, check_chunk_decode, check_duplicate_payload_offsets,
    check_payload_order,
};
use super::footer::{check_dataset_count, check_footer, check_footer_metadata_limits};
use super::report::{TetVerifyReport, VerifySummary, ok_finding};

/// Verify a mapped `.tet` byte slice.
#[must_use]
pub fn verify_tet_bytes(data: &[u8], path: Option<&Path>) -> TetVerifyReport {
    let file_len = u64::try_from(data.len()).unwrap_or(u64::MAX);
    let path_s = path.map(|p| p.display().to_string());

    let summary = match read_tet_summary_v1(data) {
        Ok(s) => s,
        Err(e) => {
            return TetVerifyReport::from_fatal(path_s, file_len, "parse", e.to_string());
        }
    };

    let mut findings = vec![
        ok_finding("superblock", None),
        ok_finding("dataset_directory", None),
        ok_finding("chunk_index", None),
    ];

    findings.push(ok_finding(
        "chunk_payloads",
        Some("index spans validated during parse".to_owned()),
    ));

    if let Some(f) = check_chunk_dataset_ids(&summary.datasets, &summary.chunks) {
        findings.push(f);
        return TetVerifyReport {
            ok: false,
            path: path_s,
            file_len,
            findings,
            recommendations: Vec::new(),
            summary: None,
        }
        .finalize();
    }
    findings.push(ok_finding("chunk_dataset_ids", None));

    if let Some(f) = check_duplicate_payload_offsets(&summary.chunks) {
        findings.push(f);
    }

    if let Some(f) = check_payload_order(&summary.chunks) {
        findings.push(f);
    }

    let (decode_findings, deep_decode) = check_chunk_decode(data, &summary.chunks);
    findings.extend(decode_findings);

    if let Some(f) = check_dataset_count(&summary) {
        findings.push(f);
        return TetVerifyReport {
            ok: false,
            path: path_s,
            file_len,
            findings,
            recommendations: Vec::new(),
            summary: None,
        }
        .finalize();
    }
    findings.push(ok_finding("dataset_count", None));

    match check_footer(data, &summary) {
        Ok(f) => findings.push(f),
        Err(f) => {
            findings.push(f);
            return TetVerifyReport {
                ok: false,
                path: path_s,
                file_len,
                findings,
                recommendations: Vec::new(),
                summary: None,
            }
            .finalize();
        }
    }

    let meta_finding = check_footer_metadata_limits(&summary);
    if !meta_finding.ok {
        findings.push(meta_finding);
        return TetVerifyReport {
            ok: false,
            path: path_s,
            file_len,
            findings,
            recommendations: Vec::new(),
            summary: None,
        }
        .finalize();
    }
    findings.push(meta_finding);

    let has_metadata = !summary.metadata.datasets.is_empty() || summary.metadata.file.is_some();
    let history_footer =
        summary.superblock.flags & crate::layout::SUPERBLOCK_FLAG_HISTORY_FOOTER != 0;

    TetVerifyReport {
        ok: true,
        path: path_s,
        file_len,
        findings,
        recommendations: Vec::new(),
        summary: Some(VerifySummary {
            layout_version: summary.superblock.layout_version,
            dataset_count: summary.datasets.len(),
            chunk_count: summary.chunks.len(),
            history_events: summary.history.len(),
            has_metadata,
            history_footer,
            deep_chunk_decode: deep_decode,
        }),
    }
    .finalize()
}

/// Verify a `.tet` file on disk (reads entire file into memory).
///
/// # Errors
///
/// Returns [`CatalogError::Io`] when the file cannot be read.
pub fn verify_tet_file(path: &Path) -> Result<TetVerifyReport, CatalogError> {
    let data = std::fs::read(path)?;
    Ok(verify_tet_bytes(&data, Some(path)))
}
