//! Map verification findings to actionable recommendations.

use super::report::{VerifyFinding, VerifyFixHint, VerifyRecommendation, VerifySeverity};

/// Build deduplicated recommendations from the current finding set.
#[must_use]
pub fn recommendations_for_findings(findings: &[VerifyFinding]) -> Vec<VerifyRecommendation> {
    let mut out = Vec::new();
    for f in findings {
        if f.ok {
            continue;
        }
        let Some(rec) = recommendation_for_finding(f) else {
            continue;
        };
        if out
            .iter()
            .any(|r: &VerifyRecommendation| r.code == rec.code)
        {
            continue;
        }
        out.push(rec);
    }
    out
}

fn recommendation_for_finding(f: &VerifyFinding) -> Option<VerifyRecommendation> {
    let detail = f.detail.as_deref().unwrap_or("");
    match f.check.as_str() {
        "parse" if detail.contains("footer") => Some(VerifyRecommendation {
            code: "footer_invalid".to_owned(),
            message: "History/metadata footer is present but malformed.".to_owned(),
            fix: Some(VerifyFixHint {
                summary: "Strip the invalid footer (truncate after chunk payloads).".to_owned(),
                command: None,
            }),
        }),
        "parse" | "superblock" | "dataset_directory" | "chunk_index" => Some(VerifyRecommendation {
            code: "invalid_layout".to_owned(),
            message: "File layout is invalid or truncated; it cannot be read reliably.".to_owned(),
            fix: Some(VerifyFixHint {
                summary: "Re-create the `.tet` from the source array (convert or writer session)."
                    .to_owned(),
                command: Some("tet convert <source> <output.tet>".to_owned()),
            }),
        }),
        "chunk_payloads" | "payload_bounds" => Some(VerifyRecommendation {
            code: "payload_out_of_bounds".to_owned(),
            message: "A chunk index entry points outside the file payload region.".to_owned(),
            fix: Some(VerifyFixHint {
                summary: "Restore from backup or re-export; do not append to a truncated file."
                    .to_owned(),
                command: None,
            }),
        }),
        "chunk_dataset_ids" => Some(VerifyRecommendation {
            code: "bad_dataset_id".to_owned(),
            message: "Chunk index references a non-existent dataset id.".to_owned(),
            fix: Some(VerifyFixHint {
                summary: "Re-write the file; the catalog index is internally inconsistent.".to_owned(),
                command: None,
            }),
        }),
        "chunk_decode" => Some(VerifyRecommendation {
            code: "chunk_decode_failed".to_owned(),
            message: format!("Chunk payload could not be decoded: {detail}"),
            fix: Some(VerifyFixHint {
                summary: "Re-convert from the original HDF5/NetCDF/Zarr source.".to_owned(),
                command: Some("tet convert <source> <output.tet>".to_owned()),
            }),
        }),
        "footer" => Some(VerifyRecommendation {
            code: "footer_invalid".to_owned(),
            message: "History/metadata footer is present but malformed.".to_owned(),
            fix: Some(VerifyFixHint {
                summary: "Strip the invalid footer (truncate after chunk payloads).".to_owned(),
                command: None,
            }),
        }),
        "dataset_count" => Some(VerifyRecommendation {
            code: "dataset_count_mismatch".to_owned(),
            message: "Superblock dataset count does not match the catalog blob.".to_owned(),
            fix: Some(VerifyFixHint {
                summary: "Re-create the file; superblock and catalog disagree.".to_owned(),
                command: None,
            }),
        }),
        "payload_order" if f.severity == VerifySeverity::Warning => Some(VerifyRecommendation {
            code: "non_contiguous_payloads".to_owned(),
            message: "Chunk payloads are not stored in contiguous read-plan order.".to_owned(),
            fix: Some(VerifyFixHint {
                summary: "Optional: re-convert for sequential layout; full-file linear scan may be slower."
                    .to_owned(),
                command: None,
            }),
        }),
        "chunk_decode_skipped" if f.severity == VerifySeverity::Warning => Some(VerifyRecommendation {
            code: "run_deep_verify".to_owned(),
            message: "Not all chunks were decode-checked (large file).".to_owned(),
            fix: Some(VerifyFixHint {
                summary: "Re-run with deep decode when a library API adds that mode, or spot-check with query execute."
                    .to_owned(),
                command: None,
            }),
        }),
        _ => None,
    }
}
