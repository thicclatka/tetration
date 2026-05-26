//! Footer (`THST`) verification.

use crate::catalog::{MetadataLimitsV1, TetFileSummaryV1, read_footer_blob};
use crate::layout::SUPERBLOCK_FLAG_HISTORY_FOOTER;

use super::report::{VerifyFinding, err_finding, ok_finding};

pub(crate) fn check_footer(
    data: &[u8],
    summary: &TetFileSummaryV1,
) -> Result<VerifyFinding, VerifyFinding> {
    let history_footer = summary.superblock.flags & SUPERBLOCK_FLAG_HISTORY_FOOTER != 0;
    if !history_footer {
        return Ok(ok_finding("footer", Some("absent".to_owned())));
    }
    read_footer_blob(data, summary.superblock.flags)
        .map(|_| ok_finding("footer", Some("history footer valid".to_owned())))
        .map_err(|e| super::report::err_finding("footer", e.to_string()))
}

/// Resolved footer `metadata` / `history` within [`MetadataLimitsV1`] (spill read via [`read_footer_blob`]).
pub(crate) fn check_footer_metadata_limits(summary: &TetFileSummaryV1) -> VerifyFinding {
    let limits = MetadataLimitsV1::DEFAULT;
    if summary.history.len() > limits.history_events {
        return err_finding(
            "footer_metadata",
            format!(
                "history has {} events (limit {})",
                summary.history.len(),
                limits.history_events
            ),
        );
    }
    match summary.metadata.validate() {
        Ok(()) => ok_finding(
            "footer_metadata",
            Some("metadata within MetadataLimitsV1".to_owned()),
        ),
        Err(e) => err_finding("footer_metadata", e.to_string()),
    }
}

pub(crate) fn check_dataset_count(summary: &TetFileSummaryV1) -> Option<VerifyFinding> {
    if summary.superblock.dataset_count as usize != summary.datasets.len() {
        return Some(super::report::err_finding(
            "dataset_count",
            format!(
                "superblock dataset_count={} but parsed {} records",
                summary.superblock.dataset_count,
                summary.datasets.len()
            ),
        ));
    }
    None
}
