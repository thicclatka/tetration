//! Individual repair implementations.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::catalog::{CatalogError, payload_file_len};
use crate::layout::{SUPERBLOCK_FLAG_HISTORY_FOOTER, SUPERBLOCK_V1_LEN, SuperblockV1};

use super::RepairActionResult;

/// Apply one repair code to `path`.
///
/// # Errors
///
/// Returns [`CatalogError`] on I/O failure or unsupported code.
pub fn apply_repair_code(
    path: &Path,
    code: &str,
    dry_run: bool,
) -> Result<RepairActionResult, CatalogError> {
    match code {
        "footer_invalid" => strip_invalid_footer(path, dry_run),
        other => Ok(RepairActionResult {
            code: other.to_owned(),
            applied: false,
            dry_run,
            message: format!("unknown repair code: {other}"),
        }),
    }
}

/// Truncate after chunk payloads and clear the history-footer superblock flag.
fn strip_invalid_footer(path: &Path, dry_run: bool) -> Result<RepairActionResult, CatalogError> {
    let data = std::fs::read(path)?;
    let sb = SuperblockV1::from_bytes(
        data.get(..SUPERBLOCK_V1_LEN)
            .ok_or(CatalogError::TooShort {
                need: SUPERBLOCK_V1_LEN,
                got: data.len(),
            })?
            .try_into()
            .map_err(|_| CatalogError::InvalidWriteSpec("superblock length"))?,
    )
    .map_err(CatalogError::Layout)?;

    let payload_end = payload_file_len(&data, sb.flags)?;

    if dry_run {
        return Ok(RepairActionResult {
            code: "footer_invalid".to_owned(),
            applied: false,
            dry_run: true,
            message: format!(
                "would truncate to {payload_end} bytes and clear history footer flag (from {} bytes)",
                data.len()
            ),
        });
    }

    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    f.set_len(payload_end)?;
    f.seek(SeekFrom::Start(0))?;
    let mut sb_bytes = [0u8; SUPERBLOCK_V1_LEN];
    f.read_exact(&mut sb_bytes)?;
    let mut sb = SuperblockV1::from_bytes(&sb_bytes).map_err(CatalogError::Layout)?;
    sb.flags &= !SUPERBLOCK_FLAG_HISTORY_FOOTER;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&sb.to_bytes())?;
    f.sync_all()?;

    Ok(RepairActionResult {
        code: "footer_invalid".to_owned(),
        applied: true,
        dry_run: false,
        message: format!("truncated to {payload_end} bytes and cleared history footer flag"),
    })
}
