//! Optional file history footer (layout v1 extension).

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::layout::{SUPERBLOCK_FLAG_HISTORY_FOOTER, SUPERBLOCK_V1_LEN, SuperblockV1};

use super::CatalogError;

const HISTORY_FOOTER_MAGIC: &[u8; 4] = b"THST";
const HISTORY_FOOTER_VERSION: u32 = 1;
const HISTORY_FOOTER_TAIL_LEN: usize = 16;

/// One history row: `(operation, source, timestamp_utc)`.
///
/// Convert events use `op = "convert"`, `source = "h5"` or `"nc"`, and `at` as Unix seconds (string).
pub type HistoryEventV1 = (String, String, String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct HistoryBlobV1 {
    history: Vec<HistoryEventV1>,
}

/// Append a convert event footer and set [`SUPERBLOCK_FLAG_HISTORY_FOOTER`] on the superblock.
///
/// Returns the events written (today: one row).
///
/// # Errors
///
/// Returns [`CatalogError`] when JSON encoding or I/O fails.
pub fn append_convert_history(
    path: &Path,
    source: &str,
) -> Result<Vec<HistoryEventV1>, CatalogError> {
    let event = (
        "convert".to_owned(),
        source.to_owned(),
        unix_timestamp_string(),
    );
    let blob = HistoryBlobV1 {
        history: vec![event.clone()],
    };
    append_history_blob(path, &blob)?;
    Ok(vec![event])
}

fn unix_timestamp_string() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or_else(|_| "0".to_owned(), |d| d.as_secs().to_string())
}

fn append_history_blob(path: &Path, blob: &HistoryBlobV1) -> Result<(), CatalogError> {
    let json =
        serde_json::to_vec(blob).map_err(|_| CatalogError::InvalidWriteSpec("history JSON"))?;
    let json_len = u64::try_from(json.len()).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "history_json_len",
        value: u64::MAX,
    })?;

    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    f.seek(SeekFrom::End(0))?;
    f.write_all(&json)?;
    f.write_all(&json_len.to_le_bytes())?;
    f.write_all(&HISTORY_FOOTER_VERSION.to_le_bytes())?;
    f.write_all(HISTORY_FOOTER_MAGIC)?;

    let mut sb_bytes = [0u8; SUPERBLOCK_V1_LEN];
    f.seek(SeekFrom::Start(0))?;
    f.read_exact(&mut sb_bytes)?;
    let mut sb = SuperblockV1::from_bytes(&sb_bytes).map_err(CatalogError::Layout)?;
    sb.flags |= SUPERBLOCK_FLAG_HISTORY_FOOTER;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&sb.to_bytes())?;
    f.sync_all()?;
    Ok(())
}

/// Parse history from a mapped file when the superblock flag or trailing magic is present.
///
/// # Errors
///
/// Returns [`CatalogError`] when the footer is present but malformed.
pub fn read_history(data: &[u8], flags: u32) -> Result<Vec<HistoryEventV1>, CatalogError> {
    let Some(footer_len) = history_footer_len(data)? else {
        return Ok(Vec::new());
    };
    if flags & SUPERBLOCK_FLAG_HISTORY_FOOTER == 0 {
        return Ok(Vec::new());
    }
    let json_start = data
        .len()
        .checked_sub(footer_len)
        .ok_or(CatalogError::TooShort {
            need: footer_len,
            got: data.len(),
        })?;
    let json: HistoryBlobV1 =
        serde_json::from_slice(&data[json_start..data.len() - HISTORY_FOOTER_TAIL_LEN])
            .map_err(|_| CatalogError::InvalidWriteSpec("history JSON parse"))?;
    Ok(json.history)
}

/// File length used for chunk payload bounds (excludes optional history footer).
///
/// # Errors
///
/// Returns [`CatalogError`] when a history footer is declared but malformed.
pub fn payload_file_len(data: &[u8], flags: u32) -> Result<u64, CatalogError> {
    let len = u64::try_from(data.len()).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "file_len",
        value: u64::MAX,
    })?;
    if flags & SUPERBLOCK_FLAG_HISTORY_FOOTER == 0 {
        return Ok(len);
    }
    let footer = history_footer_len(data)?.ok_or(CatalogError::InvalidWriteSpec(
        "superblock history flag set but footer missing",
    ))?;
    len.checked_sub(
        u64::try_from(footer).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "history_footer_len",
            value: u64::MAX,
        })?,
    )
    .ok_or(CatalogError::InvalidWriteSpec(
        "history footer longer than file",
    ))
}

fn history_footer_len(data: &[u8]) -> Result<Option<usize>, CatalogError> {
    if data.len() < HISTORY_FOOTER_TAIL_LEN {
        return Ok(None);
    }
    let tail = data.len() - HISTORY_FOOTER_TAIL_LEN;
    if &data[tail + 12..tail + 16] != HISTORY_FOOTER_MAGIC {
        return Ok(None);
    }
    let version = u32::from_le_bytes(data[tail + 8..tail + 12].try_into().unwrap());
    if version != HISTORY_FOOTER_VERSION {
        return Err(CatalogError::InvalidWriteSpec(
            "unsupported history footer version",
        ));
    }
    let json_len = u64::from_le_bytes(data[tail..tail + 8].try_into().unwrap());
    let json_len_usize =
        usize::try_from(json_len).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "history_json_len",
            value: json_len,
        })?;
    let total = HISTORY_FOOTER_TAIL_LEN
        .checked_add(json_len_usize)
        .ok_or(CatalogError::InvalidWriteSpec("history footer overflow"))?;
    if total > data.len() {
        return Err(CatalogError::InvalidWriteSpec(
            "history footer out of bounds",
        ));
    }
    Ok(Some(total))
}
