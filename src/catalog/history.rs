//! Optional file footer at EOF (`THST`): history rows and optional `metadata` JSON.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::layout::{SUPERBLOCK_FLAG_HISTORY_FOOTER, SUPERBLOCK_V1_LEN, SuperblockV1};

use super::CatalogError;
use super::metadata::{self, TetMetadataV1};

/// On-disk `THST` history footer tail (layout v1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryFooterWireV1 {
    /// EOF magic (`THST`).
    pub magic: [u8; 4],
    /// Footer format version (must match on read).
    pub version: u32,
    /// Byte length of the fixed tail (`history_json_len` + `history_version` + magic).
    pub tail_len: usize,
}

impl HistoryFooterWireV1 {
    /// Layout v1 wire constants for the history footer.
    pub const DEFAULT: Self = Self {
        magic: *b"THST",
        version: 1,
        tail_len: 16,
    };
}

/// One history row: `(operation, source, timestamp_utc)`.
///
/// Convert events use `op = "convert"`, `source = "h5"` or `"nc"`, and `at` as Unix seconds (string).
pub type HistoryEventV1 = (String, String, String);

/// Parsed `THST` JSON payload (history + optional metadata).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FooterBlobV1 {
    #[serde(default)]
    pub history: Vec<HistoryEventV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<TetMetadataV1>,
}

/// Append a convert event footer and set [`SUPERBLOCK_FLAG_HISTORY_FOOTER`] on the superblock.
///
/// Merges with an existing footer when present (preserves `metadata`).
///
/// # Errors
///
/// Returns [`CatalogError`] when JSON encoding or I/O fails.
pub fn append_convert_history(
    path: &Path,
    source: &str,
) -> Result<Vec<HistoryEventV1>, CatalogError> {
    let mut blob = read_footer_blob_from_path(path).unwrap_or_default();
    let event = (
        "convert".to_owned(),
        source.to_owned(),
        unix_timestamp_now(),
    );
    blob.history.push(event.clone());
    write_footer_blob(path, &blob)?;
    Ok(vec![event])
}

/// Unix seconds since epoch as a decimal string (history `at` field).
#[must_use]
pub fn unix_timestamp_now() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or_else(|_| "0".to_owned(), |d| d.as_secs().to_string())
}

/// Append history rows (merges with existing footer `metadata` when present).
///
/// # Errors
///
/// Returns [`CatalogError`] when JSON encoding or I/O fails.
pub fn append_history_events(path: &Path, events: &[HistoryEventV1]) -> Result<(), CatalogError> {
    if events.is_empty() {
        return Ok(());
    }
    let mut blob = read_footer_blob_from_path(path).unwrap_or_default();
    blob.history.extend_from_slice(events);
    write_footer_blob(path, &blob)
}

/// Write footer JSON (`history` + optional `metadata`), replacing any prior footer.
///
/// # Errors
///
/// Returns [`CatalogError`] when validation, JSON encoding, or I/O fails.
pub fn write_footer_blob(path: &Path, blob: &FooterBlobV1) -> Result<(), CatalogError> {
    if blob.history.is_empty() && blob.metadata.is_none() {
        return Ok(());
    }
    if let Some(meta) = &blob.metadata {
        meta.validate()?;
    }
    let json =
        serde_json::to_vec(blob).map_err(|_| CatalogError::InvalidWriteSpec("footer JSON"))?;
    metadata::validate_footer_json_len(&json)?;
    rewrite_footer_bytes(path, &json)
}

fn read_footer_blob_from_path(path: &Path) -> Result<FooterBlobV1, CatalogError> {
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
    read_footer_blob(&data, sb.flags)
}

/// Parse the `THST` footer when present.
///
/// # Errors
///
/// Returns [`CatalogError`] when the footer is present but malformed.
pub fn read_footer_blob(data: &[u8], flags: u32) -> Result<FooterBlobV1, CatalogError> {
    let Some(footer_len) = history_footer_len(data)? else {
        return Ok(FooterBlobV1::default());
    };
    if flags & SUPERBLOCK_FLAG_HISTORY_FOOTER == 0 {
        return Ok(FooterBlobV1::default());
    }
    let json_start = data
        .len()
        .checked_sub(footer_len)
        .ok_or(CatalogError::TooShort {
            need: footer_len,
            got: data.len(),
        })?;
    let wire = HistoryFooterWireV1::DEFAULT;
    serde_json::from_slice::<FooterBlobV1>(&data[json_start..data.len() - wire.tail_len])
        .map_err(|_| CatalogError::InvalidWriteSpec("footer JSON parse"))
}

/// Parse history rows from the footer (empty when absent).
///
/// # Errors
///
/// Returns [`CatalogError`] when the footer is present but malformed.
pub fn read_history(data: &[u8], flags: u32) -> Result<Vec<HistoryEventV1>, CatalogError> {
    Ok(read_footer_blob(data, flags)?.history)
}

/// Parse optional metadata from the footer.
///
/// # Errors
///
/// Returns [`CatalogError`] when the footer is present but malformed.
pub fn read_metadata(data: &[u8], flags: u32) -> Result<TetMetadataV1, CatalogError> {
    Ok(read_footer_blob(data, flags)?.metadata.unwrap_or_default())
}

fn rewrite_footer_bytes(path: &Path, json: &[u8]) -> Result<(), CatalogError> {
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

    let payload_end = usize::try_from(payload_file_len(&data, sb.flags)?).map_err(|_| {
        CatalogError::TooLargeForPlatform {
            field: "payload_end",
            value: u64::MAX,
        }
    })?;

    let json_len = u64::try_from(json.len()).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "footer_json_len",
        value: u64::MAX,
    })?;

    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    f.set_len(payload_end as u64)?;
    f.seek(SeekFrom::End(0))?;
    let wire = HistoryFooterWireV1::DEFAULT;
    f.write_all(json)?;
    f.write_all(&json_len.to_le_bytes())?;
    f.write_all(&wire.version.to_le_bytes())?;
    f.write_all(&wire.magic)?;

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
    let wire = HistoryFooterWireV1::DEFAULT;
    if data.len() < wire.tail_len {
        return Ok(None);
    }
    let tail = data.len() - wire.tail_len;
    if data[tail + 12..tail + 16] != wire.magic {
        return Ok(None);
    }
    let version = u32::from_le_bytes(data[tail + 8..tail + 12].try_into().unwrap());
    if version != wire.version {
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
    let total = wire
        .tail_len
        .checked_add(json_len_usize)
        .ok_or(CatalogError::InvalidWriteSpec("history footer overflow"))?;
    if total > data.len() {
        return Err(CatalogError::InvalidWriteSpec(
            "history footer out of bounds",
        ));
    }
    Ok(Some(total))
}
