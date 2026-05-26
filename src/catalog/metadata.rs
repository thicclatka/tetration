//! Optional file/dataset metadata in the v1 `THST` footer JSON (`metadata` key).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::CatalogError;

/// Validation limits for footer `metadata` and the combined footer JSON blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataLimitsV1 {
    /// Max UTF-8 bytes for the entire footer JSON blob (history + metadata).
    pub footer_json_bytes: usize,
    /// Max dataset entries in `metadata.datasets`.
    pub metadata_datasets: usize,
    /// Max attribute keys per dataset.
    pub dataset_attrs: usize,
    /// Max bytes per attribute key or value string.
    pub attr_string_bytes: usize,
    /// Max axis dimension names (`ndim` strings).
    pub dim_names: usize,
    /// Max coordinate axes per dataset in `coords`.
    pub coord_axes: usize,
    /// Max index labels per coordinate axis (inline storage).
    pub coord_labels_per_axis: usize,
    /// Max rows in footer `history`.
    pub history_events: usize,
    /// Max `parents` entries per history row.
    pub history_parents: usize,
    /// Max `params` keys per history row.
    pub history_params: usize,
    /// Max UTF-8 bytes for spilled `metadata` blob (before `THST` JSON).
    pub metadata_blob_bytes: usize,
}

impl MetadataLimitsV1 {
    /// Layout v1 wire limits for footer metadata.
    pub const DEFAULT: Self = Self {
        footer_json_bytes: 65_536,
        metadata_datasets: 256,
        dataset_attrs: 64,
        attr_string_bytes: 1024,
        dim_names: 8,
        coord_axes: 8,
        coord_labels_per_axis: 64,
        history_events: 4096,
        history_parents: 16,
        history_params: 32,
        metadata_blob_bytes: 16 * 1024 * 1024,
    };
}

/// Inline coordinate index labels for one axis (Phase 7 baseline).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CoordAxisV1 {
    pub labels: Vec<String>,
}

/// File-level metadata (tool, library version, creation time).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FileMetadataV1 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

/// Per-dataset metadata keyed by catalog name in [`TetMetadataV1::datasets`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DatasetMetadataV1 {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attrs: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dim_names: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coords: Option<BTreeMap<String, CoordAxisV1>>,
}

/// Footer `metadata` object: file header fields + per-dataset map.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TetMetadataV1 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<FileMetadataV1>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub datasets: BTreeMap<String, DatasetMetadataV1>,
}

impl DatasetMetadataV1 {
    /// True when there is nothing to persist for this dataset (convert or session import).
    ///
    /// `dim_names` / `coords` use `Option<&T>` so callers can pass `.as_ref()` on owned fields.
    #[must_use]
    pub fn import_is_empty(
        attrs: &BTreeMap<String, String>,
        dim_names: Option<&Vec<String>>,
        coords: Option<&BTreeMap<String, CoordAxisV1>>,
    ) -> bool {
        attrs.is_empty() && dim_names.is_none() && coords.is_none()
    }

    /// Merge import-time fields from convert or the embedder session writer (present fields only).
    pub fn apply_import(
        &mut self,
        attrs: &BTreeMap<String, String>,
        dim_names: Option<&Vec<String>>,
        coords: Option<&BTreeMap<String, CoordAxisV1>>,
    ) {
        if !attrs.is_empty() {
            self.attrs.clone_from(attrs);
        }
        if let Some(names) = dim_names {
            self.dim_names = Some(names.clone());
        }
        if let Some(c) = coords {
            self.coords = Some(c.clone());
        }
    }
}

impl TetMetadataV1 {
    /// Metadata for one dataset, creating the map entry when missing.
    pub fn dataset_mut(&mut self, name: &str) -> &mut DatasetMetadataV1 {
        self.datasets.entry(name.to_owned()).or_default()
    }

    /// Validate bounds before writing to disk.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::InvalidWriteSpec`] when limits are exceeded.
    pub fn validate(&self) -> Result<(), CatalogError> {
        let limits = MetadataLimitsV1::DEFAULT;
        if self.datasets.len() > limits.metadata_datasets {
            return Err(CatalogError::InvalidWriteSpec(
                "metadata.datasets exceeds metadata_datasets limit",
            ));
        }
        if let Some(file) = &self.file {
            validate_opt_string(
                file.tool.as_deref(),
                "metadata.file.tool",
                limits.attr_string_bytes,
            )?;
            validate_opt_string(
                file.library_version.as_deref(),
                "metadata.file.library_version",
                limits.attr_string_bytes,
            )?;
            validate_opt_string(
                file.created_at.as_deref(),
                "metadata.file.created_at",
                limits.attr_string_bytes,
            )?;
        }
        for (name, ds) in &self.datasets {
            validate_opt_string(
                Some(name.as_str()),
                "metadata.datasets key",
                limits.attr_string_bytes,
            )?;
            if ds.attrs.len() > limits.dataset_attrs {
                return Err(CatalogError::InvalidWriteSpec(
                    "metadata dataset attrs exceeds dataset_attrs limit",
                ));
            }
            for (k, v) in &ds.attrs {
                validate_attr_string(k, "metadata attr key", limits.attr_string_bytes)?;
                validate_attr_string(v, "metadata attr value", limits.attr_string_bytes)?;
            }
            if let Some(dim_names) = &ds.dim_names {
                if dim_names.len() > limits.dim_names {
                    return Err(CatalogError::InvalidWriteSpec(
                        "metadata dim_names exceeds dim_names limit",
                    ));
                }
                for d in dim_names {
                    validate_attr_string(d, "metadata dim_name", limits.attr_string_bytes)?;
                }
            }
            if let Some(coords) = &ds.coords {
                if coords.len() > limits.coord_axes {
                    return Err(CatalogError::InvalidWriteSpec(
                        "metadata coords exceeds coord_axes limit",
                    ));
                }
                for (axis, c) in coords {
                    validate_attr_string(
                        axis,
                        "metadata coord axis name",
                        limits.attr_string_bytes,
                    )?;
                    if c.labels.len() > limits.coord_labels_per_axis {
                        return Err(CatalogError::InvalidWriteSpec(
                            "metadata coord labels exceeds coord_labels_per_axis limit",
                        ));
                    }
                    for label in &c.labels {
                        validate_attr_string(
                            label,
                            "metadata coord label",
                            limits.attr_string_bytes,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn validate_opt_string(
    s: Option<&str>,
    field: &'static str,
    max_bytes: usize,
) -> Result<(), CatalogError> {
    if let Some(s) = s {
        validate_attr_string(s, field, max_bytes)?;
    }
    Ok(())
}

pub(crate) fn validate_attr_string(
    s: &str,
    field: &'static str,
    max_bytes: usize,
) -> Result<(), CatalogError> {
    if s.len() > max_bytes {
        return Err(CatalogError::InvalidWriteSpec(
            "metadata string exceeds attr_string_bytes limit",
        ));
    }
    if s.is_empty() {
        return Err(CatalogError::InvalidWriteSpec(field));
    }
    Ok(())
}

/// Reject footer JSON payloads that exceed the wire size cap.
///
/// # Errors
///
/// Returns [`CatalogError::InvalidWriteSpec`] when `json.len()` exceeds
/// [`MetadataLimitsV1::DEFAULT`] [`MetadataLimitsV1::footer_json_bytes`].
pub fn validate_footer_json_len(json: &[u8]) -> Result<(), CatalogError> {
    let limits = MetadataLimitsV1::DEFAULT;
    if json.len() > limits.footer_json_bytes {
        return Err(CatalogError::InvalidWriteSpec(
            "footer JSON exceeds footer_json_bytes limit",
        ));
    }
    Ok(())
}

/// Reject spilled metadata blobs that exceed the wire size cap.
///
/// # Errors
///
/// Returns [`CatalogError::InvalidWriteSpec`] when `json.len()` exceeds
/// [`MetadataLimitsV1::metadata_blob_bytes`].
pub fn validate_metadata_blob_len(json: &[u8]) -> Result<(), CatalogError> {
    let limits = MetadataLimitsV1::DEFAULT;
    if json.len() > limits.metadata_blob_bytes {
        return Err(CatalogError::InvalidWriteSpec(
            "metadata spill exceeds metadata_blob_bytes limit",
        ));
    }
    Ok(())
}
