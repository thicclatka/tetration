//! Map foreign dataset attributes into footer `metadata` on convert.

use std::collections::BTreeMap;
use std::path::Path;

use crate::catalog::{
    CatalogError, FileMetadataV1, FooterBlobV1, HistoryEventV1, TetMetadataV1, unix_timestamp_now,
    write_footer_blob,
};

use super::shared::ImportPlan;

/// CF / discovery attribute names imported first when present.
const PREFERRED_ATTRS: &[&str] = &[
    "units",
    "long_name",
    "standard_name",
    "description",
    "title",
    "scale_factor",
    "add_offset",
    "_FillValue",
];

/// Write convert history plus optional dataset metadata into the `THST` footer.
///
/// # Errors
///
/// Returns [`CatalogError`] when metadata validation or footer I/O fails.
pub fn finish_convert_footer(
    output: &Path,
    source: &str,
    plans: &[ImportPlan],
) -> Result<Vec<HistoryEventV1>, CatalogError> {
    let metadata = build_convert_metadata(plans)?;
    let event = (
        "convert".to_owned(),
        source.to_owned(),
        unix_timestamp_now(),
    );
    write_footer_blob(
        output,
        &FooterBlobV1 {
            history: vec![event.clone()],
            metadata,
        },
    )?;
    Ok(vec![event])
}

fn build_convert_metadata(plans: &[ImportPlan]) -> Result<Option<TetMetadataV1>, CatalogError> {
    let mut meta = TetMetadataV1 {
        file: Some(FileMetadataV1 {
            tool: Some("tet convert".to_owned()),
            library_version: Some(env!("CARGO_PKG_VERSION").to_owned()),
            created_at: Some(unix_timestamp_now()),
        }),
        ..TetMetadataV1::default()
    };
    for plan in plans {
        if plan.import_attrs.is_empty() && plan.import_dim_names.is_none() {
            continue;
        }
        let entry = meta.dataset_mut(&plan.name);
        entry.attrs.clone_from(&plan.import_attrs);
        entry.dim_names.clone_from(&plan.import_dim_names);
    }
    meta.validate()?;
    Ok(Some(meta))
}

#[cfg(feature = "tetration-hdf5")]
pub(crate) fn hdf5_dataset_attrs(ds: &hdf5_metno::Dataset) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let Ok(names) = ds.attr_names() else {
        return out;
    };
    import_named_attrs(&mut out, &names, |name| hdf5_attr_string(ds, name));
    out
}

#[cfg(feature = "tetration-hdf5")]
fn hdf5_attr_string(ds: &hdf5_metno::Dataset, name: &str) -> Option<String> {
    use hdf5_metno::types::{VarLenAscii, VarLenUnicode};

    let attr = ds.attr(name).ok()?;
    let reader = attr.as_reader();
    if let Ok(s) = reader.read_scalar::<VarLenUnicode>() {
        return non_empty_string(s.as_str().to_owned());
    }
    if let Ok(s) = reader.read_scalar::<VarLenAscii>() {
        return non_empty_string(s.as_str().to_owned());
    }
    if let Ok(f) = reader.read_scalar::<f32>() {
        return Some(f.to_string());
    }
    if let Ok(f) = reader.read_scalar::<f64>() {
        return Some(f.to_string());
    }
    if let Ok(n) = reader.read_scalar::<i32>() {
        return Some(n.to_string());
    }
    if let Ok(n) = reader.read_scalar::<i64>() {
        return Some(n.to_string());
    }
    None
}

#[cfg(feature = "tetration-netcdf")]
pub(crate) fn netcdf_variable_attrs(var: &netcdf::Variable<'_>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let names: Vec<String> = var.attributes().map(|a| a.name().to_owned()).collect();
    import_named_attrs(&mut out, &names, |name| {
        var.attribute_value(name)
            .and_then(|r| r.ok())
            .and_then(nc_attr_value_to_string)
    });
    out
}

#[cfg(feature = "tetration-netcdf")]
pub(crate) fn netcdf_dim_names(var: &netcdf::Variable<'_>) -> Option<Vec<String>> {
    let names: Vec<String> = var
        .dimensions()
        .iter()
        .map(|d| d.name())
        .filter(|n| !n.is_empty())
        .collect();
    if names.is_empty() {
        return None;
    }
    let limits = crate::catalog::MetadataLimitsV1::DEFAULT;
    if names.len() > limits.dim_names {
        return None;
    }
    for n in &names {
        if n.len() > limits.attr_string_bytes {
            return None;
        }
    }
    Some(names)
}

#[cfg(feature = "tetration-netcdf")]
fn nc_attr_value_to_string(value: netcdf::AttributeValue) -> Option<String> {
    use netcdf::AttributeValue;
    let s = match value {
        AttributeValue::Str(s) => s,
        AttributeValue::Strs(v) => v.into_iter().next()?,
        AttributeValue::Uchar(x) => return Some(x.to_string()),
        AttributeValue::Schar(x) => return Some(x.to_string()),
        AttributeValue::Ushort(x) => return Some(x.to_string()),
        AttributeValue::Short(x) => return Some(x.to_string()),
        AttributeValue::Uint(x) => return Some(x.to_string()),
        AttributeValue::Int(x) => return Some(x.to_string()),
        AttributeValue::Ulonglong(x) => return Some(x.to_string()),
        AttributeValue::Longlong(x) => return Some(x.to_string()),
        AttributeValue::Float(x) => return Some(x.to_string()),
        AttributeValue::Double(x) => return Some(x.to_string()),
        _ => return None,
    };
    non_empty_string(s)
}

fn import_named_attrs(
    out: &mut BTreeMap<String, String>,
    names: &[String],
    mut read: impl FnMut(&str) -> Option<String>,
) {
    for key in PREFERRED_ATTRS {
        if names.iter().any(|n| n == *key)
            && let Some(v) = read(key)
        {
            out.insert((*key).to_owned(), v);
        }
    }
    let limits = crate::catalog::MetadataLimitsV1::DEFAULT;
    for name in names {
        if out.len() >= limits.dataset_attrs {
            break;
        }
        if PREFERRED_ATTRS.contains(&name.as_str()) {
            continue;
        }
        if name.len() > limits.attr_string_bytes {
            continue;
        }
        if let Some(v) = read(name)
            && v.len() <= limits.attr_string_bytes
        {
            out.insert(name.clone(), v);
        }
    }
}

fn non_empty_string(s: String) -> Option<String> {
    if s.is_empty() { None } else { Some(s) }
}
