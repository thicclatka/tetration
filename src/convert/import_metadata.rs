//! Map foreign dataset attributes into footer `metadata` on convert.

use std::collections::BTreeMap;
use std::path::Path;

use crate::catalog::{
    CatalogError, CoordAxisV1, FileMetadataV1, FooterBlobV1, HistoryEventV1, MetadataLimitsV1,
    TetMetadataV1, unix_timestamp_now, write_footer_blob,
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
        if plan.import_attrs.is_empty()
            && plan.import_dim_names.is_none()
            && plan.import_coords.is_none()
        {
            continue;
        }
        let entry = meta.dataset_mut(&plan.name);
        entry.attrs.clone_from(&plan.import_attrs);
        entry.dim_names.clone_from(&plan.import_dim_names);
        entry.coords.clone_from(&plan.import_coords);
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
    if let Ok(dt) = attr.dtype()
        && let Ok(td) = dt.to_descriptor()
        && let Some(s) = hdf5_attr_string_fixed(&reader, &td)
    {
        return non_empty_string(s);
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

/// Zarr v3 array `attributes` object from `zarr.json` → footer metadata attrs.
pub(crate) fn zarr_array_attrs(
    attributes: &serde_json::Map<String, serde_json::Value>,
) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let names: Vec<String> = attributes.keys().cloned().collect();
    import_named_attrs(&mut out, &names, |name| {
        attributes.get(name).and_then(json_attr_value_to_string)
    });
    out
}

#[cfg(feature = "tetration-hdf5")]
fn hdf5_attr_string_fixed(
    reader: &hdf5_metno::Reader<'_>,
    td: &hdf5_metno::types::TypeDescriptor,
) -> Option<String> {
    use hdf5_metno::types::TypeDescriptor;
    match td {
        TypeDescriptor::FixedAscii(n) => try_hdf5_fixed_ascii(reader, *n),
        TypeDescriptor::FixedUnicode(n) => try_hdf5_fixed_unicode(reader, *n),
        _ => None,
    }
}

#[cfg(feature = "tetration-hdf5")]
fn try_hdf5_fixed_ascii(reader: &hdf5_metno::Reader<'_>, cap: usize) -> Option<String> {
    use hdf5_metno::types::FixedAscii;
    macro_rules! try_cap {
        ($($n:literal),+) => {
            $(
                if cap == $n {
                    if let Ok(v) = reader.read_scalar::<FixedAscii<$n>>() {
                        return Some(v.as_str().to_owned());
                    }
                }
            )+
        };
    }
    try_cap!(
        8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 512, 1024, 2048, 4096
    );
    None
}

#[cfg(feature = "tetration-hdf5")]
fn try_hdf5_fixed_unicode(reader: &hdf5_metno::Reader<'_>, cap: usize) -> Option<String> {
    use hdf5_metno::types::FixedUnicode;
    macro_rules! try_cap {
        ($($n:literal),+) => {
            $(
                if cap == $n {
                    if let Ok(v) = reader.read_scalar::<FixedUnicode<$n>>() {
                        return Some(v.as_str().to_owned());
                    }
                }
            )+
        };
    }
    try_cap!(
        8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 512, 1024, 2048, 4096
    );
    None
}

#[cfg(feature = "tetration-hdf5")]
pub(crate) fn enrich_hdf5_cf_coordinates(file: &hdf5_metno::File, plans: &mut [ImportPlan]) {
    for plan in plans {
        let Some(coord_list) = plan.import_attrs.get("coordinates") else {
            continue;
        };
        let mut coords = BTreeMap::new();
        for axis in coord_list.split_whitespace() {
            let Some(ds) = resolve_hdf5_coord_dataset(file, axis) else {
                continue;
            };
            if let Some(axis_coords) = hdf5_1d_coord_labels(&ds) {
                coords.insert(axis.to_owned(), axis_coords);
            }
        }
        if !coords.is_empty() {
            plan.import_coords = Some(coords);
        }
    }
}

#[cfg(feature = "tetration-hdf5")]
fn resolve_hdf5_coord_dataset(
    file: &hdf5_metno::File,
    axis: &str,
) -> Option<hdf5_metno::Dataset> {
    for path in [format!("coordinates/{axis}"), axis.to_owned()] {
        if let Ok(ds) = file.dataset(&path) {
            return Some(ds);
        }
    }
    None
}

#[cfg(feature = "tetration-hdf5")]
pub(crate) fn hdf5_1d_coord_labels(ds: &hdf5_metno::Dataset) -> Option<CoordAxisV1> {
    use crate::utils::dtype::ElementDtype;

    let dtype = map_hdf5_element_dtype(ds).ok()?;
    let shape = ds.shape();
    if shape.len() != 1 {
        return None;
    }
    let n = shape[0];
    let limits = MetadataLimitsV1::DEFAULT;
    if n == 0 || n > limits.coord_labels_per_axis {
        return None;
    }
    let labels = match dtype {
        ElementDtype::F32 => ds
            .read_raw::<f32>()
            .ok()?
            .into_iter()
            .map(|v| trim_coord_label(&v.to_string()))
            .collect(),
        ElementDtype::F64 => ds
            .read_raw::<f64>()
            .ok()?
            .into_iter()
            .map(|v| trim_coord_label(&v.to_string()))
            .collect(),
        ElementDtype::I32 => ds
            .read_raw::<i32>()
            .ok()?
            .into_iter()
            .map(|v| v.to_string())
            .collect(),
        ElementDtype::I64 => ds
            .read_raw::<i64>()
            .ok()?
            .into_iter()
            .map(|v| v.to_string())
            .collect(),
    };
    Some(CoordAxisV1 { labels })
}

#[cfg(feature = "tetration-hdf5")]
fn map_hdf5_element_dtype(
    ds: &hdf5_metno::Dataset,
) -> Result<crate::utils::dtype::ElementDtype, ()> {
    use crate::utils::dtype::ElementDtype;
    use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor};
    let desc = ds.dtype().map_err(|_| ())?.to_descriptor().map_err(|_| ())?;
    match desc {
        TypeDescriptor::Float(FloatSize::U4) => Ok(ElementDtype::F32),
        TypeDescriptor::Float(FloatSize::U8) => Ok(ElementDtype::F64),
        TypeDescriptor::Integer(IntSize::U4) => Ok(ElementDtype::I32),
        TypeDescriptor::Integer(IntSize::U8) => Ok(ElementDtype::I64),
        _ => Err(()),
    }
}

#[cfg(feature = "tetration-netcdf")]
pub(crate) fn netcdf_inline_coord_labels(var: &netcdf::Variable<'_>) -> Option<CoordAxisV1> {
    use netcdf::types::{FloatType, NcVariableType};
    if var.dimensions().len() != 1 {
        return None;
    }
    let n = var.dimensions()[0].len();
    let limits = MetadataLimitsV1::DEFAULT;
    if n == 0 || n > limits.coord_labels_per_axis {
        return None;
    }
    let labels: Vec<String> = match var.vartype() {
        NcVariableType::Float(FloatType::F32) => (0..n)
            .filter_map(|i| {
                var.get_value::<f32, _>(i)
                    .ok()
                    .map(|v| trim_coord_label(&v.to_string()))
            })
            .collect(),
        NcVariableType::Float(FloatType::F64) => (0..n)
            .filter_map(|i| {
                var.get_value::<f64, _>(i)
                    .ok()
                    .map(|v| trim_coord_label(&v.to_string()))
            })
            .collect(),
        NcVariableType::Int(netcdf::types::IntType::I32) => (0..n)
            .filter_map(|i| var.get_value::<i32, _>(i).ok().map(|v| v.to_string()))
            .collect(),
        NcVariableType::Int(netcdf::types::IntType::I64) => (0..n)
            .filter_map(|i| var.get_value::<i64, _>(i).ok().map(|v| v.to_string()))
            .collect(),
        _ => return None,
    };
    (labels.len() == n).then_some(CoordAxisV1 { labels })
}

fn trim_coord_label(s: &str) -> String {
    if s.len() > MetadataLimitsV1::DEFAULT.attr_string_bytes {
        s[..MetadataLimitsV1::DEFAULT.attr_string_bytes].to_owned()
    } else {
        s.to_owned()
    }
}

fn json_attr_value_to_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => non_empty_string(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Null => None,
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => None,
    }
}
