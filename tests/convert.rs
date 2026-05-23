//! Convert integration tests against tracked `fixtures/small/` tensors.

use std::path::{Path, PathBuf};

#[cfg(feature = "tetration-hdf5")]
use tetration::convert_h5_to_tet_with_progress;
use tetration::{
    DATASET_DTYPE_TAG_V1, convert_netcdf_to_tet_with_progress, materialize_read_plan_f32_le,
    materialize_read_plan_f64_le, materialize_read_plan_i32_le, materialize_read_plan_i64_le,
    mmap_file_read, parse_query_json, plan_query_with_tet_mmap, read_tet_summary_v1,
    validate_query,
};

const FIXTURE_DTYPES: [&str; 4] = ["f32", "f64", "i32", "i64"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

#[cfg(feature = "tetration-hdf5")]
fn small_h5(name: &str) -> PathBuf {
    repo_root().join("fixtures/small/h5").join(name)
}

fn small_nc(name: &str) -> PathBuf {
    repo_root().join("fixtures/small/netcdf").join(name)
}

fn materialize_dataset_le_bytes(mmap: &[u8], dataset: &str) -> Vec<u8> {
    let doc = parse_query_json(&format!(r#"{{"dataset":"{dataset}"}}"#)).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, mmap, None).unwrap();
    let cat = plan.catalog.as_ref().unwrap();
    let dtype = cat.dtype.expect("matched dataset dtype");
    let rp = plan.read_plan.as_ref().unwrap();
    let tags = DATASET_DTYPE_TAG_V1;
    if tags.is_f32(dtype) {
        let (vals, _, _) = materialize_read_plan_f32_le(mmap, rp, None).unwrap();
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if tags.is_f64(dtype) {
        let (vals, _, _) = materialize_read_plan_f64_le(mmap, rp, None).unwrap();
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if tags.is_i32(dtype) {
        let (vals, _, _) = materialize_read_plan_i32_le(mmap, rp, None).unwrap();
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if tags.is_i64(dtype) {
        let (vals, _, _) = materialize_read_plan_i64_le(mmap, rp, None).unwrap();
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else {
        panic!("unexpected dtype {dtype}");
    }
}

#[cfg(feature = "tetration-hdf5")]
fn hdf5_dataset_le_bytes(ds: &hdf5_metno::Dataset, name: &str) -> Vec<u8> {
    match name {
        "f32" => ds
            .read_raw::<f32>()
            .unwrap()
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
        "f64" => ds
            .read_raw::<f64>()
            .unwrap()
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect(),
        "i32" => ds
            .read_raw::<i32>()
            .unwrap()
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect(),
        "i64" => ds
            .read_raw::<i64>()
            .unwrap()
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect(),
        other => panic!("unexpected dataset {other}"),
    }
}

#[cfg(feature = "tetration-hdf5")]
fn assert_small_fixture_h5(stem: &str) {
    assert_small_fixture_h5_with_jobs(stem, 1);
}

#[cfg(feature = "tetration-hdf5")]
fn assert_small_fixture_h5_with_jobs(stem: &str, parallel_jobs: usize) {
    let input = small_h5(&format!("{stem}.h5"));
    assert!(input.is_file(), "missing fixture {}", input.display());
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join(format!("{stem}.tet"));
    convert_h5_to_tet_with_progress(&input, &output, parallel_jobs, None::<fn(_)>).unwrap();
    let mmap = mmap_file_read(&output).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets.len(), 4);
    assert_eq!(summary.history.len(), 1);
    assert_eq!(summary.history[0].0, "convert");
    assert_eq!(summary.history[0].1, "h5");

    let src = hdf5_metno::File::open(&input).unwrap();
    for name in FIXTURE_DTYPES {
        let ds = src.dataset(name).unwrap();
        let want = hdf5_dataset_le_bytes(&ds, name);
        let got = materialize_dataset_le_bytes(&mmap, name);
        assert_eq!(got, want, "dataset {name} bytes differ for {stem}");
    }
}

fn assert_small_fixture_netcdf(stem: &str) {
    assert_small_fixture_netcdf_with_jobs(stem, 1);
}

fn assert_small_fixture_netcdf_with_jobs(stem: &str, parallel_jobs: usize) {
    let input = small_nc(&format!("{stem}.nc"));
    assert!(input.is_file(), "missing fixture {}", input.display());
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join(format!("{stem}.tet"));
    convert_netcdf_to_tet_with_progress(&input, &output, parallel_jobs, None::<fn(_)>).unwrap();
    let mmap = mmap_file_read(&output).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets.len(), 4);
    assert_eq!(summary.history.len(), 1);
    assert_eq!(summary.history[0].0, "convert");
    assert_eq!(summary.history[0].1, "nc");

    let src = netcdf::open(&input).unwrap();
    for name in FIXTURE_DTYPES {
        let var = src.variable(name).expect("source variable");
        let want = var.get_raw_values(..).unwrap();
        let got = materialize_dataset_le_bytes(&mmap, name);
        assert_eq!(got, want, "dataset {name} bytes differ for {stem}");
    }
}

#[cfg(feature = "tetration-hdf5")]
#[test]
fn convert_small_h5_tensor_3d_matches_source_bytes() {
    assert_small_fixture_h5("tensor_3d");
}

#[cfg(feature = "tetration-hdf5")]
#[test]
fn convert_small_h5_tensor_4d_matches_source_bytes() {
    assert_small_fixture_h5("tensor_4d");
}

#[cfg(feature = "tetration-hdf5")]
#[test]
fn convert_small_h5_tensor_5d_matches_source_bytes() {
    assert_small_fixture_h5("tensor_5d");
}

#[test]
fn detect_convert_format_from_extension() {
    use tetration::{
        ConvertInputFormat, Hdf5ConvertInput, NetcdfConvertInput, detect_convert_format,
    };

    for ext in Hdf5ConvertInput::EXTENSIONS {
        let path = format!("tensor.{ext}");
        assert_eq!(
            detect_convert_format(Path::new(&path)).unwrap(),
            ConvertInputFormat::H5,
            "expected HDF5 for .{ext}"
        );
    }
    assert_eq!(
        detect_convert_format(Path::new("a.HDF5")).unwrap(),
        ConvertInputFormat::H5
    );

    for ext in NetcdfConvertInput::EXTENSIONS {
        let path = format!("tensor.{ext}");
        assert_eq!(
            detect_convert_format(Path::new(&path)).unwrap(),
            ConvertInputFormat::Netcdf,
            "expected NetCDF for .{ext}"
        );
    }
    assert_eq!(
        detect_convert_format(Path::new("a.NETCDF")).unwrap(),
        ConvertInputFormat::Netcdf
    );
    assert_eq!(
        detect_convert_format(Path::new("model.nc.gz")).unwrap(),
        ConvertInputFormat::Netcdf
    );
    assert!(detect_convert_format(Path::new("data.csv")).is_err());
}

#[test]
fn detect_convert_format_sniffs_file_signature() {
    use tetration::{
        ConvertInputFormat, Hdf5ConvertInput, NetcdfConvertInput, detect_convert_format,
    };

    let dir = tempfile::tempdir().unwrap();
    let h5_path = dir.path().join("payload.bin");
    std::fs::write(&h5_path, Hdf5ConvertInput::MAGIC).unwrap();
    assert_eq!(
        detect_convert_format(&h5_path).unwrap(),
        ConvertInputFormat::H5
    );

    let nc_path = dir.path().join("payload.dat");
    std::fs::write(&nc_path, NetcdfConvertInput::NETCDF3_V1).unwrap();
    assert_eq!(
        detect_convert_format(&nc_path).unwrap(),
        ConvertInputFormat::Netcdf
    );
}

#[cfg(feature = "tetration-hdf5")]
#[test]
fn convert_small_h5_parallel_jobs_matches_source_bytes() {
    assert_small_fixture_h5_with_jobs("tensor_3d", 4);
}

#[test]
fn convert_small_netcdf_parallel_jobs_matches_source_bytes() {
    assert_small_fixture_netcdf_with_jobs("tensor_3d", 4);
}

#[test]
fn convert_small_netcdf_tensor_3d_matches_source_bytes() {
    assert_small_fixture_netcdf("tensor_3d");
}

#[test]
fn convert_small_netcdf_tensor_4d_matches_source_bytes() {
    assert_small_fixture_netcdf("tensor_4d");
}

#[test]
fn convert_small_netcdf_tensor_5d_matches_source_bytes() {
    assert_small_fixture_netcdf("tensor_5d");
}
