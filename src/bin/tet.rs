//! `tet` — CLI for JSON queries and (planned) HDF5 / `NetCDF` conversion.

use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use tetration::{
    mmap_file_read, parse_query_json, plan_query_empty, plan_query_with_tet_mmap,
    read_tet_summary_v1, validate_query,
};

#[derive(Parser)]
#[command(
    name = "tet",
    version,
    about = "Tetration CLI: JSON queries and format conversion"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Print superblock metadata for a `.tet` file (layout v1).
    Info {
        /// Path to `.tet` file.
        path: PathBuf,
    },
    /// Validate a JSON query document and print a JSON plan; with `--tet`, attach catalog + read plan; with `--execute`, mmap-read raw `f32` preview (see `--preview-f32`).
    Query {
        /// Path to JSON query; omit or use `-` to read stdin.
        #[arg(short = 'f', long = "file", value_name = "PATH")]
        file: Option<PathBuf>,
        /// Optional `.tet` file: resolve `dataset` against the on-disk catalog (metadata only).
        #[arg(long = "tet", value_name = "PATH")]
        tet: Option<PathBuf>,
        /// After planning, mmap-read planned chunk payloads (raw or zstd `f32`); attach `execution` with capped `f32_preview`. If the query JSON includes **`operation`** (`sum` / `mean` with `axes: []`), the full planned tensor is decoded for stats (see `operation_sum` / `operation_mean`).
        #[arg(long = "execute", default_value_t = false)]
        execute: bool,
        /// Max decoded `f32` values in `execution` when using `--execute` (default 64). Use `0` with a query `operation` to skip preview floats while still aggregating.
        #[arg(long = "preview-f32", value_name = "N")]
        preview_f32: Option<usize>,
    },
    /// Convert foreign formats into Tetration (importers are staged behind the on-disk layout).
    Convert {
        #[command(subcommand)]
        target: ConvertTarget,
    },
}

#[derive(Subcommand)]
enum ConvertTarget {
    /// HDF5 → Tetration (not implemented until `.tet` writer + HDF5 reader are linked).
    H5 { input: PathBuf, output: PathBuf },
    /// `NetCDF` → Tetration (not implemented until `.tet` writer + `NetCDF` reader are linked).
    Netcdf { input: PathBuf, output: PathBuf },
}

fn read_stdin_string() -> io::Result<String> {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

fn read_query_payload(file: Option<&PathBuf>) -> io::Result<String> {
    match file {
        None => read_stdin_string(),
        Some(p) if p.as_os_str() == "-" => read_stdin_string(),
        Some(path) => fs::read_to_string(path),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    const QUERY_PREVIEW_F32_CAP: usize = 64;
    match cli.command {
        Commands::Info { path } => {
            let mmap = mmap_file_read(&path).map_err(|e| e.to_string())?;
            let summary = read_tet_summary_v1(&mmap).map_err(|e| e.to_string())?;
            let out = serde_json::json!({
                "path": path.display().to_string(),
                "summary": summary,
            });
            let pretty = serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?;
            println!("{pretty}");
            Ok(())
        }
        Commands::Query {
            file,
            tet,
            execute,
            preview_f32,
        } => {
            let raw = read_query_payload(file.as_ref()).map_err(|e| e.to_string())?;
            let doc = parse_query_json(raw.trim()).map_err(|e| e.to_string())?;
            validate_query(&doc).map_err(|e| e.to_string())?;
            let preview = if execute {
                Some(preview_f32.unwrap_or(QUERY_PREVIEW_F32_CAP))
            } else if preview_f32.is_some() {
                return Err("`--preview-f32` requires `--execute`".into());
            } else {
                None
            };
            let response = if let Some(path) = tet.as_ref() {
                let path_display = path.display().to_string();
                let mmap = mmap_file_read(path).map_err(|e| e.to_string())?;
                plan_query_with_tet_mmap(&doc, Some(path_display.as_str()), &mmap, preview)
                    .map_err(|e| e.to_string())?
            } else {
                if execute {
                    return Err("`--execute` requires `--tet PATH` (mmap read needs a file)".into());
                }
                plan_query_empty(&doc)
            };
            let out = serde_json::to_string_pretty(&response).map_err(|e| e.to_string())?;
            println!("{out}");
            Ok(())
        }
        Commands::Convert { target } => match target {
            ConvertTarget::H5 { input, output } => Err(format!(
                "HDF5 → Tetration conversion is not implemented yet.\n\
                 Paths were: input={} output={}\n\
                 Next steps: stable `.tet` layout writer, then chunked copy from `hdf5` crate datasets.",
                input.display(),
                output.display()
            )),
            ConvertTarget::Netcdf { input, output } => Err(format!(
                "NetCDF → Tetration conversion is not implemented yet.\n\
                 Paths were: input={} output={}\n\
                 Next steps: stable `.tet` layout writer, then chunked copy from `netcdf` / `netcdf-sys` bindings.",
                input.display(),
                output.display()
            )),
        },
    }
}
