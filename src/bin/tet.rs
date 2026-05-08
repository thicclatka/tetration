//! `tet` — CLI for JSON queries and (planned) HDF5 / NetCDF conversion.

use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use tetration::{parse_query_json, plan_query, validate_query};

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
    /// Validate a JSON query document and print a JSON plan (execution hooks land later).
    Query {
        /// Path to JSON query; omit or use `-` to read stdin.
        #[arg(short = 'f', long = "file", value_name = "PATH")]
        file: Option<PathBuf>,
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
    /// NetCDF → Tetration (not implemented until `.tet` writer + NetCDF reader are linked).
    Netcdf { input: PathBuf, output: PathBuf },
}

fn read_stdin_string() -> io::Result<String> {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

fn read_query_payload(file: Option<PathBuf>) -> io::Result<String> {
    match file.as_ref() {
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
    match cli.command {
        Commands::Query { file } => {
            let raw = read_query_payload(file).map_err(|e| e.to_string())?;
            let doc = parse_query_json(raw.trim()).map_err(|e| e.to_string())?;
            validate_query(&doc).map_err(|e| e.to_string())?;
            let response = plan_query(&doc);
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
