//! `tet` — CLI for JSON queries and foreign-format conversion.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use tetration::{
    CLI_QUERY_HISTORY_MAX, ConvertProgress, ConvertReport, SpillPathAllowlist,
    append_cli_query_history, clear_cli_query_history, convert_to_tet_with_progress,
    detect_convert_format, list_cli_query_history, mmap_file_read, parse_query_json,
    plan_query_empty, plan_query_with_tet_mmap_ex, read_tet_summary_v1, validate_query,
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
        /// After planning, mmap-read planned chunk payloads (raw or zstd `f32`); attach `execution` with capped `f32_preview`. With **`operation`** (`sum`, `mean`, `min`, `max`, `count`), aggregates use the full logical selection (`operation_*` / `operation_reduced_*`; scalar `axes: []` uses a single-pass fold).
        #[arg(long = "execute", default_value_t = false)]
        execute: bool,
        /// Max decoded `f32` values in `execution` when using `--execute` (default 64). Use `0` with a query `operation` to skip preview floats while still aggregating.
        #[arg(long = "preview-f32", value_name = "N")]
        preview_f32: Option<usize>,
        /// Additional allowed directory roots for spill (repeatable). Default roots: `.tet` parent, platform cache (`~/.cache/tetration` and `~/.local/cache/tetration` on Linux; `~/.local/cache/tetration` on macOS; Windows `%LOCALAPPDATA%\\tetration`; temp dirs).
        #[arg(long = "spill-allow", value_name = "DIR")]
        spill_allow: Vec<PathBuf>,
    },
    /// Convert HDF5 / `NetCDF` into `.tet` (format from input extension).
    Convert {
        /// Source array file (`.h5`/`.hdf5`/`.hdf`/`.he2`/`.he5`, `.nc`/`.netcdf`/`.nc4`/`.nc3`/`.cdf`, or recognizable signature).
        input: PathBuf,
        /// Destination `.tet` file.
        output: PathBuf,
        /// Parallel chunk read workers (`0` = host `available_parallelism`, capped at 64).
        #[arg(long = "jobs", default_value_t = 0)]
        jobs: usize,
    },
    /// List or clear recent `tet query` documents (platform cache; not stored in `.tet`).
    History {
        /// Max rows to print (default 10; file retains [`CLI_QUERY_HISTORY_MAX`]).
        #[arg(short = 'n', long, default_value_t = CLI_QUERY_HISTORY_MAX)]
        limit: usize,
        /// Remove the history file.
        #[arg(long)]
        clear: bool,
    },
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

fn finish_convert_report(
    pb: &indicatif::ProgressBar,
    label: &str,
    report: &ConvertReport,
) -> Result<(), String> {
    pb.finish_with_message(format!("{label} done in {:.2}s", report.elapsed_secs));
    let pretty = serde_json::to_string_pretty(report).map_err(|e| e.to_string())?;
    println!();
    println!("{pretty}");
    Ok(())
}

fn run_convert(input: &Path, output: &Path, jobs: usize) -> Result<(), String> {
    use indicatif::{ProgressBar, ProgressStyle};

    let format = detect_convert_format(input).map_err(|e| e.to_string())?;
    let label = match format {
        tetration::ConvertInputFormat::H5 => "HDF5 convert",
        tetration::ConvertInputFormat::Netcdf => "NetCDF convert",
    };
    let progress_prefix = match format {
        tetration::ConvertInputFormat::H5 => "HDF5 → .tet",
        tetration::ConvertInputFormat::Netcdf => "NetCDF → .tet",
    };

    let pb = ProgressBar::new(0);
    pb.set_style(
        ProgressStyle::with_template("{msg} [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
            .map_err(|e| e.to_string())?
            .progress_chars("=>-"),
    );
    pb.set_message(progress_prefix.to_owned());

    let progress = Some(|p: ConvertProgress| {
        if pb.length().unwrap_or(0) != p.chunks_total {
            pb.set_length(p.chunks_total);
        }
        pb.set_position(p.chunks_done);
        pb.set_message(format!("{progress_prefix} ({})", p.dataset));
    });

    let report =
        convert_to_tet_with_progress(input, output, jobs, progress).map_err(|e| e.to_string())?;
    finish_convert_report(&pb, label, &report)
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
            spill_allow,
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
            let mut spill_owned = None;
            if execute {
                if let Some(path) = tet.as_ref() {
                    spill_owned = Some(
                        SpillPathAllowlist::default_for_tet(path, spill_allow)
                            .map_err(|e| e.to_string())?,
                    );
                }
            } else if !spill_allow.is_empty() {
                return Err("`--spill-allow` requires `--execute` and `--tet`".into());
            }
            let response = if let Some(path) = tet.as_ref() {
                let path_display = path.display().to_string();
                let mmap = mmap_file_read(path).map_err(|e| e.to_string())?;
                plan_query_with_tet_mmap_ex(
                    &doc,
                    Some(path_display.as_str()),
                    &mmap,
                    preview,
                    spill_owned.as_ref(),
                )
                .map_err(|e| e.to_string())?
            } else {
                if execute {
                    return Err("`--execute` requires `--tet PATH` (mmap read needs a file)".into());
                }
                plan_query_empty(&doc)
            };
            let out = serde_json::to_string_pretty(&response).map_err(|e| e.to_string())?;
            println!("{out}");
            let tet_display = tet.as_ref().map(|p| p.display().to_string());
            if let Err(e) = append_cli_query_history(&doc, tet_display.as_deref(), execute) {
                eprintln!("warning: query history not saved: {e}");
            }
            Ok(())
        }
        Commands::History { limit, clear } => {
            if clear {
                clear_cli_query_history().map_err(|e| e.to_string())?;
                eprintln!("query history cleared");
                return Ok(());
            }
            let path = tetration::cli_query_history_path();
            let entries = list_cli_query_history(limit).map_err(|e| e.to_string())?;
            let out = serde_json::json!({
                "path": path.as_ref().map(|p| p.display().to_string()),
                "max_retained": CLI_QUERY_HISTORY_MAX,
                "entries": entries,
            });
            let pretty = serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?;
            println!("{pretty}");
            Ok(())
        }
        Commands::Convert {
            input,
            output,
            jobs,
        } => run_convert(&input, &output, jobs),
    }
}
