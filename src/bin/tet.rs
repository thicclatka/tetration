//! `tet` — CLI for JSON queries and foreign-format conversion.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use tetration::{
    CLI_QUERY_HISTORY_MAX, ConvertProgress, ConvertReport, QueryOutputFormat, SpillPathAllowlist,
    append_cli_query_history, clear_cli_query_history, convert_to_tet_with_progress,
    detect_convert_format, format_query_response, list_cli_query_history, mmap_file_read,
    parse_query_json, plan_query_empty, plan_query_with_tet_mmap_ex, read_tet_summary_v1,
    validate_query,
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
    /// Run a JSON query (plan, execute, or both).
    ///
    /// Typical flows:
    ///   tet query q.json -t data.tet
    ///   tet query q.json -t data.tet -x -q
    ///   tet query '{"dataset":"f32","operation":{"mean":{"axes":[]}}}' -t data.tet -x
    #[command(
        visible_alias = "q",
        after_help = "QUERY: path to .json, inline JSON, or `-` for stdin; omit QUERY to read stdin. \
                      -x decodes chunks (requires -t). -q is --format quiet; else default full."
    )]
    Query {
        /// Query document: `.json` path, inline JSON, or `-` for stdin.
        #[arg(value_name = "QUERY")]
        query: Option<String>,
        /// `.tet` file (catalog + optional execute).
        #[arg(short = 't', long, value_name = "PATH")]
        tet: Option<PathBuf>,
        /// Decode planned chunks and attach `execution` (requires `-t`).
        #[arg(short = 'x', long, requires = "tet")]
        execute: bool,
        /// stdout: full (pretty JSON), json, stats (slim), quiet (one line). Default: full.
        #[arg(long, value_enum, default_value_t = QueryStdoutFormat::Full, conflicts_with = "quiet")]
        format: QueryStdoutFormat,
        /// Shorthand for `--format quiet` (one-line stdout).
        #[arg(short = 'q', long, conflicts_with = "format")]
        quiet: bool,
        /// Sample values in JSON when executing (all dtypes). Default: 64 (full/json), 0 (quiet/stats).
        #[arg(long, visible_alias = "preview-f32", value_name = "N")]
        preview: Option<usize>,
        /// Extra spill directory roots (repeatable; needs `-x` and `-t`).
        #[arg(long, requires = "execute", requires = "tet")]
        spill_allow: Vec<PathBuf>,
    },
    /// Convert HDF5 / `NetCDF` / Zarr v3 directory store into `.tet` (format from input extension or sniff).
    Convert {
        /// Source array file (`.h5`/`.hdf5`/`.hdf`/`.he2`/`.he5`, `.nc`/`.netcdf`/`.nc4`/`.nc3`/`.cdf`, Zarr v3 directory with root `zarr.json`, or recognizable signature).
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

/// How `tet query` prints success on stdout (errors always go to stderr).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum QueryStdoutFormat {
    /// Pretty JSON of the full `QueryResponse` (default).
    #[default]
    Full,
    /// Compact one-line JSON (scripts, `jq`).
    Json,
    /// Slim JSON: plan summary + aggregates, no chunk list or preview arrays.
    Stats,
    /// One human-readable line (`dataset=… op=… mean=…`).
    Quiet,
}

impl From<QueryStdoutFormat> for QueryOutputFormat {
    fn from(f: QueryStdoutFormat) -> Self {
        match f {
            QueryStdoutFormat::Full => Self::Full,
            QueryStdoutFormat::Json => Self::Json,
            QueryStdoutFormat::Stats => Self::Stats,
            QueryStdoutFormat::Quiet => Self::Quiet,
        }
    }
}

/// Default preview cap when `--preview` is omitted.
const QUERY_PREVIEW_DEFAULT: usize = 64;

fn resolve_execute_preview_limit(
    execute: bool,
    stdout: QueryOutputFormat,
    explicit: Option<usize>,
) -> Result<Option<usize>, String> {
    if !execute {
        if explicit.is_some() {
            return Err("--preview requires -x / --execute".into());
        }
        return Ok(None);
    }
    Ok(Some(explicit.unwrap_or(match stdout {
        QueryOutputFormat::Full | QueryOutputFormat::Json => QUERY_PREVIEW_DEFAULT,
        QueryOutputFormat::Stats | QueryOutputFormat::Quiet => 0,
    })))
}

fn read_stdin_string() -> io::Result<String> {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

/// Read query JSON from a positional arg (file path or inline JSON) or stdin.
fn read_query_payload(query: Option<&str>) -> io::Result<String> {
    let Some(arg) = query else {
        return read_stdin_string();
    };
    if arg == "-" {
        return read_stdin_string();
    }
    let path = Path::new(arg);
    if path.is_file() {
        fs::read_to_string(path)
    } else {
        Ok(arg.to_owned())
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
        tetration::ConvertInputFormat::Zarr => "Zarr convert",
    };
    let progress_prefix = match format {
        tetration::ConvertInputFormat::H5 => "HDF5 → .tet",
        tetration::ConvertInputFormat::Netcdf => "NetCDF → .tet",
        tetration::ConvertInputFormat::Zarr => "Zarr → .tet",
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
            query,
            tet,
            execute,
            format,
            quiet,
            preview,
            spill_allow,
        } => {
            let stdout = if quiet {
                QueryOutputFormat::Quiet
            } else {
                format.into()
            };
            let preview = resolve_execute_preview_limit(execute, stdout, preview)?;
            let raw = read_query_payload(query.as_deref()).map_err(|e| e.to_string())?;
            let doc = parse_query_json(raw.trim()).map_err(|e| e.to_string())?;
            validate_query(&doc).map_err(|e| e.to_string())?;
            let mut spill_owned = None;
            if execute {
                let path = tet.as_ref().expect("clap requires --tet when -x is set");
                spill_owned = Some(
                    SpillPathAllowlist::default_for_tet(path, spill_allow)
                        .map_err(|e| e.to_string())?,
                );
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
                plan_query_empty(&doc)
            };
            let out = format_query_response(&response, stdout)?;
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
