//! Clap argument definitions for `tet`.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use tetration::{HistorySettings, QueryOutputFormat};

#[derive(Parser)]
#[command(
    name = "tet",
    version,
    about = "Tetration CLI: JSON queries and format conversion"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Summarize a `.tet` file (default: dataset table). Use `--json` for full dump.
    Info {
        /// Path to `.tet` file.
        path: PathBuf,
        /// Full pretty JSON (superblock, catalog, chunks, history).
        #[arg(long)]
        json: bool,
        /// One-line summary on stdout.
        #[arg(short = 'q', long, conflicts_with = "json")]
        quiet: bool,
        /// All text sections (layout, execution, datasets, chunks, history).
        #[arg(long)]
        all: bool,
        /// Superblock / layout fields.
        #[arg(long)]
        layout: bool,
        /// Per-file execution defaults from the chunk index header.
        #[arg(long)]
        execution: bool,
        /// Dataset catalog table (default when no section flags).
        #[arg(long)]
        datasets: bool,
        /// Chunk index table (`-n` limits rows; `0` = all).
        #[arg(long)]
        chunks: bool,
        /// Convert / provenance footer events (`tet info --history`; not `tet qhist`).
        #[arg(long = "history")]
        show_footer_history: bool,
        /// Max chunk rows when `--chunks` or `--all` (default 32; `0` = all).
        #[arg(short = 'n', long, value_name = "N")]
        limit: Option<usize>,
        /// Case-insensitive substring on dataset name.
        #[arg(long)]
        dataset: Option<String>,
        /// Case-insensitive substring on dataset name or dtype.
        #[arg(long)]
        grep: Option<String>,
    },
    /// Run a JSON query (plan, execute, or both).
    ///
    /// Typical flows:
    ///   tet query q.json -t data.tet
    ///   tet query q.json -t data.tet -x -q
    ///   tet query '{"dataset":"f32","mean":[]}' -t data.tet -x
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
        /// stdout: full, json, stats, plan (`read_plan` only), quiet. Default: full.
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
    /// Recent `tet query` log (platform cache; not in `.tet` footer). Default: `list`.
    Qhist(QhistArgs),
}

/// `tet qhist` — platform query history (not `tet info --history` footer).
#[derive(Args)]
#[command(name = "qhist", visible_alias = "hist", about = "Recent tet query log (platform cache; not in .tet footer). Default: list")]
pub struct QhistArgs {
    #[command(subcommand)]
    pub cmd: Option<QhistCmd>,
    /// Remove the query history file (`list --clear` or bare `tet qhist --clear`).
    #[arg(long, global = true)]
    pub clear: bool,
}

#[derive(Subcommand)]
pub enum QhistCmd {
    /// List recent queries (default). Use `--all` for every retained row.
    List {
        /// Max rows to print (ignored when `--all` is set).
        #[arg(short = 'n', long, default_value_t = HistorySettings::default().cli_query_max)]
        limit: usize,
        /// Print all retained rows (up to `TET_QUERY_HISTORY_MAX` on disk).
        #[arg(long)]
        all: bool,
        /// Pretty JSON (full entries); default is a compact table.
        #[arg(long)]
        json: bool,
        /// Case-insensitive substring on dataset name.
        #[arg(long)]
        dataset: Option<String>,
        /// Case-insensitive substring on saved `.tet` path.
        #[arg(long)]
        tet: Option<String>,
        /// `x` / `execute` or `p` / `plan` (matches list `mode` column).
        #[arg(long, value_name = "x|p")]
        mode: Option<String>,
        /// Search dataset, tet path, and operation label.
        #[arg(long)]
        grep: Option<String>,
    },
    /// Re-run a saved query (`N`: 1 = newest; indices match filtered `list`).
    Run {
        /// History index (1 = newest).
        #[arg(value_name = "N")]
        index: usize,
        /// Override `.tet` from the saved row.
        #[arg(short = 't', long, value_name = "PATH")]
        tet: Option<PathBuf>,
        /// Force execute (needs `-t` on the row or this flag).
        #[arg(short = 'x', long, conflicts_with = "plan")]
        execute: bool,
        /// Plan only (ignore saved execute).
        #[arg(long, conflicts_with = "execute")]
        plan: bool,
        #[arg(long, value_enum, default_value_t = QueryStdoutFormat::Full, conflicts_with = "quiet")]
        format: QueryStdoutFormat,
        #[arg(short = 'q', long, conflicts_with = "format")]
        quiet: bool,
        #[arg(long, visible_alias = "preview-f32", value_name = "N")]
        preview: Option<usize>,
        #[arg(long)]
        spill_allow: Vec<PathBuf>,
        /// Same filters as `qhist list` (with positional `N`).
        #[arg(long)]
        dataset: Option<String>,
        #[arg(long, value_name = "PATH")]
        tet_filter: Option<String>,
        #[arg(long, value_name = "x|p")]
        mode: Option<String>,
        #[arg(long)]
        grep: Option<String>,
    },
}

/// How `tet query` prints success on stdout (errors always go to stderr).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum QueryStdoutFormat {
    /// Pretty JSON of the full `QueryResponse` (default).
    #[default]
    Full,
    /// Compact one-line JSON (scripts, `jq`).
    Json,
    /// Slim JSON: plan summary + aggregates, no chunk list or preview arrays.
    Stats,
    /// Slim JSON: catalog + `read_plan` summary only (no chunks, no execution).
    Plan,
    /// One human-readable line (`dataset=… op=… mean=…`).
    Quiet,
}

impl From<QueryStdoutFormat> for QueryOutputFormat {
    fn from(f: QueryStdoutFormat) -> Self {
        match f {
            QueryStdoutFormat::Full => Self::Full,
            QueryStdoutFormat::Json => Self::Json,
            QueryStdoutFormat::Stats => Self::Stats,
            QueryStdoutFormat::Plan => Self::Plan,
            QueryStdoutFormat::Quiet => Self::Quiet,
        }
    }
}
