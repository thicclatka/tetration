//! `tet` — CLI for JSON queries and foreign-format conversion.
//!
//! Command bodies live in [`src/bin/tet/`](tet/) (separate source files). Cargo only
//! builds `tet.rs` as the binary; siblings under `src/bin/` would become extra binaries.

#[path = "tet/args.rs"]
mod args;
#[path = "tet/convert.rs"]
mod convert;
#[path = "tet/history.rs"]
mod history;
#[path = "tet/query.rs"]
mod query;
#[path = "tet/util.rs"]
mod util;

use std::process::ExitCode;

use clap::Parser;
use tetration::{mmap_file_read, parse_query_json, read_tet_summary_v1};

use args::{Cli, Commands};
use convert::run_convert;
use history::run_history;
use query::{QueryRunOpts, run_query};
use util::{cli_error, read_query_payload, resolve_stdout};

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Commands::Info { path } => {
            let mmap = mmap_file_read(&path).map_err(cli_error)?;
            let summary = read_tet_summary_v1(&mmap).map_err(cli_error)?;
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
            let stdout = resolve_stdout(quiet, format);
            let raw = read_query_payload(query.as_deref()).map_err(cli_error)?;
            let doc = parse_query_json(raw.trim()).map_err(cli_error)?;
            run_query(QueryRunOpts {
                doc,
                tet,
                execute,
                stdout,
                preview,
                spill_allow,
                record_history: true,
            })
        }
        Commands::History { cmd, clear } => run_history(cmd, clear),
        Commands::Convert {
            input,
            output,
            jobs,
        } => run_convert(&input, &output, jobs),
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
