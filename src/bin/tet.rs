//! `tet` — CLI for JSON queries and foreign-format conversion.
//!
//! Command bodies live in [`src/bin/tet/`](tet/) (separate source files). Cargo only
//! builds `tet.rs` as the binary; siblings under `src/bin/` would become extra binaries.

#[path = "tet/args.rs"]
mod args;
#[path = "tet/convert.rs"]
mod convert;
#[path = "tet/info.rs"]
mod info;
#[path = "tet/qhist.rs"]
mod qhist;
#[path = "tet/query.rs"]
mod query;
#[path = "tet/util.rs"]
mod util;

use std::process::ExitCode;

use clap::Parser;
use tetration::query::parse_query_json;

use args::{Cli, Commands};
use convert::run_convert;
use info::{InfoRunOpts, run_info};
use qhist::run_qhist;
use query::{QueryRunOpts, run_query};
use util::{cli_error, read_query_payload, resolve_stdout};

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Commands::Info {
            path,
            json,
            quiet,
            all,
            layout,
            execution,
            datasets,
            chunks,
            show_footer_history,
            limit,
            dataset,
            grep,
        } => run_info(InfoRunOpts {
            path,
            json,
            quiet,
            all,
            layout,
            execution,
            datasets,
            chunks,
            show_footer_history,
            limit,
            dataset,
            grep,
        }),
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
        Commands::Qhist(args) => run_qhist(args.cmd, args.clear),
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
