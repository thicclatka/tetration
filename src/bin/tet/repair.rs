//! `tet repair` — planned or applied in-place `.tet` fixes.

use std::path::PathBuf;

use tetration::repair::{
    RepairOptions, format_plan_json, format_plan_text, format_repair_json, format_repair_text,
    repair_plan, repair_tet_file,
};
use tetration::verify::verify_tet_file;

use crate::util::cli_error;

pub(crate) struct RepairRunOpts {
    pub path: PathBuf,
    pub json: bool,
    pub dry_run: bool,
    pub apply: Vec<String>,
}

pub(crate) fn run_repair(opts_ref: &RepairRunOpts) -> Result<(), String> {
    if opts_ref.apply.is_empty() {
        let verify = verify_tet_file(&opts_ref.path).map_err(cli_error)?;
        let plan = repair_plan(&opts_ref.path, &verify);
        let out = if opts_ref.json {
            format_plan_json(&plan).map_err(cli_error)?
        } else {
            format_plan_text(&plan)
        };
        print!("{out}");
        return Ok(());
    }

    let repair_opts = RepairOptions {
        dry_run: opts_ref.dry_run,
        apply: opts_ref.apply.clone(),
        plan_codes: Vec::new(),
    };
    let report = repair_tet_file(&opts_ref.path, &repair_opts).map_err(cli_error)?;

    let out = if opts_ref.json {
        format_repair_json(&report).map_err(cli_error)?
    } else {
        format_repair_text(&report)
    };
    print!("{out}");

    if !report.dry_run && report.verify_after_ok == Some(false) {
        Err("repair finished but verification still failed".to_owned())
    } else {
        Ok(())
    }
}
