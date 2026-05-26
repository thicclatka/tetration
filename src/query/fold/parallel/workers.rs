//! Rayon pool sizing for parallel folds.

use crate::query::fold::fold_policy;
use crate::query::types::ReadPlan;

/// Run `f` on a Rayon pool capped to `workers` when below the global thread count.
pub(crate) fn with_fold_workers<R>(workers: Option<usize>, f: impl FnOnce() -> R + Send) -> R
where
    R: Send,
{
    if let Some(n) = workers.filter(|&n| n > 0 && n < rayon::current_num_threads()) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("fold rayon pool")
            .install(f)
    } else {
        f()
    }
}

/// Use Rayon when policy requests parallel fold and more than one chunk is touched.
#[must_use]
pub(crate) fn use_parallel_fold(plan: &ReadPlan, policy: &fold_policy::FoldIoPolicy) -> bool {
    !policy.linear_scan && policy.parallel && plan.chunks.len() > 1
}
