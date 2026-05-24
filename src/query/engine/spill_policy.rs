//! Host-controlled spill path allowlist (not set from query JSON).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::query::types::TetError;

/// Allowed directory roots for query JSON `spill` paths.
///
/// Default policy (see [`Self::default_for_tet`]): `.tet` parent directory plus OS cache/scratch
/// locations. Relative spill handles resolve against the `.tet` parent directory.
#[derive(Debug, Clone)]
pub struct SpillPathAllowlist {
    roots: Vec<PathBuf>,
    relative_base: PathBuf,
}

impl SpillPathAllowlist {
    /// Build the default allowlist for a query against `tet_path`.
    ///
    /// Roots: canonical `.tet` parent, platform cache dirs ([`platform_cache_roots`]), then
    /// `extra_roots` from `--spill-allow` (union, deduped).
    ///
    /// # Errors
    ///
    /// Returns [`TetError::Validation`] when the `.tet` parent or a cache root cannot be resolved.
    pub fn default_for_tet(
        tet_path: &Path,
        extra_roots: impl IntoIterator<Item = PathBuf>,
    ) -> Result<Self, TetError> {
        let relative_base = tet_parent_dir(tet_path)?;
        let mut roots = Vec::new();
        push_root(&mut roots, &relative_base)?;
        for r in platform_cache_roots() {
            try_push_root(&mut roots, &r);
        }
        for r in extra_roots {
            push_root(&mut roots, &r)?;
        }
        Ok(Self {
            roots,
            relative_base,
        })
    }

    /// Explicit roots for tests or custom hosts; relative spill handles use `relative_base`.
    #[must_use]
    pub fn from_roots(relative_base: PathBuf, roots: impl IntoIterator<Item = PathBuf>) -> Self {
        Self {
            roots: roots.into_iter().collect(),
            relative_base,
        }
    }

    /// True when no spill roots were configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    /// Allocate a unique temp spill path under platform cache roots (fallback: allowlist roots).
    ///
    /// Engine-managed temp spills are deleted when execution finishes; they are not export spill paths.
    ///
    /// # Errors
    ///
    /// Returns [`TetError::Validation`] when no writable root is available.
    pub fn allocate_temp_spill_path(&self) -> Result<PathBuf, TetError> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let file_name = format!("spill-{pid}-{nanos}.bin");

        for root in platform_cache_roots() {
            if let Ok(path) = try_allocate_temp_in_root(&root, &file_name) {
                return Ok(path);
            }
        }
        for root in &self.roots {
            if let Ok(path) = try_allocate_temp_in_root(root, &file_name) {
                return Ok(path);
            }
        }
        Err(TetError::Validation(
            "no writable temp spill directory under platform cache or spill allowlist roots".into(),
        ))
    }

    /// Resolve `path` (relative to [`.tet` parent](Self::default_for_tet)) and ensure it lies under a root.
    ///
    /// # Errors
    ///
    /// Returns [`TetError::Validation`] when resolution fails or the path is outside all roots.
    pub fn validate(&self, path: &Path) -> Result<PathBuf, TetError> {
        let resolved = resolve_spill_path(path, &self.relative_base)?;
        for root in &self.roots {
            let root_canon = canonicalize_root(root)?;
            if resolved.starts_with(&root_canon) {
                return Ok(resolved);
            }
        }
        Err(TetError::Validation(format!(
            "spill path {} is not under an allowed root (.tet directory, platform cache, or --spill-allow); resolved to {}",
            path.display(),
            resolved.display()
        )))
    }
}

fn tet_parent_dir(tet_path: &Path) -> Result<PathBuf, TetError> {
    tet_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(std::path::absolute)
        .ok_or_else(|| {
            TetError::Validation(format!(
                "cannot derive spill relative base from `.tet` path {}",
                tet_path.display()
            ))
        })?
        .map_err(|e| TetError::Validation(format!("`.tet` path resolution failed: {e}")))
}

fn push_root(roots: &mut Vec<PathBuf>, path: &Path) -> Result<(), TetError> {
    let canon = canonicalize_root(path)?;
    try_push_root(roots, &canon);
    Ok(())
}

fn try_push_root(roots: &mut Vec<PathBuf>, canon: &Path) {
    let canon = canon.to_path_buf();
    if roots.iter().all(|r| r != &canon) {
        roots.push(canon);
    }
}

fn try_allocate_temp_in_root(root: &Path, file_name: &str) -> Result<PathBuf, TetError> {
    if std::fs::create_dir_all(root).is_err() {
        return Err(TetError::Validation("temp spill root not writable".into()));
    }
    let path = root.join(file_name);
    std::fs::File::create(&path).map_err(|e| {
        TetError::Validation(format!(
            "temp spill path {} could not be created: {e}",
            path.display()
        ))
    })?;
    Ok(path)
}

fn canonicalize_root(path: &Path) -> Result<PathBuf, TetError> {
    if path.exists() {
        std::fs::canonicalize(path).map_err(|e| {
            TetError::Validation(format!(
                "spill root {} is not accessible: {e}",
                path.display()
            ))
        })
    } else {
        std::fs::create_dir_all(path).map_err(|e| {
            TetError::Validation(format!(
                "spill root {} could not be created: {e}",
                path.display()
            ))
        })?;
        std::fs::canonicalize(path).map_err(|e| {
            TetError::Validation(format!(
                "spill root {} could not be canonicalized: {e}",
                path.display()
            ))
        })
    }
}

fn resolve_spill_path(path: &Path, relative_base: &Path) -> Result<PathBuf, TetError> {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        relative_base.join(path)
    };
    let parent = abs
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = abs.file_name().ok_or_else(|| {
        TetError::Validation("spill path must name a file, not a directory".into())
    })?;
    let parent_canon = if parent.exists() {
        std::fs::canonicalize(parent).map_err(|e| {
            TetError::Validation(format!("spill path parent {}: {e}", parent.display()))
        })?
    } else {
        std::fs::create_dir_all(parent).map_err(|e| {
            TetError::Validation(format!(
                "spill path parent {} could not be created: {e}",
                parent.display()
            ))
        })?;
        std::fs::canonicalize(parent).map_err(|e| {
            TetError::Validation(format!(
                "spill path parent {} could not be canonicalized: {e}",
                parent.display()
            ))
        })?
    };
    Ok(parent_canon.join(file_name))
}

// --- Engine-managed temp spill (deleted when guard drops) ---

/// Temp spill file removed on [`Drop`] unless [`Self::keep`] is set.
pub struct TempSpillFile {
    path: PathBuf,
    keep: bool,
}

impl TempSpillFile {
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Allocate a unique spill path under platform cache roots (fallback: allowlist roots).
    ///
    /// # Errors
    ///
    /// Returns [`TetError::Validation`] when no writable temp root is available.
    pub fn create(allowlist: &SpillPathAllowlist) -> Result<Self, TetError> {
        let path = allowlist.allocate_temp_spill_path()?;
        Ok(Self { path, keep: false })
    }

    /// Construct a guard for integration tests (same drop cleanup as engine temp spills).
    #[doc(hidden)]
    #[must_use]
    pub fn with_path_for_test(path: PathBuf) -> Self {
        Self { path, keep: false }
    }
}

impl Drop for TempSpillFile {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Best-effort platform cache / scratch directories for spill (`…/tetration` subdirs).
fn platform_cache_roots() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();
    let mut push_candidate = |p: PathBuf| {
        if seen.insert(p.clone()) {
            candidates.push(p);
        }
    };

    if let Some(home) = user_home_dir() {
        if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
            push_candidate(PathBuf::from(xdg));
        } else if cfg!(target_os = "macos") {
            push_candidate(home.join(".local").join("cache"));
        } else {
            push_candidate(home.join(".cache"));
            push_candidate(home.join(".local").join("cache"));
        }
    }

    #[cfg(windows)]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        push_candidate(PathBuf::from(local));
    }

    for var in ["TMPDIR", "TEMP", "TMP"] {
        if let Ok(v) = std::env::var(var) {
            push_candidate(PathBuf::from(v));
        }
    }

    let mut out = Vec::new();
    for root in candidates {
        if let Ok(canon) = canonicalize_root(&root.join("tetration")) {
            try_push_root(&mut out, &canon);
        }
    }
    out
}

/// First resolved platform cache directory (`…/tetration`), shared by spill temps and CLI history.
pub(crate) fn platform_tetration_cache_dir() -> Option<PathBuf> {
    platform_cache_roots().into_iter().next()
}

fn user_home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE").map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}
