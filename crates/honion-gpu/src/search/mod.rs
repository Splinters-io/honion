//! Driving the search kernel: the backend-neutral surface.
//!
//! # What crosses the boundary
//!
//! Into the device: compressed public keys (32 bytes each), and the integer
//! tables built by [`crate::tables`]. Out of the device: `(thread, iteration,
//! pattern)` triples and a status word.
//!
//! No secret goes in and no key comes out. The caller holds the secret scalars
//! that generated the starting points, and reconstructs a hit's secret as
//! `a0[thread] + 8 * iteration` from its own memory. This is why the kernel can
//! be treated as an untrusted accelerator: the worst a broken one can do is
//! waste time or report candidates that fail verification.
//!
//! # The backend seam
//!
//! Everything in this module is meaningful for *any* device backend: what a
//! hit is, what a launch produces, how batches are sized, how failures are
//! reported. The pieces that talk to an actual driver — the `Searcher`, the
//! device sizing in `auto_threads`, the per-thread memory accounting — live in
//! a backend submodule and are re-exported from here, so consumers name
//! `honion_gpu::Searcher` and never a backend.

// @decision DEC-BACKEND-001
// @title Backend-agnostic Searcher facade with feature-selected backend modules
// @status accepted
// @rationale The Searcher is constructed once per run and its methods are
//   called a handful of times per multi-second launch — nowhere near the
//   per-candidate hot path — so a seam here costs nothing at run time while
//   letting `honion-cli` and the tests compile against one stable surface on
//   both platforms. When both features are enabled (a cross-check build, not a
//   shipping configuration), CUDA wins the re-export and the Metal module is
//   compiled out entirely, so there is never an ambiguous `Searcher`.

/// The NVIDIA CUDA backend: NVRTC-compiled kernel, `cudarc` host driver.
#[cfg(feature = "cuda")]
pub mod cuda;

/// The Apple Metal backend: runtime-compiled MSL kernels over the `msl.rs`
/// driver, on unified-memory buffers.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub mod metal;

#[cfg(feature = "cuda")]
pub use cuda::{Searcher, auto_threads, local_bytes_per_thread};
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub use metal::{Searcher, auto_threads, local_bytes_per_thread};

/// Positive offsets in the kernel's precomputed table.
///
/// Each offset yields two candidates — `base + off` and `base - off` — via the
/// dual addition law, so a batch covers `2 * HALF + 1` candidates: the base
/// point and `HALF` pairs either side of it.
///
/// Compiled into the kernel, so changing it changes the generated code rather
/// than a runtime parameter. Larger values amortise the batch's single modular
/// inversion further, but need proportionally more per-thread local memory and
/// more shared memory for the table, both of which cost occupancy.
pub const DEFAULT_HALF: u32 = 512;

/// Candidates a thread examines per batch, for a given `half`.
#[must_use]
pub const fn candidates_per_batch(half: u32) -> u32 {
    2 * half + 1
}

/// A candidate the device believes matches.
///
/// "Believes" is exact: this is a claim to be checked, not a result. The
/// `pattern_id` in particular is advisory — verification re-derives which
/// patterns actually match rather than trusting it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct Hit {
    /// Index of the thread that found it; selects the starting scalar.
    pub thread_id: u32,
    /// Signed step count from that thread's starting scalar: the `m` in
    /// `a = a0 + 8m`.
    ///
    /// Signed because the search covers a symmetric range either side of each
    /// starting point — the dual addition law produces `base + off` and
    /// `base - off` together — so a match may lie below where the thread
    /// started.
    pub offset: i32,
    /// Which pattern the device matched. Advisory.
    pub pattern_id: u32,
    /// Padding, so the layout matches the device struct exactly.
    pub reserved: u32,
}

/// What one launch produced.
#[derive(Clone, Debug)]
pub struct LaunchOutcome {
    /// Candidates to verify.
    pub hits: Vec<Hit>,
    /// Candidates the device found, which may exceed `hits.len()` if the
    /// buffer overflowed.
    pub total_found: u32,
    /// Public keys examined.
    pub examined: u64,
}

// @decision DEC-BACKEND-008
// @title DeviceInfo replaces compute_capability on the backend-agnostic surface
// @status accepted
// @rationale Compute capability is a CUDA-only concept that used to leak into
//   the CLI's output. Each backend instead describes its device in its own
//   terms — a Metal GPU has a name and a family, not a capability tuple — and
//   the CLI prints the description without knowing which backend produced it.
//   `compute_capability()` survives as a CUDA-only inherent method for the
//   tests and tools that genuinely mean CUDA.

/// A backend-neutral description of the device a search will run on.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    /// The backend that owns the device: `"CUDA"` or `"Metal"`.
    pub backend: &'static str,
    /// The device's own name, as its driver reports it.
    pub name: String,
    /// Backend-specific detail worth showing a user — e.g. the compute
    /// capability on CUDA. May be empty.
    pub detail: String,
}

impl DeviceInfo {
    /// One line suitable for a progress header.
    #[must_use]
    pub fn description(&self) -> String {
        if self.detail.is_empty() {
            format!("{} ({})", self.name, self.backend)
        } else {
            format!("{} ({}, {})", self.name, self.backend, self.detail)
        }
    }

    /// A placeholder for when the device cannot be queried.
    ///
    /// Querying can fail for reasons that should not abort a search that is
    /// otherwise working — the description is cosmetic — so callers fall back
    /// to this rather than propagating the error.
    #[must_use]
    pub fn unknown() -> Self {
        Self {
            backend: "unknown",
            name: "unknown device".into(),
            detail: String::new(),
        }
    }
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.description())
    }
}

/// Why a search could not be set up or run.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SearchError {
    /// The device driver reported a failure.
    #[error("device driver error: {0}")]
    Driver(String),
    /// Device code failed to compile.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Nvrtc(#[from] crate::nvrtc::NvrtcError),
    /// The kernel raised a status flag.
    #[error("device reported a fault: {0}")]
    Device(String),
    /// A construction parameter was out of range.
    #[error("{0}")]
    BadParameter(String),
    /// The wrong number of starting points was supplied.
    #[error("expected {expected} starting points, got {found}")]
    WrongPointCount {
        /// Points the searcher was configured for.
        expected: usize,
        /// Points supplied.
        found: usize,
    },
}
