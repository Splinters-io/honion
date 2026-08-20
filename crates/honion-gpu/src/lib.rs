//! The CUDA search kernel and its host driver.
//!
//! # Trust boundary
//!
//! The device is treated as an untrusted accelerator, not as an authority. It
//! receives only public data — starting points, and integer tables compiled
//! from a pattern — and it returns only *claims*: "thread `t`, iteration `k`
//! looked like a match". Nothing it says is acted on until the host has
//! re-derived the key from its own secret material and re-checked the address
//! with [`honion_core`]. A miscompiled kernel, a bit flip, or an outright
//! malicious PTX substitution can therefore waste time, but cannot cause a
//! wrong key to be written.
//!
//! No secret ever crosses into device memory. See `docs/05-security-model.md`.

pub mod nvrtc;
pub mod search;
pub mod tables;

pub use search::{
    DEFAULT_HALF, Hit, LaunchOutcome, SearchError, Searcher, auto_threads, candidates_per_batch,
    local_bytes_per_thread,
};
pub use tables::{DeviceTables, TableError};
