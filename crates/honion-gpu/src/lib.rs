//! The device search kernel and its host driver.
//!
//! # Trust boundary
//!
//! The device is treated as an untrusted accelerator, not as an authority. It
//! receives only public data — starting points, and integer tables compiled
//! from a pattern — and it returns only *claims*: "thread `t`, iteration `k`
//! looked like a match". Nothing it says is acted on until the host has
//! re-derived the key from its own secret material and re-checked the address
//! with [`honion_core`]. A miscompiled kernel, a bit flip, or an outright
//! malicious kernel substitution can therefore waste time, but cannot cause a
//! wrong key to be written.
//!
//! No secret ever crosses into device memory. See `docs/05-security-model.md`.
//!
//! # Backends
//!
//! The crate hosts one device backend per platform, selected by a Cargo
//! feature: `cuda` (NVIDIA, the reference backend) or `metal` (Apple Silicon).
//! `honion-cli` picks the right one per platform; building this crate with
//! neither is a configuration error, caught below rather than as a cascade of
//! unresolved names. The trust boundary above is backend-independent — it is
//! a property of what crosses to *any* device, not of a driver.

#[cfg(not(any(feature = "cuda", feature = "metal")))]
compile_error!(
    "honion-gpu needs a device backend: build with `--features cuda` (NVIDIA) \
     or `--features metal` (Apple Silicon)."
);

#[cfg(feature = "metal")]
pub mod msl;
#[cfg(feature = "cuda")]
pub mod nvrtc;
pub mod search;
pub mod tables;

pub use search::{
    DEFAULT_HALF, DeviceInfo, Hit, LaunchOutcome, SearchError, candidates_per_batch,
};
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use search::{Searcher, auto_threads, local_bytes_per_thread};
pub use tables::{DeviceTables, TableError};
