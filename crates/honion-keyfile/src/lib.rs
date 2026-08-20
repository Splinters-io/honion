//! Secret scalar handling, hit verification, and Tor hidden-service key files.
//!
//! This crate is where secret material lives. It is deliberately separate from
//! [`honion_gpu`](../honion_gpu/index.html), which never links against it: the
//! search kernel and its host driver deal only in public points and integer
//! tables, so there is no code path by which a secret could reach the device.
//!
//! The flow is:
//!
//! 1. [`SecretScalar::generate`] draws a clamped scalar per search thread.
//! 2. Its public key goes to the GPU; the scalar stays here.
//! 3. When the device reports `(thread, k)`, [`VerifiedKey::verify`] rebuilds
//!    `scalar + 8k` and proves the result before anything is written.
//! 4. [`write_service_dir`] serialises it in the layout Tor expects.

pub mod key;
pub mod scalar;
pub mod store;

pub use key::{ExpandedSecretKey, VerifiedKey, VerifyError};
pub use scalar::{SCALAR_LEN, ScalarError, SecretScalar};
pub use store::{StoreError, write_service_dir};
