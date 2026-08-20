//! Byte-exact Tor v3 onion address construction and the vanity pattern language.
//!
//! This crate holds the entire correctness surface of `honion`. It is pure CPU
//! code with no GPU dependency, so every claim it makes can be tested directly.
//! The GPU search kernel is, by design, only an accelerated filter for the
//! predicate defined here — [`pattern::CompiledPattern::matches_pubkey`] is the
//! reference semantics that the device must agree with, and every hit the
//! device reports is re-checked against it before a key is written.
//!
//! # Module map
//!
//! - [`base32`] — the one and only base32 codec.
//! - [`address`] — `pubkey -> onion address`, and address recognition.
//! - [`pattern`] — the vanity pattern language: grammar, parser, compiler.

// The crate-level lints in Cargo.toml keep the library total: no indexing that
// could panic, no `unwrap`, no `expect`. Tests are the one place where
// panicking *is* the correct response to an unexpected value, so they are
// relaxed rather than contorted around the rule.
#![cfg_attr(
    test,
    allow(
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::expect_used
    )
)]

pub mod address;
pub mod base32;
pub mod pattern;

pub use address::{OnionAddress, PUBKEY_LEN};
