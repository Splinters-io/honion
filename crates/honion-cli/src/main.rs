//! `honion` — a GPU vanity generator for Tor v3 onion addresses.
//!
//! # How a run is structured
//!
//! 1. Every pattern is parsed and compiled. Nothing else happens until they all
//!    succeed.
//! 2. The compiled tables are checked for prefilter selectivity, so a pattern
//!    that would swamp the verifier is refused up front rather than discovered
//!    as a mysterious collapse in throughput.
//! 3. Each epoch: fresh clamped scalars are drawn from the system CSPRNG, their
//!    public keys are derived on the CPU, and only those public keys are sent to
//!    the GPU.
//! 4. The GPU walks each key forward and reports positions that pass its
//!    prefilter.
//! 5. Every report is rebuilt from the host's own scalar, re-derived, re-matched
//!    and signature-checked before any file is written.
//!
//! The secret scalars never leave step 3's memory. See `docs/05-security-model.md`.

mod patterns;
mod progress;
mod search;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Shown at the end of `--help`.
///
/// The alphabet note earns its place: base32 has no `0`, `1`, `8` or `9`, and a
/// pattern containing one is rejected. That is the first thing most people hit.
const EXAMPLES: &str = "\
Examples:
  honion estimate --prefix carroll          how long would this take?
  honion search --prefix carroll --out ./keys
  honion search --prefix hon2on --prefix hon2en --out ./keys --count 5
  honion verify ./keys/carroll....onion     re-derive and check a result

Patterns are base32: a-z and 2-7 only. There is no 0, 1, 8 or 9 -- use o, l,
s, z instead. `?` matches any character, `[abc]` and `[^abc]` match a choice.

Each result is written to <out>/<address>.onion/ holding hs_ed25519_secret_key,
hs_ed25519_public_key and hostname, which Tor reads directly as a
HiddenServiceDir.";

/// Command-line arguments.
#[derive(Parser, Debug)]
#[command(
    name = "honion",
    version,
    about = "GPU vanity address generator for Tor v3 onion services",
    long_about = None,
    after_help = EXAMPLES,
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// What to do.
#[derive(Subcommand, Debug)]
enum Command {
    /// Search for keys whose addresses match a pattern.
    Search(search::SearchArgs),
    /// Report how much work a pattern needs, without searching.
    Estimate(search::EstimateArgs),
    /// Re-derive a hidden-service key and check it against its own hostname.
    ///
    /// Works on any directory in Tor's layout, not only ones honion produced.
    Verify {
        /// Directory containing `hs_ed25519_secret_key` and `hostname`.
        dir: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Search(args) => search::run_search(&args),
        Command::Estimate(args) => search::run_estimate(&args),
        Command::Verify { dir } => search::run_verify(&dir),
    }
}
