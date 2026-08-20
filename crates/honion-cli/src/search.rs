//! The `search`, `estimate` and `verify` subcommands.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Args;
use honion_core::address::OnionAddress;
use honion_core::pattern::{Pattern, PatternSet};
use honion_gpu::{DeviceTables, Searcher};
use honion_keyfile::{SecretScalar, VerifiedKey, write_service_dir};
use rayon::prelude::*;

use crate::patterns::{self, SourcedPattern};
use crate::progress::{Progress, humantime, si};

/// Default number of precomputed offsets.
///
/// Each offset yields two candidates via the dual addition law, so a batch
/// covers `2 * 512 + 1 = 1025` candidates for one modular inversion.
/// See docs/06-performance.md for the tuning curve.
const DEFAULT_OFFSETS: u32 = 512;

/// Minimum prefilter selectivity, in bits.
///
/// Below this the device forwards candidates faster than the host can verify
/// them. Twenty bits leaves roughly a thousand per second at full throughput.
const DEFAULT_MIN_SELECTIVITY: f64 = 20.0;

/// How long each launch should take. Progress is reported between launches, so
/// this is also the update interval and the worst-case delay before a found key
/// is written.
const DEFAULT_LAUNCH_SECONDS: f64 = 4.0;

/// Arguments to `honion search`.
#[derive(Args, Debug)]
pub struct SearchArgs {
    /// Pattern to search for. May be repeated.
    ///
    /// A pattern is base32 characters (a-z, 2-7), `?` for any character, and
    /// `[abc]` or `[^abc]` for a choice. It matches the start of the address.
    #[arg(long = "prefix", value_name = "PATTERN")]
    prefixes: Vec<String>,

    /// File of patterns, one per line; `#` comments and blank lines ignored.
    #[arg(long, value_name = "FILE")]
    patterns_file: Option<PathBuf>,

    /// Directory to write results into. One subdirectory per address.
    #[arg(long, short, value_name = "DIR")]
    out: PathBuf,

    /// Stop after this many matches. Zero means run until interrupted.
    #[arg(long, default_value_t = 1)]
    count: usize,

    /// Concurrent walks. Defaults to as many as fit comfortably in free device
    /// memory, which is where throughput plateaus.
    #[arg(long)]
    threads: Option<u32>,

    /// Precomputed offsets. Each yields two candidates, so a batch covers
    /// `2N+1` keys per modular inversion. Larger uses more memory.
    #[arg(long, default_value_t = DEFAULT_OFFSETS)]
    offsets: u32,

    /// Seconds of GPU work per launch, which is also the progress interval.
    #[arg(long, default_value_t = DEFAULT_LAUNCH_SECONDS)]
    launch_seconds: f64,

    /// Minimum prefilter selectivity in bits.
    #[arg(long, default_value_t = DEFAULT_MIN_SELECTIVITY)]
    min_selectivity: f64,

    /// Suppress progress output.
    #[arg(long, short)]
    quiet: bool,
}

/// Arguments to `honion estimate`.
#[derive(Args, Debug)]
pub struct EstimateArgs {
    /// Pattern to estimate. May be repeated.
    #[arg(long = "prefix", value_name = "PATTERN")]
    prefixes: Vec<String>,

    /// File of patterns, one per line.
    #[arg(long, value_name = "FILE")]
    patterns_file: Option<PathBuf>,

    /// Assumed search rate in addresses per second.
    ///
    /// Defaults to the rate measured on an RTX PRO 6000 Blackwell over 300 runs;
    /// see
    /// docs/07-benchmarks.md. Pass your own to estimate for a different card.
    #[arg(long, default_value_t = 1.2514e10)]
    rate: f64,
}

/// Gather patterns from both sources, failing on the first malformed one.
fn collect(prefixes: &[String], file: Option<&PathBuf>) -> Result<Vec<SourcedPattern>> {
    let mut all = patterns::from_args(prefixes)?;
    if let Some(path) = file {
        all.extend(patterns::from_file(path)?);
    }
    if all.is_empty() {
        bail!("no patterns given; use --prefix or --patterns-file");
    }
    Ok(all)
}

/// Compile a pattern set from sourced patterns.
fn compile(sourced: &[SourcedPattern]) -> Result<PatternSet> {
    let list: Vec<Pattern> = sourced.iter().map(|s| s.pattern.clone()).collect();
    PatternSet::compile(&list).context("compiling patterns")
}

/// `honion estimate`.
///
/// # Errors
///
/// If patterns are malformed.
pub fn run_estimate(args: &EstimateArgs) -> Result<()> {
    let sourced = collect(&args.prefixes, args.patterns_file.as_ref())?;
    let set = compile(&sourced)?;
    let tables = DeviceTables::build(&set);

    println!("patterns: {}", sourced.len());
    for s in &sourced {
        println!(
            "  {:<24} {:>6.1} bits  expected {:>10}   [{}]",
            s.pattern.source(),
            s.pattern.difficulty_log2(),
            humantime(2f64.powf(s.pattern.difficulty_log2()) / args.rate),
            s.origin,
        );
    }
    let d = set.difficulty_log2();
    println!();
    println!("combined difficulty : {d:.1} bits ({} expected trials)", si(2f64.powf(d)));
    println!("assumed rate        : {}/s", si(args.rate));
    println!("expected time       : {}", humantime(2f64.powf(d) / args.rate));
    println!(
        "prefilter           : {:.1} bits of selectivity",
        tables.prefilter_selectivity_log2()
    );
    println!();
    println!("The search is memoryless: each key is an independent trial, so the");
    println!("expected time is a mean, not a deadline. There is a 63% chance of a");
    println!("result by then, 86% by twice that, and 95% by three times.");
    Ok(())
}

/// `honion search`.
///
/// # Errors
///
/// If patterns are malformed, the prefilter is too weak, no GPU is available,
/// or a result cannot be written.
pub fn run_search(args: &SearchArgs) -> Result<()> {
    let sourced = collect(&args.prefixes, args.patterns_file.as_ref())?;
    let set = compile(&sourced)?;
    let tables = DeviceTables::build(&set);

    // Refuse a hopeless configuration before allocating anything.
    tables
        .check_selectivity(args.min_selectivity)
        .context("this pattern set cannot be searched efficiently")?;

    if args.offsets == 0 {
        bail!("--offsets must be positive");
    }
    // Sized to the card unless the user pinned it: throughput rises with
    // concurrent walks until their local memory stops fitting, so the right
    // number is a property of the device, not a constant.
    let threads = match args.threads {
        Some(0) => bail!("--threads must be positive"),
        Some(n) => n,
        None => honion_gpu::auto_threads(args.offsets).context("sizing the search to the device")?,
    };
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating output directory {}", args.out.display()))?;

    let difficulty = set.difficulty_log2();
    if !args.quiet {
        eprintln!("honion: searching for {} pattern(s)", sourced.len());
        for s in &sourced {
            eprintln!(
                "  {} ({:.1} bits, from {})",
                s.pattern.source(),
                s.pattern.difficulty_log2(),
                s.origin
            );
        }
        eprintln!(
            "  combined difficulty {difficulty:.1} bits, prefilter {:.1} bits",
            tables.prefilter_selectivity_log2()
        );
    }

    // The hit buffer is sized for a very unlucky launch. Overflow is reported
    // rather than silent, but it should never happen.
    let max_hits = 4096u32;
    let mut searcher = Searcher::new(&tables, threads, args.offsets, max_hits)
        .context("preparing the GPU search")?;
    if !args.quiet {
        let (major, minor) = searcher.compute_capability().unwrap_or((0, 0));
        eprintln!(
            "  device compute capability {major}.{minor}, {} threads, \
             {} offsets ({} candidates per inversion, {:.1} GB device memory)",
            searcher.num_threads(),
            searcher.half(),
            searcher.candidates_per_batch(),
            honion_gpu::local_bytes_per_thread(searcher.half()) as f64
                * f64::from(searcher.num_threads()) / 1e9
        );
    }

    let mut progress = Progress::new(difficulty, args.quiet);
    // Start small so the first launch cannot run for an unexpectedly long time
    // on a slow device; the rate measured from it sizes every later launch.
    let mut candidates: u32 = 4096;
    let mut written = 0usize;

    // Draw the first epoch's scalars. They live only here, in host memory; the
    // device receives their public keys and nothing else.
    let mut scalars = draw_scalars(threads)?;
    searcher
        .set_start_points(&public_keys(&scalars))
        .context("uploading starting points")?;

    loop {
        let launch_start = std::time::Instant::now();
        let batches = searcher
            .launch_async(candidates)
            .context("running the search kernel")?;

        // The GPU is busy; draw the next epoch's scalars while it works. On a
        // large card this is hundreds of milliseconds that would otherwise be
        // dead time before every launch.
        let next_scalars = draw_scalars(threads)?;
        let next_points = public_keys(&next_scalars);

        let outcome = searcher.collect(batches).context("collecting search results")?;
        let launch_time = launch_start.elapsed().as_secs_f64();

        if outcome.total_found as usize > outcome.hits.len() {
            eprintln!(
                "warning: the device found {} candidates but only {} fit the buffer; \
                 {} were dropped. Raise --min-selectivity or use a longer pattern.",
                outcome.total_found,
                outcome.hits.len(),
                outcome.total_found as usize - outcome.hits.len()
            );
        }

        // Verify every candidate against the host's own arithmetic before it is
        // allowed to become a file.
        for hit in &outcome.hits {
            let Some(start) = scalars.get(hit.thread_id as usize) else {
                bail!(
                    "device reported thread {} but only {} were launched",
                    hit.thread_id,
                    scalars.len()
                );
            };
            let verified =
                VerifiedKey::verify(start, i64::from(hit.offset), &set).with_context(|| {
                    format!(
                        "verifying the candidate reported at thread {} offset {}. \
                         This means the GPU and the CPU disagree, which is a bug or a \
                         hardware fault, not a user error.",
                        hit.thread_id, hit.offset
                    )
                })?;

            let dir = write_service_dir(&args.out, &verified)
                .context("writing the hidden service directory")?;
            written += 1;
            println!("{}", verified.address().to_hostname());
            if !args.quiet {
                eprintln!("  written to {}", dir.display());
            }
            if args.count != 0 && written >= args.count {
                progress.record(outcome.examined, outcome.hits.len());
                if !args.quiet {
                    progress.report();
                    eprintln!(
                        "honion: wrote {written} key(s); the device reported {} candidate(s) \
                         while examining {} addresses in {}",
                        progress.found(),
                        si(progress.examined() as f64),
                        humantime(progress.elapsed().as_secs_f64())
                    );
                }
                return Ok(());
            }
        }

        progress.record(outcome.examined, outcome.hits.len());
        progress.report();

        // Hand the prepared epoch to the device and keep its scalars, which the
        // next round's hits will be reconstructed from.
        searcher
            .set_start_points(&next_points)
            .context("uploading starting points")?;
        scalars = next_scalars;

        // Size the next launch from the rate just measured, so the loop settles
        // on the requested wall-clock interval regardless of device speed.
        if launch_time > 0.0 {
            let per_candidate = launch_time / f64::from(candidates);
            let want = (args.launch_seconds / per_candidate).round();
            candidates = want.clamp(1.0, f64::from(u32::MAX / 2)) as u32;
        }
    }
}

/// Draw one clamped secret scalar per thread from the system CSPRNG.
fn draw_scalars(threads: u32) -> Result<Vec<SecretScalar>> {
    (0..threads)
        .into_par_iter()
        .map(|_| SecretScalar::generate())
        .collect::<Result<Vec<_>, _>>()
        .context("drawing secret scalars from the system CSPRNG")
}

/// The public keys of those scalars. Only these reach the device.
fn public_keys(scalars: &[SecretScalar]) -> Vec<[u8; 32]> {
    scalars.par_iter().map(SecretScalar::public_key).collect()
}

/// `honion verify`.
///
/// Re-derives the address from the stored secret key and checks it against the
/// stored `hostname`, so a directory can be checked long after it was made —
/// including one produced by other tools.
///
/// # Errors
///
/// If the directory is incomplete, malformed, or internally inconsistent.
pub fn run_verify(dir: &Path) -> Result<()> {
    let secret_path = dir.join("hs_ed25519_secret_key");
    let hostname_path = dir.join("hostname");

    let secret = std::fs::read(&secret_path)
        .with_context(|| format!("reading {}", secret_path.display()))?;
    if secret.len() != 96 {
        bail!(
            "{} is {} bytes, expected 96 (a 32-byte tag and a 64-byte expanded key)",
            secret_path.display(),
            secret.len()
        );
    }
    let scalar_bytes: [u8; 32] = secret[32..64].try_into().expect("slice is 32 bytes");
    let scalar = SecretScalar::from_clamped(scalar_bytes)
        .context("the stored scalar is not in clamped form")?;

    let public = scalar.public_key();
    let derived = OnionAddress::from_pubkey(&public);

    let stored = std::fs::read_to_string(&hostname_path)
        .with_context(|| format!("reading {}", hostname_path.display()))?;
    let stored = stored.trim();
    if stored != derived.to_hostname() {
        bail!(
            "mismatch: {} says {stored}, but the secret key derives {}",
            hostname_path.display(),
            derived.to_hostname()
        );
    }

    // Confirm the public key file agrees too, if present.
    let public_path = dir.join("hs_ed25519_public_key");
    if public_path.exists() {
        let bytes = std::fs::read(&public_path)
            .with_context(|| format!("reading {}", public_path.display()))?;
        if bytes.len() != 64 || bytes[32..] != public[..] {
            bail!(
                "{} does not contain the public key implied by the secret key",
                public_path.display()
            );
        }
    }

    println!("{} verified: {}", dir.display(), derived.to_hostname());
    Ok(())
}
