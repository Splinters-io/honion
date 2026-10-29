//! The `search`, `estimate` and `verify` subcommands.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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

/// Stack size for per-device search workers. Each worker compiles the kernel
/// with NVRTC and lets the driver JIT the PTX on its own thread; that code
/// recurses deeply enough to overflow the 2 MiB a spawned thread gets by
/// default. Measured crashing at 2 MiB on CUDA 12 NVRTC, comfortable at this.
const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

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

    /// CUDA devices to search on: `all`, or comma-separated device numbers.
    ///
    /// Each device runs its own search with its own fresh scalars, so the
    /// rates add. `--threads` is per device, not a total. Host preparation
    /// (drawing and deriving each epoch's starting keys) also runs per
    /// device, so a fully populated box wants CPU cores to match. Listing
    /// one device twice is allowed and time-slices two workers on it,
    /// which is slower than one worker and mostly a way to exercise the
    /// multi-worker path on a single-card machine.
    #[arg(long, default_value = "0", value_name = "DEV")]
    device: String,

    /// Stop after this many matches, across all devices. Zero means run until
    /// interrupted.
    #[arg(long, default_value_t = 1)]
    count: usize,

    /// Concurrent walks, per device. Defaults to as many as fit comfortably
    /// in that device's free memory, which is where throughput plateaus.
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

/// Resolve `--device` to the list of ordinals to search on.
///
/// Accepts `all` or a comma-separated list of numbers. Numbers are checked
/// against `available` now rather than failing later inside `Searcher::new`,
/// so `--device 3` on a two-card box says so before anything compiles.
/// Duplicates are allowed: two workers on one card are slow, but they are a
/// legitimate configuration and a useful way to exercise the multi-worker
/// path on single-card hardware.
fn parse_devices(spec: &str, available: u32) -> Result<Vec<u32>> {
    let devices = if spec.trim().eq_ignore_ascii_case("all") {
        (0..available).collect()
    } else {
        spec.split(',')
            .map(|part| -> Result<u32> {
                let entry = part.trim();
                if entry.is_empty() {
                    bail!("--device: {spec:?} has an empty entry");
                }
                let ordinal: u32 = entry
                    .parse()
                    .with_context(|| format!("--device: {entry:?} is not a device number"))?;
                if ordinal >= available {
                    bail!(
                        "--device: no device {ordinal}; the driver sees {available} (0..{})",
                        available.saturating_sub(1),
                    );
                }
                Ok(ordinal)
            })
            .collect::<Result<Vec<u32>>>()?
    };
    if devices.is_empty() {
        bail!("--device {spec:?} resolved to no devices; is a CUDA device visible?");
    }
    Ok(devices)
}

/// What the per-device search workers share.
struct Shared {
    /// Combined progress across every device, so the rate line a user reads
    /// is the machine's throughput rather than one card's.
    progress: Mutex<Progress>,
    /// Keys written so far, across all workers. This is what `--count`
    /// counts.
    written: AtomicUsize,
    /// Set when `--count` is met or a worker fails; workers check it at
    /// launch boundaries, so a stop takes effect within one launch interval
    /// and in-flight work is kept rather than discarded.
    stop: AtomicBool,
    /// One entry per failed worker, so every device's failure is reported
    /// rather than only the first.
    errors: Mutex<Vec<String>>,
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
    let devices = parse_devices(
        &args.device,
        honion_gpu::device_count().context("querying CUDA devices")?,
    )?;
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating output directory {}", args.out.display()))?;

    let difficulty = set.difficulty_log2();
    if !args.quiet {
        eprintln!(
            "honion: searching for {} pattern(s) on {} device(s)",
            sourced.len(),
            devices.len(),
        );
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

    let shared = Shared {
        progress: Mutex::new(Progress::new(difficulty, args.quiet)),
        written: AtomicUsize::new(0),
        stop: AtomicBool::new(false),
        errors: Mutex::new(Vec::new()),
    };

    // One worker per device. Each owns its Searcher and its epoch's scalars,
    // which are drawn, uploaded and destroyed on that thread — every device
    // sees only the public keys of its own worker's scalars, so the
    // no-secret-on-the-device invariant of docs/05 holds per card unchanged.
    std::thread::scope(|scope| {
        for &device in &devices {
            // Move a *reference* into each worker, not the shared state
            // itself: `&Shared` is `Copy`, so every worker gets its own
            // pointer to the one Shared.
            let shared = &shared;
            let set = &set;
            let tables = &tables;
            std::thread::Builder::new()
                .name(format!("honion-gpu{device}"))
                // NVRTC compiles the kernel and the driver JITs the PTX on
                // this thread now, not the main one. That recursion is deep
                // enough to overflow the 2 MiB default a spawned thread
                // gets; the main thread's 8 MiB never noticed.
                .stack_size(WORKER_STACK_BYTES)
                .spawn_scoped(scope, move || {
                    if let Err(e) = search_device(device, args, set, tables, shared) {
                        // Report the whole context chain, then take the
                        // other workers down at their next launch boundary
                        // rather than letting them search for a result that
                        // will not be wanted.
                        eprintln!("error: device {device}: {e:?}");
                        shared
                            .errors
                            .lock()
                            .expect("the error list was poisoned")
                            .push(format!("device {device}"));
                        shared.stop.store(true, Ordering::Relaxed);
                    }
                })
                .expect("spawning a search worker");
        }
    });

    let written = shared.written.load(Ordering::SeqCst);
    shared
        .progress
        .lock()
        .expect("the progress counter was poisoned")
        .report();

    let errors = shared.errors.lock().expect("the error list was poisoned");
    if !errors.is_empty() {
        bail!(
            "{} of {} device(s) failed ({} key(s) written)",
            errors.len(),
            devices.len(),
            written,
        );
    }
    if !args.quiet {
        let progress = shared
            .progress
            .lock()
            .expect("the progress counter was poisoned");
        eprintln!(
            "honion: wrote {written} key(s); the search reported {} candidate(s) \
             while examining {} addresses in {}",
            progress.found(),
            si(progress.examined() as f64),
            humantime(progress.elapsed().as_secs_f64()),
        );
    }
    Ok(())
}

/// Search on one device until the run is over.
///
/// The loop is the single-device search unchanged — same launch pacing, same
/// host preparation overlapped with device work — with three additions:
/// `--count` is decided by the shared counter rather than a local one, the
/// shared `stop` flag is checked at launch boundaries, and progress is
/// recorded into the combined counter. A run with `--device 0` is this loop
/// and nothing else.
fn search_device(
    device: u32,
    args: &SearchArgs,
    set: &PatternSet,
    tables: &DeviceTables,
    shared: &Shared,
) -> Result<()> {
    // Sized to the card unless the user pinned it: throughput rises with
    // concurrent walks until their local memory stops fitting, so the right
    // number is a property of the device, not a constant.
    let threads = match args.threads {
        Some(0) => bail!("--threads must be positive"),
        Some(n) => n,
        None => honion_gpu::auto_threads(device, args.offsets)
            .with_context(|| format!("sizing the search to device {device}"))?,
    };

    // The hit buffer is sized for a very unlucky launch. Overflow is reported
    // rather than silent, but it should never happen.
    let max_hits = 4096u32;
    let mut searcher = Searcher::new(device, tables, threads, args.offsets, max_hits)
        .with_context(|| format!("preparing the GPU search on device {device}"))?;
    if !args.quiet {
        let (major, minor) = searcher.compute_capability().unwrap_or((0, 0));
        let name = searcher.name().unwrap_or_default();
        eprintln!(
            "  device {device} ({name}) compute capability {major}.{minor}, {} threads, \
             {} offsets ({} candidates per inversion, {:.1} GB device memory)",
            searcher.num_threads(),
            searcher.half(),
            searcher.candidates_per_batch(),
            honion_gpu::local_bytes_per_thread(searcher.half()) as f64
                * f64::from(searcher.num_threads()) / 1e9
        );
    }

    // Start small so the first launch cannot run for an unexpectedly long time
    // on a slow device; the rate measured from it sizes every later launch.
    let mut candidates: u32 = 4096;

    // Draw the first epoch's scalars. They live only here, on this thread, in
    // host memory; the device receives their public keys and nothing else.
    let mut scalars = draw_scalars(threads)?;
    searcher
        .set_start_points(&public_keys(&scalars))
        .context("uploading starting points")?;

    loop {
        if shared.stop.load(Ordering::Relaxed) {
            return Ok(());
        }
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
                "warning: device {device} found {} candidates but only {} fit the buffer; \
                 {} were dropped. Raise --min-selectivity or use a longer pattern.",
                outcome.total_found,
                outcome.hits.len(),
                outcome.total_found as usize - outcome.hits.len()
            );
        }

        // Verify every candidate against the host's own arithmetic before it
        // is allowed to become a file. Each worker checks its own device's
        // claims against its own scalars; one device's hit can never be
        // reconstructed from another's starting point.
        for hit in &outcome.hits {
            let Some(start) = scalars.get(hit.thread_id as usize) else {
                bail!(
                    "device reported thread {} but only {} were launched",
                    hit.thread_id,
                    scalars.len()
                );
            };
            let verified =
                VerifiedKey::verify(start, i64::from(hit.offset), set).with_context(|| {
                    format!(
                        "verifying the candidate reported at thread {} offset {}. \
                         This means the GPU and the CPU disagree, which is a bug or a \
                         hardware fault, not a user error.",
                        hit.thread_id, hit.offset
                    )
                })?;

            let dir = write_service_dir(&args.out, &verified)
                .context("writing the hidden service directory")?;
            let total = shared.written.fetch_add(1, Ordering::SeqCst) + 1;
            println!("{}", verified.address().to_hostname());
            if !args.quiet {
                eprintln!("  written to {}", dir.display());
            }
            if args.count != 0 && total >= args.count {
                // Every key found up to and including this one is written —
                // a found key is never thrown away to honour a count. Two
                // devices finishing the same instant can therefore overshoot
                // `--count` by a handful, which is the only way the total
                // can exceed it.
                shared
                    .progress
                    .lock()
                    .expect("the progress counter was poisoned")
                    .record(outcome.examined, outcome.hits.len());
                shared.stop.store(true, Ordering::Relaxed);
                return Ok(());
            }
        }

        {
            let mut progress = shared
                .progress
                .lock()
                .expect("the progress counter was poisoned");
            progress.record(outcome.examined, outcome.hits.len());
            progress.report();
        }

        // Hand the prepared epoch to the device and keep its scalars, which
        // the next round's hits will be reconstructed from.
        searcher
            .set_start_points(&next_points)
            .context("uploading starting points")?;
        scalars = next_scalars;

        // Size the next launch from the rate just measured, so the loop
        // settles on the requested wall-clock interval regardless of how
        // many devices are competing for the host's CPU.
        if launch_time > 0.0 {
            let per_candidate = launch_time / f64::from(candidates);
            let want = (args.launch_seconds / per_candidate).round();
            candidates = want.clamp(1.0, f64::from(u32::MAX / 2)) as u32;
        }
    }
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

#[cfg(test)]
mod tests {
    use super::parse_devices;

    #[test]
    fn all_expands_to_every_visible_device() {
        assert_eq!(parse_devices("all", 3).unwrap(), vec![0, 1, 2]);
        assert_eq!(parse_devices("ALL", 1).unwrap(), vec![0]);
    }

    #[test]
    fn lists_keep_their_order_and_allow_repeats() {
        assert_eq!(parse_devices("0", 2).unwrap(), vec![0]);
        assert_eq!(parse_devices("2,0", 4).unwrap(), vec![2, 0]);
        assert_eq!(parse_devices(" 1 , 1 ", 2).unwrap(), vec![1, 1]);
    }

    #[test]
    fn malformed_specs_are_refused() {
        assert!(parse_devices("", 2).is_err());
        assert!(parse_devices("0,,1", 2).is_err());
        assert!(parse_devices("x", 2).is_err());
        // A machine with no devices cannot satisfy any request, however
        // well-formed.
        assert!(parse_devices("all", 0).is_err());
        assert!(parse_devices("0", 0).is_err());
    }

    #[test]
    fn ordinals_beyond_the_machine_are_named_in_the_error() {
        let err = parse_devices("3", 2).unwrap_err().to_string();
        assert!(err.contains("no device 3"), "unhelpful: {err}");
        assert!(err.contains("0..1"), "should state the valid range: {err}");
    }
}
