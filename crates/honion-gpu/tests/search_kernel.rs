//! End-to-end tests for the search kernel.
//!
//! The field and group tests establish that the device does correct arithmetic.
//! These establish the two remaining claims:
//!
//! * **It walks the keys it says it walks.** Iteration `k` on thread `t` must
//!   be the public key of `a0[t] + 8k`, because that identity is how the host
//!   reconstructs a secret from a hit. `walk_matches_scalar_arithmetic` checks
//!   the entire visited sequence, not just its endpoints.
//!
//! * **It finds exactly the right keys — no more, no fewer.** The device's set
//!   of hits over a search space is compared for equality against the host
//!   reference matcher run over the same space. A missing hit means wasted GPU
//!   time; a spurious one means a bug the verifier would have to catch. Neither
//!   is acceptable, so the test demands exact agreement rather than containment.

use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
use curve25519_dalek::edwards::EdwardsPoint;
use curve25519_dalek::scalar::Scalar;
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use honion_core::address::OnionAddress;
use honion_core::pattern::{Pattern, PatternSet};
use honion_gpu::{DeviceTables, Searcher};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Whether a CUDA device is usable; tests no-op without one.
fn have_gpu() -> bool {
    match CudaContext::new(0) {
        Ok(_) => true,
        Err(e) => {
            eprintln!("skipping: no CUDA device available ({e:?})");
            false
        }
    }
}

/// Random starting scalars and their public points.
fn starting_points(count: usize, seed: u64) -> (Vec<Scalar>, Vec<[u8; 32]>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let scalars: Vec<Scalar> = (0..count)
        .map(|_| {
            let mut b = [0u8; 32];
            rng.fill(&mut b);
            Scalar::from_bytes_mod_order(b)
        })
        .collect();
    let points = scalars
        .iter()
        .map(|s| (ED25519_BASEPOINT_POINT * s).compress().to_bytes())
        .collect();
    (scalars, points)
}

/// Public keys of `start + 8m` for `m` in `first ..< first + count`.
///
/// The kernel covers a contiguous run of offsets *centred* on each thread's
/// starting scalar, because the dual addition law yields `base + off` and
/// `base - off` together. A batch spans `[-half, +half]`, and successive
/// batches step by `2*half + 1`, so the ranges tile exactly: overall coverage
/// is the contiguous range `[-half, batches*(2*half+1) - half - 1]`.
fn host_offsets(start: &Scalar, first: i64, count: u32) -> Vec<(i64, [u8; 32])> {
    let eight = Scalar::from(8u64);
    let eight_b = ED25519_BASEPOINT_POINT * eight;
    let first_scalar = if first >= 0 {
        start + eight * Scalar::from(first.unsigned_abs())
    } else {
        start - eight * Scalar::from(first.unsigned_abs())
    };
    let mut p: EdwardsPoint = ED25519_BASEPOINT_POINT * first_scalar;
    let mut out = Vec::with_capacity(count as usize);
    for i in 0..count {
        out.push((first + i64::from(i), p.compress().to_bytes()));
        p += eight_b;
    }
    out
}

/// Offsets the kernel covers for a given configuration.
fn coverage(half: u32, candidates: u32) -> (i64, u32) {
    let per_batch = honion_gpu::candidates_per_batch(half);
    let batches = candidates.div_ceil(per_batch).max(1);
    (-i64::from(half), batches * per_batch)
}

/// Every key thread `t` visits, for the simple `+8B` walk the dump kernel does.
fn host_walk(start: &Scalar, iterations: u32) -> Vec<[u8; 32]> {
    host_offsets(start, 0, iterations)
        .into_iter()
        .map(|(_, k)| k)
        .collect()
}

#[test]
fn walk_matches_scalar_arithmetic() {
    if !have_gpu() {
        return;
    }
    const THREADS: usize = 64;
    const ITERS: u32 = 128;

    let (scalars, points) = starting_points(THREADS, 21);

    let ctx = CudaContext::new(0).expect("context");
    let (major, minor) = ctx.compute_capability().expect("capability");
    let ptx = honion_gpu::nvrtc::compile_cached(
        honion_gpu::nvrtc::sources::SEARCH,
        (major as u32, minor as u32),
        &[],
    )
    .expect("compiles");
    let module = ctx.load_module(ptx.into()).expect("module");
    let func = module.load_function("honion_walk_dump").expect("kernel");

    let stream = ctx.default_stream();
    let flat: Vec<u8> = points.iter().flatten().copied().collect();
    let d_in: CudaSlice<u8> = stream.clone_htod(&flat).expect("upload");
    let mut d_out: CudaSlice<u8> = stream
        .alloc_zeros(THREADS * ITERS as usize * 32)
        .expect("alloc");
    let n = THREADS as u32;
    let mut b = stream.launch_builder(&func);
    b.arg(&d_in).arg(&n).arg(&ITERS).arg(&mut d_out);
    // Safety: signature and buffer sizes match; the kernel bounds-checks `n`.
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (n.div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .expect("launch");
    stream.synchronize().expect("sync");
    let out = stream.clone_dtoh(&d_out).expect("download");

    for (t, s) in scalars.iter().enumerate() {
        let expected = host_walk(s, ITERS);
        for (k, want) in expected.iter().enumerate() {
            let base = (t * ITERS as usize + k) * 32;
            assert_eq!(
                &out[base..base + 32],
                &want[..],
                "thread {t} iteration {k}: device visited a different key than a0 + 8*{k}"
            );
        }
    }
}

#[test]
fn finds_a_planted_needle() {
    // Take a key the search will pass over, use its own address prefix as the
    // pattern, and require the device to report it at exactly the right place.
    // This is the whole tool in miniature: if the reported (thread, iteration)
    // were off by one, the host would reconstruct the wrong secret and the key
    // it wrote would not match its address.
    if !have_gpu() {
        return;
    }
    const THREADS: u32 = 256;
    const HALF: u32 = 64;
    const CANDS: u32 = 4000;
    const NEEDLE_THREAD: usize = 137;
    // Deliberately negative: a match below the starting scalar is exactly what
    // the old, one-directional enumeration could not produce, and getting its
    // sign wrong would reconstruct a different key.
    const NEEDLE_OFFSET: i64 = -37;

    let (scalars, points) = starting_points(THREADS as usize, 22);

    let needle_scalar =
        scalars[NEEDLE_THREAD] - Scalar::from(8u64) * Scalar::from(NEEDLE_OFFSET.unsigned_abs());
    let needle_key = (ED25519_BASEPOINT_POINT * needle_scalar).compress().to_bytes();

    // A ten-character prefix of the needle's own address. Long enough that a
    // second, coincidental match in 102 400 keys is essentially impossible.
    let address = OnionAddress::from_pubkey(&needle_key);
    let prefix: String = address.body().chars().take(10).collect();
    let pattern = Pattern::parse(&prefix).expect("an address prefix is valid base32");
    let set = PatternSet::compile(&[pattern]).expect("non-empty");
    let tables = DeviceTables::build(&set);

    let mut searcher = Searcher::new(&tables, THREADS, HALF, 1024).expect("searcher");
    searcher.set_start_points(&points).expect("points");
    let outcome = searcher.launch(CANDS).expect("launch");

    assert_eq!(
        outcome.total_found, 1,
        "expected exactly the planted needle, got {} hits",
        outcome.total_found
    );
    let hit = outcome.hits[0];
    assert_eq!(hit.thread_id as usize, NEEDLE_THREAD, "wrong thread reported");
    assert_eq!(i64::from(hit.offset), NEEDLE_OFFSET, "wrong offset reported");

    // Reconstruct the secret exactly as the real verifier will, and confirm it
    // reproduces the address we searched for.
    let m = i64::from(hit.offset);
    let recovered = if m >= 0 {
        scalars[hit.thread_id as usize] + Scalar::from(8u64) * Scalar::from(m.unsigned_abs())
    } else {
        scalars[hit.thread_id as usize] - Scalar::from(8u64) * Scalar::from(m.unsigned_abs())
    };
    let recovered_key = (ED25519_BASEPOINT_POINT * recovered).compress().to_bytes();
    assert_eq!(recovered_key, needle_key);
    assert!(
        OnionAddress::from_pubkey(&recovered_key).body().starts_with(&prefix),
        "reconstructed key does not have the prefix we searched for"
    );
}

/// Run a search and, independently, the host reference over the same space;
/// require the two hit sets to be equal.
fn assert_device_agrees_with_host(pattern_src: &str, threads: u32, candidates: u32, half: u32) {
    let (scalars, points) = starting_points(threads as usize, 23);
    let pattern = Pattern::parse(pattern_src).expect("valid pattern");
    let set = PatternSet::compile(&[pattern]).expect("non-empty");
    let tables = DeviceTables::build(&set);

    let mut searcher = Searcher::new(&tables, threads, half, 1 << 16).expect("searcher");
    searcher.set_start_points(&points).expect("points");
    let outcome = searcher.launch(candidates).expect("launch");
    assert_eq!(
        outcome.total_found as usize,
        outcome.hits.len(),
        "hit buffer overflowed; raise max_hits in the test"
    );

    let mut device: Vec<(u32, i64)> = outcome
        .hits
        .iter()
        .map(|h| (h.thread_id, i64::from(h.offset)))
        .collect();
    device.sort_unstable();

    let (first, count) = coverage(half, candidates);
    assert_eq!(
        outcome.examined,
        u64::from(threads) * u64::from(count),
        "examined count must match the offsets actually covered"
    );

    let mut host: Vec<(u32, i64)> = Vec::new();
    for (t, s) in scalars.iter().enumerate() {
        for (m, key) in host_offsets(s, first, count) {
            if !set.matching_patterns(&key).is_empty() {
                host.push((t as u32, m));
            }
        }
    }
    host.sort_unstable();

    assert_eq!(
        device, host,
        "device and host reference disagree for pattern {pattern_src:?} \
         (half {half}); device found {} hits, host found {}",
        device.len(),
        host.len()
    );
    assert!(
        !host.is_empty(),
        "the test space contained no matches, so agreement is vacuous"
    );
}

#[test]
fn device_hits_exactly_match_the_host_reference() {
    if !have_gpu() {
        return;
    }
    // A two-character prefix: ten bits, so ~128 hits in 131 072 keys — enough
    // that both a missing and a spurious hit would show.
    assert_device_agrees_with_host("ab", 256, 512, 32);
    // And a case where coverage is not a whole number of batches, so the
    // rounding up to a full batch is exercised.
    assert_device_agrees_with_host("ab", 256, 500, 32);
}

#[test]
fn agreement_holds_for_patterns_needing_residual_checks() {
    if !have_gpu() {
        return;
    }
    // A character class cannot be expressed by the prefilter, so these exercise
    // the residual path on the device.
    assert_device_agrees_with_host("[ab][cd]", 256, 512, 32);
    // A wildcard between literals: positions must stay aligned.
    assert_device_agrees_with_host("a?c", 512, 512, 32);
}

#[test]
fn agreement_is_independent_of_table_size() {
    if !have_gpu() {
        return;
    }
    // The table size is purely an optimisation; changing it must not change
    // which keys are found — only how they are enumerated. Small values also
    // exercise the degenerate case where a batch is almost all base point.
    // 256 and above put the shared-memory offset table past 30 KB, and the
    // per-thread fraction arrays past 40 KB of local memory; both are worth
    // exercising because a silent shared-memory overflow would corrupt results
    // rather than fail the launch.
    for half in [1u32, 2, 8, 16, 64, 128, 256, 512] {
        assert_device_agrees_with_host("ab", 64, 3000, half);
    }
}

#[test]
fn multiple_patterns_are_all_reported() {
    if !have_gpu() {
        return;
    }
    let sources = ["ab", "cd", "ef"];
    let patterns: Vec<Pattern> = sources
        .iter()
        .map(|s| Pattern::parse(s).expect("valid"))
        .collect();
    let set = PatternSet::compile(&patterns).expect("non-empty");
    let tables = DeviceTables::build(&set);

    let threads = 256u32;
    let half = 32u32;
    let candidates = 512u32;
    let (scalars, points) = starting_points(threads as usize, 24);
    let mut searcher = Searcher::new(&tables, threads, half, 1 << 16).expect("searcher");
    searcher.set_start_points(&points).expect("points");
    let outcome = searcher.launch(candidates).expect("launch");

    let mut device: Vec<(u32, i64, u32)> = outcome
        .hits
        .iter()
        .map(|h| (h.thread_id, i64::from(h.offset), h.pattern_id))
        .collect();
    device.sort_unstable();

    let (first, count) = coverage(half, candidates);
    let mut host: Vec<(u32, i64, u32)> = Vec::new();
    for (t, s) in scalars.iter().enumerate() {
        for (m, key) in host_offsets(s, first, count) {
            for pid in set.matching_patterns(&key) {
                host.push((t as u32, m, pid));
            }
        }
    }
    host.sort_unstable();

    assert_eq!(device, host, "device missed or invented a pattern match");
    // Every pattern should have been hit at least once in this much space.
    for pid in 0..sources.len() as u32 {
        assert!(
            host.iter().any(|h| h.2 == pid),
            "pattern {} never matched; the test is not exercising what it claims",
            sources[pid as usize]
        );
    }
}
