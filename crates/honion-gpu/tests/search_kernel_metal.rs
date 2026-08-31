//! End-to-end tests for the Metal search kernel.
//!
//! The Metal mirror of `tests/search_kernel.rs`. The field and group tests
//! established correct arithmetic; these establish the two remaining claims:
//!
//! * **It walks the keys it says it walks.** Iteration `k` on thread `t` must be
//!   the public key of `a0[t] + 8k` — the identity the host uses to reconstruct
//!   a secret from a hit.
//! * **It finds exactly the right keys.** The device's hit set over a search
//!   space is compared for *equality* against the host reference matcher over
//!   the same space: no missing hits, no spurious ones.
//!
//! `honion_gpu::Searcher` resolves to the Metal backend under the `metal`
//! feature, so these use the same public API the CUDA tests do.

// @decision DEC-METAL-006 (verified here)
// @title The Metal search kernel earns the CUDA set-equality bar, traps neutralised
// @status accepted
// @rationale This is the first kernel with a batch loop over the cold inversion,
//   so it is where the second compiler trap (full unrolling of the HALF loop,
//   guarded by `#pragma clang loop unroll(disable)`) becomes observable. That
//   the suite compiles the kernel at HALF up to 512 in seconds — rather than
//   hanging or emitting a huge library — is the verification that the pragma is
//   honoured. Correctness is the CUDA set-equality bar: exact agreement with an
//   independent host reference over the full covered space, incl. negative
//   offsets, residual patterns, multiple patterns, and every table size.

use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
use curve25519_dalek::edwards::EdwardsPoint;
use curve25519_dalek::scalar::Scalar;
use honion_core::address::OnionAddress;
use honion_core::pattern::{Pattern, PatternSet};
use honion_gpu::{DeviceTables, SearchError, Searcher};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Whether a Metal device is usable; tests no-op without one. Probed by trying
/// to build a searcher for a trivial pattern.
fn have_gpu() -> bool {
    let set = PatternSet::compile(&[Pattern::parse("a").expect("valid")]).expect("non-empty");
    let tables = DeviceTables::build(&set);
    match Searcher::new(&tables, 64, 8, 16) {
        Ok(_) => true,
        Err(SearchError::Driver(msg)) if msg.contains("no Metal device") => {
            eprintln!("skipping: no Metal device available");
            false
        }
        Err(e) => panic!("unexpected searcher error while probing: {e}"),
    }
}

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

#[test]
fn finds_a_planted_needle() {
    if !have_gpu() {
        return;
    }
    const THREADS: u32 = 256;
    const HALF: u32 = 64;
    const CANDS: u32 = 4000;
    const NEEDLE_THREAD: usize = 137;
    const NEEDLE_OFFSET: i64 = -37; // negative: below the starting scalar

    let (scalars, points) = starting_points(THREADS as usize, 22);

    let needle_scalar =
        scalars[NEEDLE_THREAD] - Scalar::from(8u64) * Scalar::from(NEEDLE_OFFSET.unsigned_abs());
    let needle_key = (ED25519_BASEPOINT_POINT * needle_scalar).compress().to_bytes();

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

/// Run a search and the host reference over the same space; require equality.
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
        "device and host reference disagree for pattern {pattern_src:?} (half {half}); \
         device found {} hits, host found {}",
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
    assert_device_agrees_with_host("ab", 256, 512, 32);
    // Coverage not a whole number of batches, so the round-up is exercised.
    assert_device_agrees_with_host("ab", 256, 500, 32);
}

/// Splitting a launch across dispatches must not change what it finds.
///
/// A launch is normally chunked to a wall-clock target (DEC-METAL-009), and on
/// a fast device that means one or two dispatches — so the resume path barely
/// runs unless it is forced. Here the same search runs twice over identical
/// starting points: once with the sizing left alone, once capped to one batch
/// per dispatch, which splits it the maximum number of ways. Every reported hit
/// must match, offsets included, because an offset is what the host reconstructs
/// a key from: a resume that lost track of where the walk was would still report
/// plausible-looking hits, just at the wrong absolute offsets.
#[test]
fn splitting_a_launch_across_dispatches_changes_nothing() {
    if !have_gpu() {
        return;
    }
    const THREADS: u32 = 256;
    const HALF: u32 = 32;
    const CANDS: u32 = 512;

    let (_scalars, points) = starting_points(THREADS as usize, 24);
    let pattern = Pattern::parse("ab").expect("valid pattern");
    let set = PatternSet::compile(&[pattern]).expect("non-empty");
    let tables = DeviceTables::build(&set);

    let run = |cap: Option<u32>| {
        let mut s = Searcher::new(&tables, THREADS, HALF, 1 << 16).expect("searcher");
        s.set_max_batches_per_dispatch(cap);
        s.set_start_points(&points).expect("points");
        let outcome = s.launch(CANDS).expect("launch");
        let mut hits: Vec<(u32, i32, u32)> = outcome
            .hits
            .iter()
            .map(|h| (h.thread_id, h.offset, h.pattern_id))
            .collect();
        hits.sort_unstable();
        (hits, outcome.total_found, outcome.examined)
    };

    // `CANDS / (2 * HALF + 1)` is 8 batches, so the capped run uses eight
    // dispatches where the free-running one uses one or two.
    let (whole, whole_found, whole_examined) = run(None);
    let (split, split_found, split_examined) = run(Some(1));

    assert!(
        !whole.is_empty(),
        "the test space contained no matches, so the comparison is vacuous"
    );
    assert_eq!(whole, split, "chunking changed which candidates were reported");
    assert_eq!(whole_found, split_found, "chunking changed the hit count");
    assert_eq!(
        whole_examined, split_examined,
        "chunking changed how many addresses the launch claims to have examined"
    );
}

/// A capped launch must stay correct across repeats, not just on a cold
/// searcher: the walk buffer is reused, so a launch that left it dirty would
/// show up as the second launch disagreeing with the first.
#[test]
fn a_chunked_searcher_repeats_a_launch_identically() {
    if !have_gpu() {
        return;
    }
    const THREADS: u32 = 256;
    const HALF: u32 = 32;
    const CANDS: u32 = 512;

    let (_scalars, points) = starting_points(THREADS as usize, 25);
    let pattern = Pattern::parse("ab").expect("valid pattern");
    let set = PatternSet::compile(&[pattern]).expect("non-empty");
    let tables = DeviceTables::build(&set);

    let mut s = Searcher::new(&tables, THREADS, HALF, 1 << 16).expect("searcher");
    s.set_max_batches_per_dispatch(Some(1));

    let mut seen = Vec::new();
    for round in 0..3 {
        s.set_start_points(&points).expect("points");
        let outcome = s.launch(CANDS).expect("launch");
        let mut hits: Vec<(u32, i32, u32)> = outcome
            .hits
            .iter()
            .map(|h| (h.thread_id, h.offset, h.pattern_id))
            .collect();
        hits.sort_unstable();
        if round == 0 {
            assert!(!hits.is_empty(), "no matches, so the comparison is vacuous");
            seen = hits;
        } else {
            assert_eq!(seen, hits, "round {round} disagreed with the first launch");
        }
    }
}

#[test]
fn agreement_holds_for_patterns_needing_residual_checks() {
    if !have_gpu() {
        return;
    }
    assert_device_agrees_with_host("[ab][cd]", 256, 512, 32);
    assert_device_agrees_with_host("a?c", 512, 512, 32);
}

#[test]
fn agreement_is_independent_of_table_size() {
    if !have_gpu() {
        return;
    }
    // HALF up to 512 is where the unroll-disable pragma is load-bearing: the
    // kernel must still compile in seconds. Small values exercise the
    // almost-all-base-point degenerate batch.
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
    for pid in 0..sources.len() as u32 {
        assert!(
            host.iter().any(|h| h.2 == pid),
            "pattern {} never matched; the test is not exercising what it claims",
            sources[pid as usize]
        );
    }
}
