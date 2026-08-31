//! Measure Metal compile time, pipeline build time, and search throughput.
//!
//! The Metal counterpart of `examples/bench.rs`. It reports real numbers on the
//! Apple GPU present, following this project's measured-not-asserted discipline:
//! nothing here asserts a rate, it measures one. Back-to-back launches expose a
//! rate that only holds from an idle GPU as a decay rather than a headline.
//!
//! Run with `cargo run --release -p honion-gpu --features metal --example
//! bench_metal`. Tunable by env: HALF, THREADS, ITERS, REPS, PATTERN.

// @decision DEC-METAL-BENCH-001
// @title A separate Metal bench example rather than cfg-ing examples/bench.rs
// @status accepted
// @rationale examples/bench.rs is cuda-only (required-features cuda): it times
//   the NVRTC compile and reads PTX byte size, neither of which has a Metal
//   analog. Rather than thread cfg branches through it, the Metal bench is its
//   own file measuring the Metal-specific steps (MSL compile, pipeline build).
//   Both follow the same measured-not-asserted shape — warm up, then time
//   back-to-back launches — so their numbers are comparable in kind even though
//   the setup they measure differs. The figures this prints are what feed the
//   performance docs; no rate enters docs until it has been measured here.

use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
use curve25519_dalek::scalar::Scalar;
use honion_core::pattern::{Pattern, PatternSet};
use honion_gpu::msl::MslLibrary;
use honion_gpu::{DeviceTables, Searcher};
use std::time::Instant;

fn main() {
    let half: u32 = std::env::var("HALF").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let threads: u32 = std::env::var("THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1 << 18);
    let iters: u32 = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(4096);

    // Isolate the MSL compile step, which the Searcher otherwise folds into
    // construction, so its cost is visible on its own.
    let t0 = Instant::now();
    let lib = MslLibrary::compile(
        honion_gpu::msl::sources::SEARCH,
        &[("HALF", half.to_string())],
    );
    match lib {
        Ok(_) => println!("msl compile        : {:>8.2?}", t0.elapsed()),
        Err(e) => {
            eprintln!("no Metal device, or the kernel failed to compile: {e}");
            return;
        }
    }

    // A 12-character pattern is effectively unfindable, so the hot loop is never
    // disturbed by hits and the number measured is pure search throughput.
    let pat = std::env::var("PATTERN").unwrap_or_else(|_| "zzzzzzzzzzzz".to_owned());
    let set = PatternSet::compile(&[Pattern::parse(&pat).unwrap()]).unwrap();
    let tables = DeviceTables::build(&set);

    let t1 = Instant::now();
    let mut s = match Searcher::new(&tables, threads, half, 4096) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("searcher construction failed: {e}");
            return;
        }
    };
    println!("compile + build + up : {:>8.2?}", t1.elapsed());

    let points: Vec<[u8; 32]> = (0..threads)
        .map(|i| {
            let sc = Scalar::from(u64::from(i) * 2_654_435_761 + 12345);
            (ED25519_BASEPOINT_POINT * sc).compress().to_bytes()
        })
        .collect();
    s.set_start_points(&points).expect("points");

    // Warm up: the first launch pays for lazy pipeline warm-up.
    s.launch(64).expect("warmup");

    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    for r in 0..reps {
        let t2 = Instant::now();
        let out = s.launch(iters).expect("launch");
        let dt = t2.elapsed();
        let rate = out.examined as f64 / dt.as_secs_f64();
        println!(
            "half {half:>4} threads {threads:>7} cands {iters:>6} rep {:>2} : {:>7.2?}  {:.3} G addr/s  ({} hits)",
            r + 1,
            dt,
            rate / 1e9,
            out.total_found
        );
    }
}
