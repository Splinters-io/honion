//! Measure compile time, JIT time, and search throughput.
use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
use curve25519_dalek::scalar::Scalar;
use honion_core::pattern::{Pattern, PatternSet};
use honion_gpu::{DeviceTables, Searcher};
use std::time::Instant;

fn main() {
    let half: u32 = std::env::var("HALF").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    let threads: u32 = std::env::var("THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1 << 18);
    let iters: u32 = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);

    let t0 = Instant::now();
    let ptx = honion_gpu::nvrtc::compile(
        honion_gpu::nvrtc::sources::SEARCH, (12, 0),
        &[("HALF", half.to_string()), ("FE_RADIX32", "1".to_owned())],
    ).expect("compile");
    println!("nvrtc compile      : {:>8.2?}  ({} KB PTX)", t0.elapsed(), ptx.len() / 1024);

    // A 12-character pattern: effectively unfindable, so the hot loop is never
    // disturbed by hits and the number measured is pure search throughput.
    let pat = std::env::var("PATTERN").unwrap_or_else(|_| "zzzzzzzzzzzz".to_owned());
    let set = PatternSet::compile(&[Pattern::parse(&pat).unwrap()]).unwrap();
    let tables = DeviceTables::build(&set);

    let t1 = Instant::now();
    let mut s = Searcher::new(&tables, threads, half, 4096).expect("searcher");
    println!("compile + JIT + up : {:>8.2?}", t1.elapsed());

    let points: Vec<[u8; 32]> = (0..threads)
        .map(|i| {
            let sc = Scalar::from(u64::from(i) * 2_654_435_761 + 12345);
            (ED25519_BASEPOINT_POINT * sc).compress().to_bytes()
        })
        .collect();
    s.set_start_points(&points).expect("points");

    // Warm up: the first launch pays for lazy module loading.
    s.launch(64).expect("warmup");

    // Repeat launches back-to-back, so a rate that only holds from an idle GPU
    // is visible as a decay rather than reported as the headline number.
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    for r in 0..reps {
        let t2 = Instant::now();
        let out = s.launch(iters).expect("launch");
        let dt = t2.elapsed();
        let rate = out.examined as f64 / dt.as_secs_f64();
        println!(
            "half {half:>4} threads {threads:>7} cands {iters:>6} rep {:>2} : {:>7.2?}  {:.3} G addr/s  ({} hits)",
            r + 1, dt, rate / 1e9, out.total_found
        );
    }
}
