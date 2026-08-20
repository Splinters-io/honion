//! Differential tests for the device group arithmetic.
//!
//! The reference here is `curve25519-dalek`, a widely reviewed independent
//! implementation of the same curve. The device is asked to perform the exact
//! operation the search kernel relies on — repeated addition of 8*B — and the
//! host checks the answer by computing the corresponding scalar multiple
//! directly.
//!
//! Two properties matter and are tested separately:
//!
//! * **Agreement.** Walking k steps from a*B must land on (a + 8k)*B. If this
//!   holds, the host can reconstruct a secret from a step count, which is the
//!   entire basis of the design.
//! * **Staying on the curve.** A point that drifts off the curve after many
//!   additions would indicate a field-arithmetic bug that short tests miss.
//!   The walk is therefore run long enough to accumulate any such drift.

use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
use curve25519_dalek::edwards::CompressedEdwardsY;
use curve25519_dalek::scalar::Scalar;
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

struct Harness {
    ctx: Arc<CudaContext>,
    module: Arc<cudarc::driver::CudaModule>,
}

impl Harness {
    fn new() -> Option<Self> {
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no CUDA device available ({e:?})");
                return None;
            }
        };
        let (major, minor) = ctx.compute_capability().expect("compute capability");
        let ptx = honion_gpu::nvrtc::compile_cached(
            honion_gpu::nvrtc::sources::TESTKERNELS,
            (major as u32, minor as u32),
            &[],
        )
        .expect("device sources must compile");
        let module = ctx.load_module(ptx.into()).expect("module loads");
        Some(Self { ctx, module })
    }

    fn cfg(n: usize) -> LaunchConfig {
        let n = n as u32;
        LaunchConfig {
            grid_dim: (n.div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// Random scalars, plus the small ones where an implementation is most likely
/// to have a special-case bug.
fn test_scalars(count: usize, seed: u64) -> Vec<Scalar> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<Scalar> = vec![
        Scalar::from(1u64),
        Scalar::from(2u64),
        Scalar::from(8u64),
        Scalar::from(u64::MAX),
    ];
    while v.len() < count {
        let mut b = [0u8; 32];
        rng.fill(&mut b);
        v.push(Scalar::from_bytes_mod_order(b));
    }
    v
}

const CASES: usize = 1024;

#[test]
fn eight_times_base_point_matches_the_generated_constant() {
    // `cuda/gen_constants.py` prints the compressed form of 8*B alongside the
    // limbs it emits. If the generator were wrong, every limb below would be
    // wrong together and self-consistently — so the value is pinned here
    // against an implementation that had no part in producing it.
    let expected = (ED25519_BASEPOINT_POINT * Scalar::from(8u64)).compress();
    assert_eq!(
        hex(expected.as_bytes()),
        "b4b937fca95b2f1e93e41e62fc3c78818ff38a66096fad6e7973e5c90006d321",
        "the constant recorded in ge25519.cuh no longer matches 8*B"
    );
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn decompression_round_trips_public_keys() {
    let Some(h) = Harness::new() else { return };
    let scalars = test_scalars(CASES, 11);
    let points: Vec<[u8; 32]> = scalars
        .iter()
        .map(|s| (ED25519_BASEPOINT_POINT * s).compress().to_bytes())
        .collect();

    let stream = h.ctx.default_stream();
    let func = h.module.load_function("test_ge_roundtrip").expect("kernel");
    let flat: Vec<u8> = points.iter().flatten().copied().collect();
    let d_in: CudaSlice<u8> = stream.clone_htod(&flat).expect("upload");
    let mut d_out: CudaSlice<u8> = stream.alloc_zeros(points.len() * 32).expect("alloc");
    let mut d_ok: CudaSlice<u8> = stream.alloc_zeros(points.len()).expect("alloc");
    let n = points.len() as u32;
    let mut b = stream.launch_builder(&func);
    b.arg(&d_in).arg(&mut d_out).arg(&mut d_ok).arg(&n);
    // Safety: signature and buffer sizes match; the kernel bounds-checks `n`.
    unsafe { b.launch(Harness::cfg(points.len())) }.expect("launch");
    stream.synchronize().expect("sync");

    let out = stream.clone_dtoh(&d_out).expect("download");
    let ok = stream.clone_dtoh(&d_ok).expect("download");
    for (i, p) in points.iter().enumerate() {
        assert_eq!(ok[i], 1, "case {i}: a real public key failed to decompress");
        assert_eq!(&out[32 * i..32 * i + 32], &p[..], "case {i}: round trip changed the key");
    }
}

#[test]
fn decompression_rejects_non_points() {
    // Roughly half of all 32-byte strings do not encode a curve point. The
    // kernel must say so rather than silently producing a value.
    let Some(h) = Harness::new() else { return };
    let mut rng = StdRng::seed_from_u64(12);
    let candidates: Vec<[u8; 32]> = (0..2048)
        .map(|_| {
            let mut b = [0u8; 32];
            rng.fill(&mut b);
            b
        })
        .collect();

    let stream = h.ctx.default_stream();
    let func = h.module.load_function("test_ge_roundtrip").expect("kernel");
    let flat: Vec<u8> = candidates.iter().flatten().copied().collect();
    let d_in: CudaSlice<u8> = stream.clone_htod(&flat).expect("upload");
    let mut d_out: CudaSlice<u8> = stream.alloc_zeros(candidates.len() * 32).expect("alloc");
    let mut d_ok: CudaSlice<u8> = stream.alloc_zeros(candidates.len()).expect("alloc");
    let n = candidates.len() as u32;
    let mut b = stream.launch_builder(&func);
    b.arg(&d_in).arg(&mut d_out).arg(&mut d_ok).arg(&n);
    // Safety: as above.
    unsafe { b.launch(Harness::cfg(candidates.len())) }.expect("launch");
    stream.synchronize().expect("sync");
    let ok = stream.clone_dtoh(&d_ok).expect("download");

    let mut agreed = 0usize;
    let mut accepted = 0usize;
    for (i, c) in candidates.iter().enumerate() {
        let dalek_ok = CompressedEdwardsY(*c).decompress().is_some();
        assert_eq!(
            ok[i] == 1,
            dalek_ok,
            "case {i}: device and dalek disagree on whether {} is a point",
            hex(c)
        );
        agreed += 1;
        accepted += usize::from(dalek_ok);
    }
    assert_eq!(agreed, candidates.len());
    // Sanity: the sample really does contain both kinds, so the agreement above
    // is not vacuous.
    assert!(accepted > 500 && accepted < 1500, "unexpected split: {accepted}/2048");
}

/// Walk `steps` additions of 8*B from each of `scalars`, and check the device
/// lands where the scalar arithmetic says it should.
fn check_walk(h: &Harness, scalars: &[Scalar], steps: u32) {
    let points: Vec<[u8; 32]> = scalars
        .iter()
        .map(|s| (ED25519_BASEPOINT_POINT * s).compress().to_bytes())
        .collect();

    let stream = h.ctx.default_stream();
    let func = h.module.load_function("test_ge_walk").expect("kernel");
    let flat: Vec<u8> = points.iter().flatten().copied().collect();
    let d_in: CudaSlice<u8> = stream.clone_htod(&flat).expect("upload");
    let mut d_out: CudaSlice<u8> = stream.alloc_zeros(points.len() * 32).expect("alloc");
    let n = points.len() as u32;
    let mut b = stream.launch_builder(&func);
    b.arg(&d_in).arg(&mut d_out).arg(&steps).arg(&n);
    // Safety: as above.
    unsafe { b.launch(Harness::cfg(points.len())) }.expect("launch");
    stream.synchronize().expect("sync");
    let out = stream.clone_dtoh(&d_out).expect("download");

    let delta = Scalar::from(8u64) * Scalar::from(u64::from(steps));
    for (i, s) in scalars.iter().enumerate() {
        let expected = (ED25519_BASEPOINT_POINT * (s + delta)).compress().to_bytes();
        assert_eq!(
            &out[32 * i..32 * i + 32],
            &expected[..],
            "case {i}: after {steps} additions of 8B from scalar {}, device landed on {} but (a + 8k)B is {}",
            hex(s.as_bytes()),
            hex(&out[32 * i..32 * i + 32]),
            hex(&expected)
        );
    }
}

#[test]
fn a_single_addition_of_eight_b_is_correct() {
    let Some(h) = Harness::new() else { return };
    check_walk(&h, &test_scalars(CASES, 13), 1);
}

#[test]
fn a_long_walk_stays_exact() {
    // 512 additions per thread. Any error in the addition formula or the field
    // arithmetic compounds and would show up as a mismatch here even if a
    // single step happened to look right.
    let Some(h) = Harness::new() else { return };
    check_walk(&h, &test_scalars(256, 14), 512);
}

#[test]
fn points_stay_on_the_curve_throughout_a_walk() {
    let Some(h) = Harness::new() else { return };
    let scalars = test_scalars(256, 15);
    let points: Vec<[u8; 32]> = scalars
        .iter()
        .map(|s| (ED25519_BASEPOINT_POINT * s).compress().to_bytes())
        .collect();

    let stream = h.ctx.default_stream();
    let func = h.module.load_function("test_ge_on_curve").expect("kernel");
    let flat: Vec<u8> = points.iter().flatten().copied().collect();
    let d_in: CudaSlice<u8> = stream.clone_htod(&flat).expect("upload");
    let mut d_out: CudaSlice<u8> = stream.alloc_zeros(points.len()).expect("alloc");
    let steps: u32 = 1024;
    let n = points.len() as u32;
    let mut b = stream.launch_builder(&func);
    b.arg(&d_in).arg(&mut d_out).arg(&steps).arg(&n);
    // Safety: as above.
    unsafe { b.launch(Harness::cfg(points.len())) }.expect("launch");
    stream.synchronize().expect("sync");
    let out = stream.clone_dtoh(&d_out).expect("download");
    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, 1, "case {i}: point left the curve after {steps} additions");
    }
}
