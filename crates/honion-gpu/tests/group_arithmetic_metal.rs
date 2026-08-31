//! Differential tests for the Metal group arithmetic.
//!
//! The Metal mirror of `tests/group_arithmetic.rs`. The reference is
//! `curve25519-dalek`, an independent implementation of the same curve. The
//! device performs the exact operation the search kernel relies on — repeated
//! addition of 8*B — and the host checks it by computing the corresponding
//! scalar multiple directly. Skipped cleanly when no Metal device is present.

// @decision DEC-METAL-004 (curve, test side)
// @title The Metal curve arithmetic earns the same differential bar as CUDA
// @status accepted
// @rationale A line-for-line analog of the CUDA group test: the same
//   curve25519-dalek reference, the same pinned 8*B constant check, the same
//   decompress round-trip / non-point rejection / (a+8k)B walk / stays-on-curve
//   checks, including a 512-step walk where any per-step error compounds. This
//   holds the dual-law-adjacent madd walk and (de)compression to the plan's bar
//   (REQ-GOAL-002), and exercises the field port through the curve.

use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
use curve25519_dalek::edwards::CompressedEdwardsY;
use curve25519_dalek::scalar::Scalar;
use honion_gpu::msl::{self, MslError, MslKernel, MslLibrary, SharedBuffer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

struct Harness {
    lib: MslLibrary,
}

impl Harness {
    fn new() -> Option<Self> {
        match MslLibrary::compile(msl::sources::TESTKERNELS, &[]) {
            Ok(lib) => Some(Self { lib }),
            Err(MslError::NoDevice) => {
                eprintln!("skipping: no Metal device available");
                None
            }
            Err(e) => panic!("group test kernels must compile: {e}"),
        }
    }

    fn kernel(&self, name: &str) -> MslKernel {
        self.lib.kernel(name).expect("kernel present")
    }

    fn upload(&self, k: &MslKernel, records: &[[u8; 32]]) -> SharedBuffer {
        let mut buf = k.new_shared_buffer(records.len() * 32).expect("alloc");
        let flat: Vec<u8> = records.iter().flatten().copied().collect();
        buf.as_mut_slice::<u8>().copy_from_slice(&flat);
        buf
    }

    fn scalar_buf(&self, k: &MslKernel, value: u32) -> SharedBuffer {
        let mut buf = k
            .new_shared_buffer(std::mem::size_of::<u32>())
            .expect("alloc");
        buf.as_mut_slice::<u32>()[0] = value;
        buf
    }

    fn tg(k: &MslKernel) -> usize {
        k.max_threads_per_threadgroup().min(256)
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Random scalars, plus the small ones most likely to hit a special case.
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
    // The constants baked into ge25519.metal describe 8*B. Pin their compressed
    // form against dalek, which had no part in producing them.
    let expected = (ED25519_BASEPOINT_POINT * Scalar::from(8u64)).compress();
    assert_eq!(
        hex(expected.as_bytes()),
        "b4b937fca95b2f1e93e41e62fc3c78818ff38a66096fad6e7973e5c90006d321",
        "the constants in ge25519.metal no longer describe 8*B"
    );
}

/// Decompress→recompress via test_ge_roundtrip; returns (out, ok).
fn roundtrip(h: &Harness, records: &[[u8; 32]]) -> (Vec<[u8; 32]>, Vec<u8>) {
    let k = h.kernel("test_ge_roundtrip");
    let n = records.len();
    let da = h.upload(&k, records);
    let out = k.new_shared_buffer(n * 32).expect("alloc");
    let ok = k.new_shared_buffer(n).expect("alloc");
    let dn = h.scalar_buf(&k, n as u32);
    k.dispatch(n, Harness::tg(&k), &[&da, &out, &ok, &dn]).expect("dispatch");
    let obytes = out.as_slice::<u8>();
    let out_v = (0..n)
        .map(|i| {
            let mut r = [0u8; 32];
            r.copy_from_slice(&obytes[i * 32..i * 32 + 32]);
            r
        })
        .collect();
    (out_v, ok.as_slice::<u8>().to_vec())
}

#[test]
fn decompression_round_trips_public_keys() {
    let Some(h) = Harness::new() else { return };
    let scalars = test_scalars(CASES, 11);
    let points: Vec<[u8; 32]> = scalars
        .iter()
        .map(|s| (ED25519_BASEPOINT_POINT * s).compress().to_bytes())
        .collect();
    let (out, ok) = roundtrip(&h, &points);
    for (i, p) in points.iter().enumerate() {
        assert_eq!(ok[i], 1, "case {i}: a real public key failed to decompress");
        assert_eq!(&out[i], p, "case {i}: round trip changed the key");
    }
}

#[test]
fn decompression_rejects_non_points() {
    let Some(h) = Harness::new() else { return };
    let mut rng = StdRng::seed_from_u64(12);
    let candidates: Vec<[u8; 32]> = (0..2048)
        .map(|_| {
            let mut b = [0u8; 32];
            rng.fill(&mut b);
            b
        })
        .collect();
    let (_out, ok) = roundtrip(&h, &candidates);

    let mut accepted = 0usize;
    for (i, c) in candidates.iter().enumerate() {
        let dalek_ok = CompressedEdwardsY(*c).decompress().is_some();
        assert_eq!(
            ok[i] == 1,
            dalek_ok,
            "case {i}: device and dalek disagree on whether {} is a point",
            hex(c)
        );
        accepted += usize::from(dalek_ok);
    }
    // The sample really contains both kinds, so the agreement is not vacuous.
    assert!(accepted > 500 && accepted < 1500, "unexpected split: {accepted}/2048");
}

/// Walk `steps` additions of 8*B and check the device lands on (a + 8k)*B.
fn check_walk(h: &Harness, scalars: &[Scalar], steps: u32) {
    let points: Vec<[u8; 32]> = scalars
        .iter()
        .map(|s| (ED25519_BASEPOINT_POINT * s).compress().to_bytes())
        .collect();
    let k = h.kernel("test_ge_walk");
    let n = points.len();
    let da = h.upload(&k, &points);
    let out = k.new_shared_buffer(n * 32).expect("alloc");
    let dsteps = h.scalar_buf(&k, steps);
    let dn = h.scalar_buf(&k, n as u32);
    k.dispatch(n, Harness::tg(&k), &[&da, &out, &dsteps, &dn]).expect("dispatch");
    let obytes = out.as_slice::<u8>();

    let delta = Scalar::from(8u64) * Scalar::from(u64::from(steps));
    for (i, s) in scalars.iter().enumerate() {
        let expected = (ED25519_BASEPOINT_POINT * (s + delta)).compress().to_bytes();
        assert_eq!(
            &obytes[i * 32..i * 32 + 32],
            &expected[..],
            "case {i}: after {steps} additions of 8B from scalar {}, device landed on {} but (a + 8k)B is {}",
            hex(s.as_bytes()),
            hex(&obytes[i * 32..i * 32 + 32]),
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
    // 512 additions per thread. Any error in the madd formula or the field
    // arithmetic compounds and shows here even if one step looked right.
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
    let k = h.kernel("test_ge_on_curve");
    let n = points.len();
    let da = h.upload(&k, &points);
    let out = k.new_shared_buffer(n).expect("alloc");
    let steps: u32 = 1024;
    let dsteps = h.scalar_buf(&k, steps);
    let dn = h.scalar_buf(&k, n as u32);
    k.dispatch(n, Harness::tg(&k), &[&da, &out, &dsteps, &dn]).expect("dispatch");
    let res = out.as_slice::<u8>();
    for (i, v) in res.iter().enumerate() {
        assert_eq!(*v, 1, "case {i}: point left the curve after {steps} additions");
    }
}
