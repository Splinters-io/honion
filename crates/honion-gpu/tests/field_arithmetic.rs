//! Differential tests for the device field arithmetic.
//!
//! Every routine in `cuda/fe25519.cuh` is run on the GPU and compared against
//! an independent big-integer implementation on the host. The reference here
//! shares no code, no representation and no algorithm with the device: it is
//! plain `num-bigint` modular arithmetic on `2^255 - 19`. Agreement between two
//! implementations that different is meaningful evidence; agreement between an
//! implementation and itself is not.
//!
//! This is a gate, not a formality. The search kernel is built on these
//! primitives and cannot be trusted while they are unverified — a wrong
//! coefficient would produce points off the curve, and the search would run at
//! full speed finding nothing, forever.
//!
//! Skipped with a clear message when no CUDA device is present, so that
//! `cargo test` on a machine without a GPU is not a failure.

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use num_bigint::BigUint;
use num_traits::One;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

/// The field modulus, 2^255 - 19.
fn modulus() -> BigUint {
    (BigUint::one() << 255u32) - BigUint::from(19u32)
}

/// Interpret 32 bytes the way `fe_frombytes` does: little-endian, top bit
/// ignored, then reduced.
fn from_bytes(b: &[u8; 32]) -> BigUint {
    let mut c = *b;
    c[31] &= 0x7f;
    BigUint::from_bytes_le(&c) % modulus()
}

/// Canonical 32-byte little-endian encoding, matching `fe_tobytes`.
fn to_bytes(v: &BigUint) -> [u8; 32] {
    let reduced = v % modulus();
    let mut out = [0u8; 32];
    let le = reduced.to_bytes_le();
    out[..le.len()].copy_from_slice(&le);
    out
}

/// A live GPU plus the compiled test kernels.
struct Harness {
    ctx: Arc<CudaContext>,
    module: Arc<cudarc::driver::CudaModule>,
}

impl Harness {
    /// Set up for the default (radix-25.5) field implementation.
    fn new() -> Option<Self> {
        Self::with_defines(&[])
    }

    /// Set up for the 8x32-limb implementation.
    fn new_u32() -> Option<Self> {
        Self::with_defines(&[("FE_RADIX32", "1".to_owned())])
    }

    /// Set up, or return `None` when no usable CUDA device is present.
    fn with_defines(defines: &[(&str, String)]) -> Option<Self> {
        // Ask before touching cudarc: with no driver library installed it
        // panics out of a `OnceLock` initialiser rather than returning an
        // error, so the `Err` arm below never gets the chance to skip.
        if !honion_gpu::search::cuda::driver_present() {
            eprintln!("skipping: no CUDA driver library present");
            return None;
        }
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no CUDA device available ({e:?})");
                return None;
            }
        };
        let (major, minor) = ctx.compute_capability().expect("compute capability");
        eprintln!("device compute capability {major}.{minor}");
        let ptx = honion_gpu::nvrtc::compile_cached(
            honion_gpu::nvrtc::sources::TESTKERNELS,
            (major as u32, minor as u32),
            defines,
        )
        .expect("device sources must compile");
        let module = ctx.load_module(ptx.into()).expect("module loads");
        Some(Self { ctx, module })
    }

    fn run_unary(&self, name: &str, input: &[[u8; 32]]) -> Vec<[u8; 32]> {
        self.run(name, input, None, None)
    }

    fn run_binary(
        &self,
        name: &str,
        a: &[[u8; 32]],
        b: &[[u8; 32]],
        op: Option<u32>,
    ) -> Vec<[u8; 32]> {
        self.run(name, a, Some(b), op)
    }

    fn run(
        &self,
        name: &str,
        a: &[[u8; 32]],
        b: Option<&[[u8; 32]]>,
        op: Option<u32>,
    ) -> Vec<[u8; 32]> {
        let n = a.len();
        let stream = self.ctx.default_stream();
        let func = self.module.load_function(name).expect("kernel present");

        let flat_a: Vec<u8> = a.iter().flatten().copied().collect();
        let d_a: CudaSlice<u8> = stream.clone_htod(&flat_a).expect("upload a");
        let d_b: Option<CudaSlice<u8>> = b.map(|b| {
            let flat: Vec<u8> = b.iter().flatten().copied().collect();
            stream.clone_htod(&flat).expect("upload b")
        });
        let mut d_out: CudaSlice<u8> = stream.alloc_zeros(n * 32).expect("alloc out");

        let n_u32 = n as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_u32.div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&func);
        builder.arg(&d_a);
        if let Some(ref d_b) = d_b {
            builder.arg(d_b);
        }
        builder.arg(&mut d_out);
        builder.arg(&n_u32);
        if let Some(ref op) = op {
            builder.arg(op);
        }
        // Safety: the argument list above matches the kernel signature, the
        // buffers are sized `n * 32`, and every kernel bounds-checks its index
        // against `n`.
        unsafe { builder.launch(cfg) }.expect("launch");
        stream.synchronize().expect("sync");

        let flat: Vec<u8> = stream.clone_dtoh(&d_out).expect("download");
        flat.chunks_exact(32)
            .map(|c| {
                let mut r = [0u8; 32];
                r.copy_from_slice(c);
                r
            })
            .collect()
    }
}

impl Harness {
    /// Run `test_fe_prefix`, which emits 8 bytes per input.
    fn run_prefix(&self, a: &[[u8; 32]]) -> Vec<[u8; 8]> {
        let n = a.len();
        let stream = self.ctx.default_stream();
        let func = self.module.load_function("test_fe_prefix").expect("kernel");
        let flat: Vec<u8> = a.iter().flatten().copied().collect();
        let d_a: CudaSlice<u8> = stream.clone_htod(&flat).expect("upload");
        let mut d_out: CudaSlice<u8> = stream.alloc_zeros(n * 8).expect("alloc");
        let n32 = n as u32;
        let mut b = stream.launch_builder(&func);
        b.arg(&d_a).arg(&mut d_out).arg(&n32);
        // Safety: signature and buffer sizes match; the kernel bounds-checks n.
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (n32.div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("launch");
        stream.synchronize().expect("sync");
        stream
            .clone_dtoh(&d_out)
            .expect("download")
            .chunks_exact(8)
            .map(|c| { let mut r = [0u8; 8]; r.copy_from_slice(c); r })
            .collect()
    }
}

/// Random 32-byte records, preceded by edge cases that stress reduction.
fn test_inputs(count: usize, seed: u64) -> Vec<[u8; 32]> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<[u8; 32]> = Vec::with_capacity(count + 8);

    // Values sitting exactly where a carry chain or a final reduction is most
    // likely to be wrong.
    let p = modulus();
    let edges = [
        BigUint::from(0u32),
        BigUint::one(),
        p.clone() - BigUint::one(),                  // p-1, largest canonical
        p.clone() - BigUint::from(2u32),
        (BigUint::one() << 255u32) - BigUint::one(), // 2^255-1, above p
        BigUint::one() << 254u32,
        (BigUint::one() << 26u32) - BigUint::one(),  // a full even limb
        (BigUint::one() << 25u32) - BigUint::one(),  // a full odd limb
    ];
    for e in edges {
        v.push(to_bytes(&e));
    }
    while v.len() < count {
        let mut b = [0u8; 32];
        rng.fill(&mut b);
        v.push(b);
    }
    v
}

/// Compare GPU output against a host expectation, reporting the first
/// disagreement in full.
fn assert_agrees(
    label: &str,
    inputs: &[([u8; 32], Option<[u8; 32]>)],
    got: &[[u8; 32]],
    expect: impl Fn(&BigUint, Option<&BigUint>) -> BigUint,
) {
    for (i, ((a, b), g)) in inputs.iter().zip(got).enumerate() {
        let av = from_bytes(a);
        let bv = b.map(|x| from_bytes(&x));
        let want = to_bytes(&expect(&av, bv.as_ref()));
        assert_eq!(
            g, &want,
            "{label} disagreed at case {i}\n  a = {av}\n  b = {bv:?}\n  gpu  = {g:02x?}\n  host = {want:02x?}"
        );
    }
}

const CASES: usize = 4096;

#[test]
fn field_arithmetic_matches_bigint_reference() {
    let Some(h) = Harness::new() else { return };
    run_field_suite(&h);
}

/// The same differential suite, against the 8x32-limb implementation.
///
/// This is the gate the fast field arithmetic had to pass before it was allowed
/// anywhere near the search kernel. Identical checks, identical reference: only
/// the limb layout underneath differs.
#[test]
fn field_arithmetic_u32_matches_bigint_reference() {
    let Some(h) = Harness::new_u32() else { return };
    run_field_suite(&h);
}

fn run_field_suite(h: &Harness) {
    let p = modulus();
    let a = test_inputs(CASES, 1);
    let b = test_inputs(CASES, 2);
    let paired: Vec<([u8; 32], Option<[u8; 32]>)> =
        a.iter().zip(&b).map(|(x, y)| (*x, Some(*y))).collect();

    let add = h.run_binary("test_fe_binop", &a, &b, Some(0));
    assert_agrees("fe_add", &paired, &add, |x, y| {
        (x + y.expect("binary")) % modulus()
    });

    let sub = h.run_binary("test_fe_binop", &a, &b, Some(1));
    assert_agrees("fe_sub", &paired, &sub, |x, y| {
        (x + modulus() - y.expect("binary")) % modulus()
    });

    let mul = h.run_binary("test_fe_binop", &a, &b, Some(2));
    assert_agrees("fe_mul", &paired, &mul, |x, y| {
        (x * y.expect("binary")) % modulus()
    });

    let unary: Vec<([u8; 32], Option<[u8; 32]>)> = a.iter().map(|x| (*x, None)).collect();

    let sq = h.run_unary("test_fe_sq", &a);
    assert_agrees("fe_sq", &unary, &sq, |x, _| (x * x) % modulus());

    let rt = h.run_unary("test_fe_roundtrip", &a);
    assert_agrees("fe_frombytes/fe_tobytes", &unary, &rt, |x, _| x.clone());

    // Inversion is x^(p-2). Zero inverts to zero in this construction, which
    // the reference mirrors rather than the test papering over it.
    let inv = h.run_unary("test_fe_invert", &a);
    assert_agrees("fe_invert", &unary, &inv, move |x, _| {
        if x == &BigUint::from(0u32) {
            BigUint::from(0u32)
        } else {
            x.modpow(&(p.clone() - BigUint::from(2u32)), &modulus())
        }
    });

    // The prefilter probe: the first eight canonical bytes, big-endian. This is
    // the value the search kernel compares against its pattern tables, so a
    // disagreement here would make the search look at the wrong bits.
    let probe = h.run_prefix(&a);
    for (i, bytes) in a.iter().enumerate() {
        let canonical = to_bytes(&from_bytes(bytes));
        let mut want = [0u8; 8];
        want.copy_from_slice(&canonical[..8]);
        assert_eq!(probe[i], want, "fe_prefix_be64 at case {i}");
    }
}

#[test]
fn multiplication_is_correct_on_unnormalised_limbs() {
    // The search kernel multiplies the *output of an addition*, whose limbs are
    // up to twice as large as a freshly decoded element. A `fe_mul` correct
    // only on small limbs would pass every other test here and still be wrong
    // in production, so that case is exercised explicitly.
    let Some(h) = Harness::new() else { return };
    let a = test_inputs(CASES, 3);
    let b = test_inputs(CASES, 4);
    let paired: Vec<([u8; 32], Option<[u8; 32]>)> =
        a.iter().zip(&b).map(|(x, y)| (*x, Some(*y))).collect();

    let got = h.run_binary("test_fe_mul_unnormalised", &a, &b, None);
    assert_agrees("(a+b)*(a-b)", &paired, &got, |x, y| {
        let y = y.expect("binary");
        let m = modulus();
        let sum = (x + y) % &m;
        let diff = (x + &m - y) % &m;
        (sum * diff) % m
    });
}

#[test]
fn inversion_actually_inverts() {
    // A direct algebraic check, independent of the reference exponentiation:
    // x * x^-1 must be 1 for every non-zero x.
    let Some(h) = Harness::new() else { return };
    let a: Vec<[u8; 32]> = test_inputs(CASES, 5)
        .into_iter()
        .filter(|b| from_bytes(b) != BigUint::from(0u32))
        .collect();
    let inv = h.run_unary("test_fe_invert", &a);
    let prod = h.run_binary("test_fe_binop", &a, &inv, Some(2));
    let one = to_bytes(&BigUint::one());
    for (i, p) in prod.iter().enumerate() {
        assert_eq!(p, &one, "x * x^-1 != 1 at case {i}, x = {:02x?}", a[i]);
    }
}

#[test]
fn predicates_match_the_canonical_encoding() {
    let Some(h) = Harness::new() else { return };
    let a = test_inputs(1024, 6);
    let stream = h.ctx.default_stream();
    let func = h
        .module
        .load_function("test_fe_predicates")
        .expect("kernel present");
    let flat: Vec<u8> = a.iter().flatten().copied().collect();
    let d_a: CudaSlice<u8> = stream.clone_htod(&flat).expect("upload");
    let mut d_out: CudaSlice<u8> = stream.alloc_zeros(a.len() * 2).expect("alloc");
    let n = a.len() as u32;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(&d_a).arg(&mut d_out).arg(&n);
    // Safety: signature and buffer sizes match; the kernel bounds-checks `n`.
    unsafe { builder.launch(cfg) }.expect("launch");
    stream.synchronize().expect("sync");
    let out = stream.clone_dtoh(&d_out).expect("download");

    for (i, bytes) in a.iter().enumerate() {
        let v = from_bytes(bytes);
        let canonical = to_bytes(&v);
        assert_eq!(
            out[2 * i],
            u8::from(v != BigUint::from(0u32)),
            "fe_isnonzero at case {i}"
        );
        assert_eq!(out[2 * i + 1], canonical[0] & 1, "fe_isnegative at case {i}");
    }
}
