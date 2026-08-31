//! Differential tests for the Metal field arithmetic.
//!
//! The Metal mirror of `tests/field_arithmetic.rs`. Every routine in
//! `metal/fe25519.metal` is run on the GPU and compared against an independent
//! `num-bigint` implementation of arithmetic mod `2^255 - 19` — the same
//! reference the CUDA field is gated against, sharing no code, representation
//! or algorithm with the device. Agreement between two implementations that
//! different is meaningful; agreement with itself is not.
//!
//! This is a gate, not a formality: the curve arithmetic and the search
//! kernel are built on these primitives and cannot be trusted
//! while they are unverified. Skipped with a clear message when no Metal device
//! is present.

// @decision DEC-METAL-004 (test side)
// @title The radix-25.5 Metal field earns the same differential bar as CUDA
// @status accepted
// @rationale This suite is a line-for-line analog of the CUDA field test — same
//   edge cases (0, 1, p-1, 2^255-1, full limbs), same num-bigint reference,
//   same un-normalised-input stress and same algebraic x*x^-1==1 check — so the
//   Metal port is held to exactly the bar the plan requires (REQ-GOAL-002, no
//   reduced test bar), not a weaker one. It drives the kernels through
//   MslLibrary, proving the field arithmetic and the MSL driver together.

use honion_gpu::msl::{self, MslError, MslKernel, MslLibrary, SharedBuffer};
use num_bigint::BigUint;
use num_traits::One;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn modulus() -> BigUint {
    (BigUint::one() << 255u32) - BigUint::from(19u32)
}

/// Interpret 32 bytes as `fe_frombytes` does: little-endian, top bit ignored.
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

/// A live Metal device plus the compiled field test kernels.
struct Harness {
    lib: MslLibrary,
}

impl Harness {
    /// Set up, or return `None` when no Metal device is present.
    fn new() -> Option<Self> {
        match MslLibrary::compile(msl::sources::TESTKERNELS, &[]) {
            Ok(lib) => Some(Self { lib }),
            Err(MslError::NoDevice) => {
                eprintln!("skipping: no Metal device available");
                None
            }
            Err(e) => panic!("field test kernels must compile: {e}"),
        }
    }

    fn kernel(&self, name: &str) -> MslKernel {
        self.lib.kernel(name).expect("kernel present")
    }

    /// A shared buffer holding `records` of 32 bytes each, uploaded.
    fn upload(&self, k: &MslKernel, records: &[[u8; 32]]) -> SharedBuffer {
        let mut buf = k.new_shared_buffer(records.len() * 32).expect("alloc");
        let flat: Vec<u8> = records.iter().flatten().copied().collect();
        buf.as_mut_slice::<u8>().copy_from_slice(&flat);
        buf
    }

    /// A shared buffer holding one `u32`.
    fn scalar(&self, k: &MslKernel, value: u32) -> SharedBuffer {
        let mut buf = k
            .new_shared_buffer(std::mem::size_of::<u32>())
            .expect("alloc");
        buf.as_mut_slice::<u32>()[0] = value;
        buf
    }

    fn read_records(out: &SharedBuffer, n: usize, width: usize) -> Vec<Vec<u8>> {
        let bytes = out.as_slice::<u8>();
        (0..n).map(|i| bytes[i * width..(i + 1) * width].to_vec()).collect()
    }

    fn run_binary(
        &self,
        a: &[[u8; 32]],
        b: &[[u8; 32]],
        op: u32,
    ) -> Vec<[u8; 32]> {
        let k = self.kernel("test_fe_binop");
        let n = a.len();
        let da = self.upload(&k, a);
        let db = self.upload(&k, b);
        let out = k.new_shared_buffer(n * 32).expect("alloc");
        let dn = self.scalar(&k, n as u32);
        let dop = self.scalar(&k, op);
        let tg = k.max_threads_per_threadgroup().min(256);
        k.dispatch(n, tg, &[&da, &db, &out, &dn, &dop]).expect("dispatch");
        Self::read_records(&out, n, 32)
            .into_iter()
            .map(|c| c.try_into().unwrap())
            .collect()
    }

    fn run_unary(&self, function: &str, a: &[[u8; 32]]) -> Vec<[u8; 32]> {
        let k = self.kernel(function);
        let n = a.len();
        let da = self.upload(&k, a);
        let out = k.new_shared_buffer(n * 32).expect("alloc");
        let dn = self.scalar(&k, n as u32);
        let tg = k.max_threads_per_threadgroup().min(256);
        k.dispatch(n, tg, &[&da, &out, &dn]).expect("dispatch");
        Self::read_records(&out, n, 32)
            .into_iter()
            .map(|c| c.try_into().unwrap())
            .collect()
    }

    fn run_mul_unnormalised(&self, a: &[[u8; 32]], b: &[[u8; 32]]) -> Vec<[u8; 32]> {
        let k = self.kernel("test_fe_mul_unnormalised");
        let n = a.len();
        let da = self.upload(&k, a);
        let db = self.upload(&k, b);
        let out = k.new_shared_buffer(n * 32).expect("alloc");
        let dn = self.scalar(&k, n as u32);
        let tg = k.max_threads_per_threadgroup().min(256);
        k.dispatch(n, tg, &[&da, &db, &out, &dn]).expect("dispatch");
        Self::read_records(&out, n, 32)
            .into_iter()
            .map(|c| c.try_into().unwrap())
            .collect()
    }

    fn run_prefix(&self, a: &[[u8; 32]]) -> Vec<[u8; 8]> {
        let k = self.kernel("test_fe_prefix");
        let n = a.len();
        let da = self.upload(&k, a);
        let out = k.new_shared_buffer(n * 8).expect("alloc");
        let dn = self.scalar(&k, n as u32);
        let tg = k.max_threads_per_threadgroup().min(256);
        k.dispatch(n, tg, &[&da, &out, &dn]).expect("dispatch");
        Self::read_records(&out, n, 8)
            .into_iter()
            .map(|c| c.try_into().unwrap())
            .collect()
    }

    fn run_predicates(&self, a: &[[u8; 32]]) -> Vec<[u8; 2]> {
        let k = self.kernel("test_fe_predicates");
        let n = a.len();
        let da = self.upload(&k, a);
        let out = k.new_shared_buffer(n * 2).expect("alloc");
        let dn = self.scalar(&k, n as u32);
        let tg = k.max_threads_per_threadgroup().min(256);
        k.dispatch(n, tg, &[&da, &out, &dn]).expect("dispatch");
        Self::read_records(&out, n, 2)
            .into_iter()
            .map(|c| c.try_into().unwrap())
            .collect()
    }
}

/// Random 32-byte records, preceded by edge cases that stress reduction.
fn test_inputs(count: usize, seed: u64) -> Vec<[u8; 32]> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<[u8; 32]> = Vec::with_capacity(count + 8);
    let p = modulus();
    let edges = [
        BigUint::from(0u32),
        BigUint::one(),
        p.clone() - BigUint::one(),
        p.clone() - BigUint::from(2u32),
        (BigUint::one() << 255u32) - BigUint::one(),
        BigUint::one() << 254u32,
        (BigUint::one() << 26u32) - BigUint::one(),
        (BigUint::one() << 25u32) - BigUint::one(),
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
    let a = test_inputs(CASES, 1);
    let b = test_inputs(CASES, 2);
    let paired: Vec<([u8; 32], Option<[u8; 32]>)> =
        a.iter().zip(&b).map(|(x, y)| (*x, Some(*y))).collect();

    let add = h.run_binary(&a, &b, 0);
    assert_agrees("fe_add", &paired, &add, |x, y| (x + y.expect("binary")) % modulus());

    let sub = h.run_binary(&a, &b, 1);
    assert_agrees("fe_sub", &paired, &sub, |x, y| {
        (x + modulus() - y.expect("binary")) % modulus()
    });

    let mul = h.run_binary(&a, &b, 2);
    assert_agrees("fe_mul", &paired, &mul, |x, y| (x * y.expect("binary")) % modulus());

    let unary: Vec<([u8; 32], Option<[u8; 32]>)> = a.iter().map(|x| (*x, None)).collect();

    let sq = h.run_unary("test_fe_sq", &a);
    assert_agrees("fe_sq", &unary, &sq, |x, _| (x * x) % modulus());

    let rt = h.run_unary("test_fe_roundtrip", &a);
    assert_agrees("fe_frombytes/fe_tobytes", &unary, &rt, |x, _| x.clone());

    let p = modulus();
    let inv = h.run_unary("test_fe_invert", &a);
    assert_agrees("fe_invert", &unary, &inv, move |x, _| {
        if x == &BigUint::from(0u32) {
            BigUint::from(0u32)
        } else {
            x.modpow(&(p.clone() - BigUint::from(2u32)), &modulus())
        }
    });

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
    let Some(h) = Harness::new() else { return };
    let a = test_inputs(CASES, 3);
    let b = test_inputs(CASES, 4);
    let paired: Vec<([u8; 32], Option<[u8; 32]>)> =
        a.iter().zip(&b).map(|(x, y)| (*x, Some(*y))).collect();

    let got = h.run_mul_unnormalised(&a, &b);
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
    let Some(h) = Harness::new() else { return };
    let a: Vec<[u8; 32]> = test_inputs(CASES, 5)
        .into_iter()
        .filter(|b| from_bytes(b) != BigUint::from(0u32))
        .collect();
    let inv = h.run_unary("test_fe_invert", &a);
    let prod = h.run_binary(&a, &inv, 2);
    let one = to_bytes(&BigUint::one());
    for (i, p) in prod.iter().enumerate() {
        assert_eq!(p, &one, "x * x^-1 != 1 at case {i}, x = {:02x?}", a[i]);
    }
}

#[test]
fn predicates_match_the_canonical_encoding() {
    let Some(h) = Harness::new() else { return };
    let a = test_inputs(1024, 6);
    let out = h.run_predicates(&a);
    for (i, bytes) in a.iter().enumerate() {
        let v = from_bytes(bytes);
        let canonical = to_bytes(&v);
        assert_eq!(out[i][0], u8::from(v != BigUint::from(0u32)), "fe_isnonzero at case {i}");
        assert_eq!(out[i][1], canonical[0] & 1, "fe_isnegative at case {i}");
    }
}
