// Test-only kernels exposing the field and group primitives one at a time.
//
// These exist so that `honion-gpu`'s differential test can drive each routine
// in isolation and compare it against an independent implementation on the
// host. They are compiled only by the test harness and are not part of the
// search kernel.
//
// Every entry point takes and returns canonical 32-byte little-endian field
// elements, so the host side never has to know anything about the radix-25.5
// limb representation. That keeps the comparison honest: the test checks the
// mathematical behaviour, not our own idea of how it should be encoded.

#if FE_RADIX32
#include "fe25519_u32.cuh"
#else
#include "fe25519.cuh"
#endif

// Binary field operations, selected by `op`.
enum FeBinOp : unsigned { FE_OP_ADD = 0, FE_OP_SUB = 1, FE_OP_MUL = 2 };

extern "C" __global__ void test_fe_binop(const u8 *a, const u8 *b, u8 *out,
                                         unsigned n, unsigned op) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fe x, y, r;
    fe_frombytes(x, a + 32 * i);
    fe_frombytes(y, b + 32 * i);
    switch (op) {
        case FE_OP_ADD: fe_add(r, x, y); break;
        case FE_OP_SUB: fe_sub(r, x, y); break;
        default:        fe_mul(r, x, y); break;
    }
    fe_tobytes(out + 32 * i, r);
}

extern "C" __global__ void test_fe_sq(const u8 *a, u8 *out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fe x, r;
    fe_frombytes(x, a + 32 * i);
    fe_sq(r, x);
    fe_tobytes(out + 32 * i, r);
}

extern "C" __global__ void test_fe_invert(const u8 *a, u8 *out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fe x, r;
    fe_frombytes(x, a + 32 * i);
    fe_invert(r, x);
    fe_tobytes(out + 32 * i, r);
}

// Round-trip: bytes -> limbs -> bytes. Must be the identity on canonical input.
extern "C" __global__ void test_fe_roundtrip(const u8 *a, u8 *out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fe x;
    fe_frombytes(x, a + 32 * i);
    fe_tobytes(out + 32 * i, x);
}

// Exercise `fe_mul` on deliberately un-normalised inputs.
//
// Ordinary tests feed `fe_mul` freshly decoded limbs, which are small. The real
// kernel feeds it the output of `fe_add`, whose limbs can be twice as large.
// This kernel reproduces that: it forms (a+b) * (a-b) through the un-normalised
// add and sub paths, so the multiplication sees limbs near their upper bound.
// The host compares against (a+b)*(a-b) mod p computed exactly.
extern "C" __global__ void test_fe_mul_unnormalised(const u8 *a, const u8 *b,
                                                    u8 *out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fe x, y, s, d, r;
    fe_frombytes(x, a + 32 * i);
    fe_frombytes(y, b + 32 * i);
    fe_add(s, x, y);
    fe_sub(d, x, y);
    fe_mul(r, s, d);
    fe_tobytes(out + 32 * i, r);
}

// The 64-bit prefilter probe: the first eight canonical bytes, big-endian.
// This is the value the search kernel actually compares, so it is checked
// directly rather than inferred from fe_tobytes.
extern "C" __global__ void test_fe_prefix(const u8 *a, u8 *out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fe x;
    fe_frombytes(x, a + 32 * i);
    u64 p = fe_prefix_be64(x);
    for (int k = 0; k < 8; k++) out[8 * i + k] = (u8)(p >> (56 - 8 * k));
}

// Report the predicates used to compress a point.
extern "C" __global__ void test_fe_predicates(const u8 *a, u8 *out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fe x;
    fe_frombytes(x, a + 32 * i);
    out[2 * i + 0] = (u8)fe_isnonzero(x);
    out[2 * i + 1] = (u8)fe_isnegative(x);
}

// ---------------------------------------------------------------------------
// Group arithmetic
// ---------------------------------------------------------------------------
//
// Skipped when testing the 8x32 field implementation on its own: the curve
// constants are emitted for whichever limb layout is active, and the field
// tests below are what gate that layout before it is wired into the group code.
#if !FE_RADIX32
#include "ge25519.cuh"

// Decompress then recompress. Must be the identity on valid public keys, and
// must report failure on encodings that name no point.
extern "C" __global__ void test_ge_roundtrip(const u8 *a, u8 *out, u8 *ok,
                                             unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ge_p3 p;
    u32 good = ge_frombytes(&p, a + 32 * i);
    ok[i] = (u8)good;
    if (!good) {
        for (int j = 0; j < 32; j++) out[32 * i + j] = 0;
        return;
    }
    ge_p3_tobytes(out + 32 * i, &p);
}

// Add 8*B to a point `steps` times and return the compressed result.
//
// This is the exact operation the search kernel performs, exercised here
// against a host that computes (a + 8*steps) * B independently.
extern "C" __global__ void test_ge_walk(const u8 *a, u8 *out, unsigned steps,
                                        unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ge_p3 p;
    if (!ge_frombytes(&p, a + 32 * i)) {
        for (int j = 0; j < 32; j++) out[32 * i + j] = 0xff;
        return;
    }
    for (unsigned s = 0; s < steps; s++) ge_add_8b(&p);
    ge_p3_tobytes(out + 32 * i, &p);
}

// Report whether each decompressed point satisfies the curve equation.
extern "C" __global__ void test_ge_on_curve(const u8 *a, u8 *out, unsigned steps,
                                            unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ge_p3 p;
    if (!ge_frombytes(&p, a + 32 * i)) { out[i] = 2; return; }
    for (unsigned s = 0; s < steps; s++) ge_add_8b(&p);
    out[i] = (u8)ge_p3_is_on_curve(&p);
}
#endif  // !FE_RADIX32
