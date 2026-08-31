// Test-only kernels exposing the Metal field primitives one at a time.
//
// The mirror of the field half of `cuda/testkernels.cu`: each entry point takes
// and returns canonical 32-byte little-endian field elements, so the host side
// never sees the radix-25.5 limb layout and the differential test checks the
// mathematics, not our encoding. Compiled only by the test harness; not part of
// the search kernel.

#include "fe25519.metal"

using namespace metal;

// Binary field operations, selected by `op`: 0 add, 1 sub, 2 mul.
kernel void test_fe_binop(const device u8 *a     [[buffer(0)]],
                          const device u8 *b     [[buffer(1)]],
                          device u8 *out         [[buffer(2)]],
                          constant uint &n       [[buffer(3)]],
                          constant uint &op      [[buffer(4)]],
                          uint i                 [[thread_position_in_grid]]) {
    if (i >= n) return;
    fe x = fe_frombytes(a + 32 * i);
    fe y = fe_frombytes(b + 32 * i);
    fe r;
    if (op == 0)      r = fe_add(x, y);
    else if (op == 1) r = fe_sub(x, y);
    else              r = fe_mul(x, y);
    fe_tobytes(out + 32 * i, r);
}

kernel void test_fe_sq(const device u8 *a    [[buffer(0)]],
                       device u8 *out        [[buffer(1)]],
                       constant uint &n      [[buffer(2)]],
                       uint i                [[thread_position_in_grid]]) {
    if (i >= n) return;
    fe x = fe_frombytes(a + 32 * i);
    fe_tobytes(out + 32 * i, fe_sq(x));
}

kernel void test_fe_invert(const device u8 *a    [[buffer(0)]],
                           device u8 *out        [[buffer(1)]],
                           constant uint &n      [[buffer(2)]],
                           uint i                [[thread_position_in_grid]]) {
    if (i >= n) return;
    fe x = fe_frombytes(a + 32 * i);
    fe_tobytes(out + 32 * i, fe_invert(x));
}

// Round-trip: bytes -> limbs -> bytes. The identity on canonical input.
kernel void test_fe_roundtrip(const device u8 *a    [[buffer(0)]],
                              device u8 *out         [[buffer(1)]],
                              constant uint &n       [[buffer(2)]],
                              uint i                 [[thread_position_in_grid]]) {
    if (i >= n) return;
    fe x = fe_frombytes(a + 32 * i);
    fe_tobytes(out + 32 * i, x);
}

// fe_mul on deliberately un-normalised inputs: (a+b)*(a-b) through the
// un-normalised add/sub paths, so the multiply sees limbs near their bound.
kernel void test_fe_mul_unnormalised(const device u8 *a    [[buffer(0)]],
                                     const device u8 *b     [[buffer(1)]],
                                     device u8 *out         [[buffer(2)]],
                                     constant uint &n       [[buffer(3)]],
                                     uint i                 [[thread_position_in_grid]]) {
    if (i >= n) return;
    fe x = fe_frombytes(a + 32 * i);
    fe y = fe_frombytes(b + 32 * i);
    fe s = fe_add(x, y);
    fe d = fe_sub(x, y);
    fe_tobytes(out + 32 * i, fe_mul(s, d));
}

// The 64-bit prefilter probe: the first eight canonical bytes, big-endian.
kernel void test_fe_prefix(const device u8 *a    [[buffer(0)]],
                           device u8 *out         [[buffer(1)]],
                           constant uint &n       [[buffer(2)]],
                           uint i                 [[thread_position_in_grid]]) {
    if (i >= n) return;
    fe x = fe_frombytes(a + 32 * i);
    u64 p = fe_prefix_be64(x);
    for (int k = 0; k < 8; k++) out[8 * i + k] = (u8)(p >> (56 - 8 * k));
}

// The predicates used to compress a point: isnonzero, isnegative.
kernel void test_fe_predicates(const device u8 *a    [[buffer(0)]],
                               device u8 *out         [[buffer(1)]],
                               constant uint &n       [[buffer(2)]],
                               uint i                 [[thread_position_in_grid]]) {
    if (i >= n) return;
    fe x = fe_frombytes(a + 32 * i);
    out[2 * i + 0] = (u8)fe_isnonzero(x);
    out[2 * i + 1] = (u8)fe_isnegative(x);
}

// ---------------------------------------------------------------------------
// Group arithmetic
// ---------------------------------------------------------------------------

#include "ge25519.metal"

// Decompress then recompress. The identity on valid public keys; must report
// failure on encodings that name no point.
kernel void test_ge_roundtrip(const device u8 *a    [[buffer(0)]],
                              device u8 *out         [[buffer(1)]],
                              device u8 *ok          [[buffer(2)]],
                              constant uint &n       [[buffer(3)]],
                              uint i                 [[thread_position_in_grid]]) {
    if (i >= n) return;
    ge_decode d = ge_frombytes(a + 32 * i);
    ok[i] = (u8)d.ok;
    if (!d.ok) {
        for (int j = 0; j < 32; j++) out[32 * i + j] = 0;
        return;
    }
    ge_p3_tobytes(out + 32 * i, d.p);
}

// Add 8*B to a point `steps` times and return the compressed result — the exact
// operation the search kernel performs, checked against a host that computes
// (a + 8*steps) * B independently.
kernel void test_ge_walk(const device u8 *a    [[buffer(0)]],
                         device u8 *out         [[buffer(1)]],
                         constant uint &steps   [[buffer(2)]],
                         constant uint &n       [[buffer(3)]],
                         uint i                 [[thread_position_in_grid]]) {
    if (i >= n) return;
    ge_decode d = ge_frombytes(a + 32 * i);
    if (!d.ok) {
        for (int j = 0; j < 32; j++) out[32 * i + j] = 0xff;
        return;
    }
    ge_p3 p = d.p;
    for (uint s = 0; s < steps; s++) p = ge_add_8b(p);
    ge_p3_tobytes(out + 32 * i, p);
}

// Whether each decompressed (and optionally walked) point is on the curve.
kernel void test_ge_on_curve(const device u8 *a    [[buffer(0)]],
                             device u8 *out         [[buffer(1)]],
                             constant uint &steps   [[buffer(2)]],
                             constant uint &n       [[buffer(3)]],
                             uint i                 [[thread_position_in_grid]]) {
    if (i >= n) return;
    ge_decode d = ge_frombytes(a + 32 * i);
    if (!d.ok) { out[i] = 2; return; }
    ge_p3 p = d.p;
    for (uint s = 0; s < steps; s++) p = ge_add_8b(p);
    out[i] = (u8)ge_p3_is_on_curve(p);
}
