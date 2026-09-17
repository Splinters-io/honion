// Warp-cooperative field arithmetic in GF(2^255 - 19).
//
// Eight consecutive warp lanes form a cooperative group. Lane t holds limb t
// of each field element — a single u32. The full element is
//   Σ limb[t] * 2^(32*t)  for t = 0..7.
//
// This cuts register demand from ~128/thread (scalar fe_mul with carry chains)
// to ~20/thread, enabling 3-4× occupancy. Carries propagate via __shfl_sync.
//
// Requires CC 7.0+ (__shfl_sync with explicit mask).
//
// Every function takes and returns individual u32 limbs, not arrays. The caller
// holds one limb per field element in a register; the cooperative group holds
// the full element.

#pragma once

#include "fe25519_u32.cuh"   // for fe, fe_frombytes, fe_tobytes, typedefs

#define COOP_WIDTH 8

// Mask covering only this cooperative group's 8 threads within the warp.
// With 0xFFFFFFFF all 32 warp threads must be at the same shuffle — which
// breaks when groups diverge (active vs inactive, branches in frombytes).
// A per-group mask lets each group's 8 threads synchronize independently.
__device__ __forceinline__ u32 __coop_mask() {
    return 0xFFu << ((threadIdx.x & 31) & ~(COOP_WIDTH - 1));
}
#define COOP_FULL_MASK __coop_mask()

// ---------------------------------------------------------------------------
// Carry / borrow propagation
// ---------------------------------------------------------------------------

// Propagate a two-word carry (c0 + c1·2^32) rightward across 8 lanes.
// After return, `limb` holds the normalised u32 value and (c0, c1) is the
// residual carry-out (non-zero only for lane 7 in the worst case).
__device__ __forceinline__ void coop_carry_prop(
    u32 &limb, u32 &c0, u32 &c1, int lane
) {
    #pragma unroll
    for (int round = 0; round < 7; round++) {
        u32 in0 = __shfl_up_sync(COOP_FULL_MASK, c0, 1, COOP_WIDTH);
        u32 in1 = __shfl_up_sync(COOP_FULL_MASK, c1, 1, COOP_WIDTH);
        if (lane == 0) { in0 = 0; in1 = 0; }
        // Lanes 0–6: carry was consumed by the right neighbour; reset.
        // Lane 7: no right neighbour — accumulate incoming carries into it.
        if (lane != COOP_WIDTH - 1) { c0 = 0; c1 = 0; }
        u32 prev = limb;
        limb += in0;
        u32 ov = (limb < prev) ? 1u : 0u;
        u32 add = in1 + ov;
        prev = c0;
        c0 += add;
        u32 ov2 = (c0 < prev) ? 1u : 0u;
        c1 += ov2;
    }
}

// Propagate a single-bit carry rightward across 8 lanes.
__device__ __forceinline__ void coop_carry_prop1(
    u32 &limb, u32 &carry, int lane
) {
    #pragma unroll
    for (int round = 0; round < 7; round++) {
        u32 c_in = __shfl_up_sync(COOP_FULL_MASK, carry, 1, COOP_WIDTH);
        if (lane == 0) c_in = 0;
        if (lane != COOP_WIDTH - 1) carry = 0;
        u32 prev = limb;
        limb += c_in;
        carry += (limb < prev) ? 1u : 0u;
    }
}

// Propagate a single-bit borrow rightward across 8 lanes.
__device__ __forceinline__ void coop_borrow_prop1(
    u32 &limb, u32 &borrow, int lane
) {
    #pragma unroll
    for (int round = 0; round < 7; round++) {
        u32 b_in = __shfl_up_sync(COOP_FULL_MASK, borrow, 1, COOP_WIDTH);
        if (lane == 0) b_in = 0;
        if (lane != COOP_WIDTH - 1) borrow = 0;
        u32 prev = limb;
        limb -= b_in;
        borrow += (prev < b_in) ? 1u : 0u;
    }
}

// ---------------------------------------------------------------------------
// 96-bit accumulator for the schoolbook multiply
// ---------------------------------------------------------------------------

// Add a 64-bit product (plo, phi) to a 96-bit accumulator (a0, a1, a2).
__device__ __forceinline__ void coop_acc96(
    u32 &a0, u32 &a1, u32 &a2, u32 plo, u32 phi
) {
    u32 prev = a0;
    a0 += plo;
    u32 c = (a0 < prev) ? 1u : 0u;
    prev = a1;
    a1 += phi;
    u32 c2 = (a1 < prev) ? 1u : 0u;
    prev = a1;
    a1 += c;
    c2 += (a1 < prev) ? 1u : 0u;
    a2 += c2;
}

// ---------------------------------------------------------------------------
// Field operations — cooperative layout
// ---------------------------------------------------------------------------

// h = f + g (mod p), result in [0, 2^256).
__device__ __forceinline__ void fe_add_coop(u32 &h, u32 f, u32 g) {
    int lane = threadIdx.x & (COOP_WIDTH - 1);
    h = f + g;
    u32 carry = (h < f) ? 1u : 0u;
    coop_carry_prop1(h, carry, lane);

    u32 top = __shfl_sync(COOP_FULL_MASK, carry, COOP_WIDTH - 1, COOP_WIDTH);
    carry = 0;
    if (lane == 0) {
        u32 prev = h;
        h += top * 38u;
        carry = (h < prev) ? 1u : 0u;
    }
    coop_carry_prop1(h, carry, lane);

    u32 top2 = __shfl_sync(COOP_FULL_MASK, carry, COOP_WIDTH - 1, COOP_WIDTH);
    if (lane == 0) h += top2 * 38u;
}

// h = f - g (mod p), result in [0, 2^256).
__device__ __forceinline__ void fe_sub_coop(u32 &h, u32 f, u32 g) {
    int lane = threadIdx.x & (COOP_WIDTH - 1);
    h = f - g;
    u32 borrow = (f < g) ? 1u : 0u;
    coop_borrow_prop1(h, borrow, lane);

    u32 top = __shfl_sync(COOP_FULL_MASK, borrow, COOP_WIDTH - 1, COOP_WIDTH);
    borrow = 0;
    if (lane == 0) {
        u32 sub = top * 38u;
        borrow = (h < sub) ? 1u : 0u;
        h -= sub;
    }
    coop_borrow_prop1(h, borrow, lane);

    u32 top2 = __shfl_sync(COOP_FULL_MASK, borrow, COOP_WIDTH - 1, COOP_WIDTH);
    if (lane == 0) h -= top2 * 38u;
}

// h = f * g (mod p). The hot path.
//
// Schoolbook 8×8 with 96-bit accumulators, merged reduction (no separate
// carry propagation for the lower and upper halves), single carry-propagation
// pass at the end.
__device__ __forceinline__ void fe_mul_coop(u32 &h, u32 f, u32 g) {
    const int lane = threadIdx.x & (COOP_WIDTH - 1);

    // 96-bit accumulators for positions `lane` (lower) and `lane+8` (upper).
    u32 lo0 = 0, lo1 = 0, lo2 = 0;
    u32 hi0 = 0, hi1 = 0, hi2 = 0;

    // Each thread computes 8 products. For row i: broadcast f[i], fetch
    // g[(lane−i) mod 8]. The index is the same for both halves, so one
    // g-shuffle serves both. The product goes to the lower accumulator
    // when i ≤ lane (position lane), upper when i > lane (position lane+8).
    #pragma unroll
    for (int i = 0; i < COOP_WIDTH; i++) {
        u32 fi = __shfl_sync(COOP_FULL_MASK, f, i, COOP_WIDTH);
        u32 gv = __shfl_sync(COOP_FULL_MASK, g, (lane - i) & (COOP_WIDTH - 1),
                             COOP_WIDTH);

        u32 plo, phi;
        asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(plo) : "r"(fi), "r"(gv));
        asm volatile("mul.hi.u32 %0, %1, %2;" : "=r"(phi) : "r"(fi), "r"(gv));

        if (i <= lane)
            coop_acc96(lo0, lo1, lo2, plo, phi);
        else
            coop_acc96(hi0, hi1, hi2, plo, phi);
    }

    // Merged reduction: V[lane] = lo_value + 38 · hi_value.
    // Compute as a 96-bit triple (r0, r1, r2) per thread, then carry-prop.

    // 38 × hi0
    u32 m0l, m0h;
    asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(m0l) : "r"(hi0), "r"(38u));
    asm volatile("mul.hi.u32 %0, %1, %2;" : "=r"(m0h) : "r"(hi0), "r"(38u));

    // 38 × hi1
    u32 m1l, m1h;
    asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(m1l) : "r"(hi1), "r"(38u));
    asm volatile("mul.hi.u32 %0, %1, %2;" : "=r"(m1h) : "r"(hi1), "r"(38u));

    // 38 × hi2 — hi2 ≤ 8, fits in one word
    u32 m2l = hi2 * 38u;

    // Word 0: lo0 + m0l
    u32 r0 = lo0 + m0l;
    u32 ov0 = (r0 < lo0) ? 1u : 0u;

    // Word 1: lo1 + m0h + m1l + ov0
    u32 r1 = lo1;
    u32 prev;
    u32 ov1 = 0;
    prev = r1; r1 += m0h;  ov1 += (r1 < prev) ? 1u : 0u;
    prev = r1; r1 += m1l;  ov1 += (r1 < prev) ? 1u : 0u;
    prev = r1; r1 += ov0;  ov1 += (r1 < prev) ? 1u : 0u;

    // Word 2: lo2 + m1h + m2l + ov1 — all small, no overflow
    u32 r2 = lo2 + m1h + m2l + ov1;

    // Carry-propagate the 96-bit values across 8 lanes.
    coop_carry_prop(r0, r1, r2, lane);

    // Fold carry-out from lane 7 (multiples of 2^256 ≡ 38 mod p).
    // The carry is a 64-bit value (c0 + c1·2^32) that can be large (c0 up
    // to ~2^32 from the accumulated 96-bit carries). Compute 38 × carry as
    // a 64-bit result and distribute across lanes 0 and 1.
    u32 top_c0 = __shfl_sync(COOP_FULL_MASK, r1, COOP_WIDTH - 1, COOP_WIDTH);
    u32 top_c1 = __shfl_sync(COOP_FULL_MASK, r2, COOP_WIDTH - 1, COOP_WIDTH);

    u32 fold_lo, fold_hi;
    asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(fold_lo) : "r"(top_c0), "r"(38u));
    asm volatile("mul.hi.u32 %0, %1, %2;" : "=r"(fold_hi) : "r"(top_c0), "r"(38u));
    fold_hi += top_c1 * 38u;

    u32 carry = 0;
    if (lane == 0) {
        prev = r0;
        r0 += fold_lo;
        carry = (r0 < prev) ? 1u : 0u;
    } else if (lane == 1) {
        prev = r0;
        r0 += fold_hi;
        carry = (r0 < prev) ? 1u : 0u;
    }
    coop_carry_prop1(r0, carry, lane);

    u32 top2 = __shfl_sync(COOP_FULL_MASK, carry, COOP_WIDTH - 1, COOP_WIDTH);
    if (lane == 0) r0 += top2 * 38u;

    h = r0;
}

// h = f² (mod p).
__device__ __forceinline__ void fe_sq_coop(u32 &h, u32 f) {
    fe_mul_coop(h, f, f);
}

// ---------------------------------------------------------------------------
// Canonicalisation and serialisation
// ---------------------------------------------------------------------------

// Reduce to the canonical representative in [0, p).
__device__ __forceinline__ u32 fe_freeze_coop(u32 t) {
    int lane = threadIdx.x & (COOP_WIDTH - 1);

    // Stage 1: fold bit 255.
    u32 top = __shfl_sync(COOP_FULL_MASK, t >> 31, COOP_WIDTH - 1, COOP_WIDTH);
    if (lane == COOP_WIDTH - 1) t &= 0x7FFFFFFFu;

    u32 carry = 0;
    if (lane == 0) {
        u32 prev = t;
        t += top * 19u;
        carry = (t < prev) ? 1u : 0u;
    }
    coop_carry_prop1(t, carry, lane);
    // The carry-out here cannot reach 2^256 (we started below 2^256 and
    // added at most 19), so no fold is needed.

    // Stage 2: q = t + 19; t ≥ p exactly when bit 255 of q[7] is set.
    u32 q = t;
    u32 qcarry = 0;
    if (lane == 0) {
        u32 prev = q;
        q += 19u;
        qcarry = (q < prev) ? 1u : 0u;
    }
    coop_carry_prop1(q, qcarry, lane);

    u32 q7_top = __shfl_sync(COOP_FULL_MASK, q >> 31, COOP_WIDTH - 1, COOP_WIDTH);
    u32 mask = (u32)(-(i32)q7_top);
    if (lane == COOP_WIDTH - 1) q &= 0x7FFFFFFFu;

    t = (t & ~mask) | (q & mask);
    return t;
}

// Load one limb of a 32-byte little-endian field element.
__device__ __forceinline__ u32 fe_frombytes_coop(const u8 *s) {
    int lane = threadIdx.x & (COOP_WIDTH - 1);
    const u8 *p = s + 4 * lane;
    u32 limb = (u32)p[0] | ((u32)p[1] << 8)
             | ((u32)p[2] << 16) | ((u32)p[3] << 24);
    if (lane == COOP_WIDTH - 1) limb &= 0x7FFFFFFFu;
    return limb;
}

// Store a cooperative field element as 32 little-endian bytes (fully reduced).
__device__ __forceinline__ void fe_tobytes_coop(u8 *s, u32 limb) {
    int lane = threadIdx.x & (COOP_WIDTH - 1);
    u32 t = fe_freeze_coop(limb);
    u8 *p = s + 4 * lane;
    p[0] = (u8)(t);
    p[1] = (u8)(t >> 8);
    p[2] = (u8)(t >> 16);
    p[3] = (u8)(t >> 24);
}

// Scatter a scalar fe into cooperative layout: each thread reads its limb.
__device__ __forceinline__ u32 fe_scatter(const fe src) {
    int lane = threadIdx.x & (COOP_WIDTH - 1);
    return src[lane];
}

// Gather a cooperative field element into a scalar fe.
// All 8 threads write their limb; the full fe is valid after the call.
__device__ __forceinline__ void fe_gather(fe dst, u32 limb) {
    int lane = threadIdx.x & (COOP_WIDTH - 1);
    dst[lane] = limb;
}

// ---------------------------------------------------------------------------
// Cooperative utility functions
// ---------------------------------------------------------------------------

__device__ __forceinline__ void fe_1_coop(u32 &h) {
    int lane = threadIdx.x & (COOP_WIDTH - 1);
    h = (lane == 0) ? 1u : 0u;
}

__device__ __forceinline__ void fe_neg_coop(u32 &h, u32 f) {
    u32 zero = 0;
    fe_sub_coop(h, zero, f);
}

__device__ __forceinline__ u32 fe_isnonzero_coop(u32 f) {
    u32 t = fe_freeze_coop(f);
    #pragma unroll
    for (int d = COOP_WIDTH / 2; d >= 1; d >>= 1)
        t |= __shfl_xor_sync(COOP_FULL_MASK, t, d, COOP_WIDTH);
    return t != 0;
}

__device__ __forceinline__ u32 fe_isnegative_coop(u32 f) {
    u32 t = fe_freeze_coop(f);
    u32 bit = __shfl_sync(COOP_FULL_MASK, t & 1u, 0, COOP_WIDTH);
    return bit;
}

__device__ __forceinline__ u64 fe_prefix_be64_coop(u32 limb) {
    u32 t = fe_freeze_coop(limb);
    u32 lo = __shfl_sync(COOP_FULL_MASK, t, 0, COOP_WIDTH);
    u32 hi = __shfl_sync(COOP_FULL_MASK, t, 1, COOP_WIDTH);
    u32 a = __byte_perm(lo, 0, 0x0123);
    u32 b = __byte_perm(hi, 0, 0x0123);
    return ((u64)a << 32) | (u64)b;
}

// z^(2^252 - 3), same addition chain as scalar fe_pow22523.
__device__ __noinline__ void fe_pow22523_coop(u32 &out, u32 z) {
    u32 t0, t1, t2;
    int i;
    fe_sq_coop(t0, z);
    fe_sq_coop(t1, t0); fe_sq_coop(t1, t1);
    fe_mul_coop(t1, z, t1);
    fe_mul_coop(t0, t0, t1);
    fe_sq_coop(t0, t0);
    fe_mul_coop(t0, t1, t0);
    fe_sq_coop(t1, t0); for (i = 1; i < 5; i++) fe_sq_coop(t1, t1);
    fe_mul_coop(t0, t1, t0);
    fe_sq_coop(t1, t0); for (i = 1; i < 10; i++) fe_sq_coop(t1, t1);
    fe_mul_coop(t1, t1, t0);
    fe_sq_coop(t2, t1); for (i = 1; i < 20; i++) fe_sq_coop(t2, t2);
    fe_mul_coop(t1, t2, t1);
    fe_sq_coop(t1, t1); for (i = 1; i < 10; i++) fe_sq_coop(t1, t1);
    fe_mul_coop(t0, t1, t0);
    fe_sq_coop(t1, t0); for (i = 1; i < 50; i++) fe_sq_coop(t1, t1);
    fe_mul_coop(t1, t1, t0);
    fe_sq_coop(t2, t1); for (i = 1; i < 100; i++) fe_sq_coop(t2, t2);
    fe_mul_coop(t1, t2, t1);
    fe_sq_coop(t1, t1); for (i = 1; i < 50; i++) fe_sq_coop(t1, t1);
    fe_mul_coop(t0, t1, t0);
    fe_sq_coop(t0, t0); fe_sq_coop(t0, t0);
    fe_mul_coop(out, t0, z);
}

// 1/z by Fermat: z^(p-2). Same chain as scalar fe_invert.
__device__ __noinline__ void fe_invert_coop(u32 &out, u32 z) {
    u32 t0, t1, t2, t3;
    int i;
    fe_sq_coop(t0, z);
    fe_sq_coop(t1, t0); fe_sq_coop(t1, t1);
    fe_mul_coop(t1, z, t1);
    fe_mul_coop(t0, t0, t1);
    fe_sq_coop(t2, t0);
    fe_mul_coop(t1, t1, t2);
    fe_sq_coop(t2, t1); for (i = 1; i < 5; i++) fe_sq_coop(t2, t2);
    fe_mul_coop(t1, t2, t1);
    fe_sq_coop(t2, t1); for (i = 1; i < 10; i++) fe_sq_coop(t2, t2);
    fe_mul_coop(t2, t2, t1);
    fe_sq_coop(t3, t2); for (i = 1; i < 20; i++) fe_sq_coop(t3, t3);
    fe_mul_coop(t2, t3, t2);
    fe_sq_coop(t2, t2); for (i = 1; i < 10; i++) fe_sq_coop(t2, t2);
    fe_mul_coop(t1, t2, t1);
    fe_sq_coop(t2, t1); for (i = 1; i < 50; i++) fe_sq_coop(t2, t2);
    fe_mul_coop(t2, t2, t1);
    fe_sq_coop(t3, t2); for (i = 1; i < 100; i++) fe_sq_coop(t3, t3);
    fe_mul_coop(t2, t3, t2);
    fe_sq_coop(t2, t2); for (i = 1; i < 50; i++) fe_sq_coop(t2, t2);
    fe_mul_coop(t1, t2, t1);
    fe_sq_coop(t1, t1); for (i = 1; i < 5; i++) fe_sq_coop(t1, t1);
    fe_mul_coop(out, t1, t0);
}
