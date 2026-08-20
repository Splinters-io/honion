// Arithmetic in the field GF(2^255 - 19).
//
// This is the foundation everything else rests on: a single wrong coefficient
// here yields points that are not on the curve, addresses that do not
// correspond to their keys, and a search that silently finds nothing. It is
// therefore written to be *checkable* rather than clever — every routine has a
// direct counterpart in `honion-gpu`'s differential test, which compares it
// against an independent big-integer implementation on random and adversarial
// inputs before the search kernel is ever run.
//
// ## Representation
//
// A field element is ten signed 32-bit limbs with alternating radix 2^26 and
// 2^25 ("radix 25.5"), the representation used by the reference Ed25519
// implementation known as ref10:
//
//     h = h[0]
//       + h[1] * 2^26  + h[2] * 2^51  + h[3] * 2^77  + h[4] * 2^102
//       + h[5] * 2^128 + h[6] * 2^153 + h[7] * 2^179 + h[8] * 2^204
//       + h[9] * 2^230
//
// Even limbs carry 26 bits, odd limbs 25. Limbs are *signed* and the
// representation is redundant: a value has many encodings, and addition and
// subtraction may leave limbs un-normalised. Normalisation happens inside
// multiplication and in `fe_tobytes`. This is what makes add/sub cost ten
// integer operations instead of a carry chain.
//
// ### Why this representation on a GPU
//
// The alternative is eight 32-bit limbs with explicit carry chains via PTX
// `add.cc`/`addc`. That has fewer partial products (64 versus 100) but needs
// hand-written carry propagation, which is exactly the kind of code that is
// easy to get subtly wrong. Radix 25.5 keeps every intermediate in a 64-bit
// accumulator with provable headroom (see the bound analysis on `fe_mul`) and
// leaves carry handling to a single well-understood sequence. Correctness
// first; `docs/06-performance.md` records whether the trade was worth it.

#pragma once

typedef signed char        i8;
typedef unsigned char      u8;
typedef signed int         i32;
typedef unsigned int       u32;
typedef signed long long   i64;
typedef unsigned long long u64;

#define FE_LIMBS 10

// A field element. Passed by pointer; callers own the storage.
typedef i32 fe[FE_LIMBS];

// The limb type, so code that serialises limbs need not know the layout.
typedef i32 fe_limb;

// Bit width of limb `i`: 26 for even, 25 for odd.
__device__ __forceinline__ constexpr int fe_limb_bits(int i) { return (i & 1) ? 25 : 26; }

__device__ __forceinline__ void fe_0(fe h) {
#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) h[i] = 0;
}

__device__ __forceinline__ void fe_1(fe h) {
    fe_0(h);
    h[0] = 1;
}

__device__ __forceinline__ void fe_copy(fe h, const fe f) {
#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) h[i] = f[i];
}

// h = f + g. Limbs are left un-normalised; see the representation note above.
__device__ __forceinline__ void fe_add(fe h, const fe f, const fe g) {
#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) h[i] = f[i] + g[i];
}

// h = f - g. Likewise un-normalised.
__device__ __forceinline__ void fe_sub(fe h, const fe f, const fe g) {
#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) h[i] = f[i] - g[i];
}

// h = -f.
__device__ __forceinline__ void fe_neg(fe h, const fe f) {
#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) h[i] = -f[i];
}

// Conditional move: h = b ? g : h, without branching on b.
//
// `b` must be exactly 0 or 1. Written branch-free because it is used inside the
// exponentiation ladder, and because a data-dependent branch would serialise
// the warp even though every lane wants the same work done.
__device__ __forceinline__ void fe_cmov(fe h, const fe g, u32 b) {
    const i32 mask = -(i32)b;
#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) h[i] ^= mask & (h[i] ^ g[i]);
}

// Propagate carries from a 64-bit accumulator array into normalised limbs.
//
// The order below interleaves two independent chains (even limbs and odd limbs)
// so that the dependency graph is two short chains rather than one long one —
// on a GPU this matters, because a ten-deep serial dependency stalls the
// pipeline. The final wrap folds the overflow of limb 9 back into limb 0
// multiplied by 19, which is the whole point of the modulus 2^255 - 19:
// 2^255 ≡ 19 (mod p).
//
// Rounding form `(x + 2^(w-1)) >> w` keeps limbs centred on zero, which halves
// their magnitude compared with a truncating shift and buys headroom for the
// un-normalised add/sub above.
__device__ __forceinline__ void fe_carry_into(fe h, i64 t[FE_LIMBS]) {
#define CARRY(i, w, next)                                                      \
    do {                                                                       \
        i64 c = (t[i] + ((i64)1 << ((w) - 1))) >> (w);                         \
        t[next] += c;                                                          \
        t[i] -= c << (w);                                                      \
    } while (0)

    CARRY(0, 26, 1);
    CARRY(4, 26, 5);
    CARRY(1, 25, 2);
    CARRY(5, 25, 6);
    CARRY(2, 26, 3);
    CARRY(6, 26, 7);
    CARRY(3, 25, 4);
    CARRY(7, 25, 8);
    CARRY(4, 26, 5);
    CARRY(8, 26, 9);

    // Limb 9 wraps into limb 0 with the factor 19.
    {
        i64 c = (t[9] + ((i64)1 << 24)) >> 25;
        t[0] += c * 19;
        t[9] -= c << 25;
    }
    CARRY(0, 26, 1);
#undef CARRY

#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) h[i] = (i32)t[i];
}

// h = f * g.
//
// ## The algorithm
//
// Writing f = sum_i f_i 2^e(i) with e(i) = ceil(25.5 i), the product is
// sum_{i,j} f_i g_j 2^(e(i)+e(j)). Two adjustments turn that into limbs:
//
//   * When i and j are both odd, e(i) + e(j) = e(i+j) + 1, so the term carries
//     an extra factor of 2.
//   * When i + j >= 10, the term lands at 2^(e(i+j)) = 2^255 * 2^(e(i+j-10)),
//     and 2^255 ≡ 19 (mod p), so the term folds down by a factor of 19.
//
// Both conditions depend only on the loop indices, so after `#pragma unroll`
// they are resolved at compile time and cost nothing at run time. Expressing
// the rule once, rather than transcribing a hundred hand-expanded product
// terms, is deliberate: it is the difference between code that can be read for
// correctness and code that can only be tested for it.
//
// ## Overflow bound
//
// With input limbs bounded by 1.1 * 2^27 (the worst case here, since `fe_add`
// may double a normalised limb before multiplication), the largest single term
// is 2 * 19 * (1.1 * 2^27)^2 < 2^59.6, and the widest accumulator sums ten of
// them with coefficients at most 38, giving < 2^62.4. That fits a signed 64-bit
// accumulator with headroom. The differential test exercises limbs at these
// bounds explicitly.
__device__ __forceinline__ void fe_mul(fe h, const fe f, const fe g) {
    i32 g19[FE_LIMBS];
    i32 f2[FE_LIMBS];
#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) {
        g19[i] = 19 * g[i];
        f2[i] = 2 * f[i];
    }

    i64 t[FE_LIMBS];
#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) t[i] = 0;

#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) {
#pragma unroll
        for (int j = 0; j < FE_LIMBS; j++) {
            const int k = i + j;
            const bool wraps = (k >= FE_LIMBS);
            const bool doubled = (i & 1) && (j & 1);
            const i32 lhs = doubled ? f2[i] : f[i];
            const i32 rhs = wraps ? g19[j] : g[j];
            t[wraps ? k - FE_LIMBS : k] += (i64)lhs * (i64)rhs;
        }
    }
    fe_carry_into(h, t);
}

// h = f * f.
//
// Squaring admits a symmetry optimisation (each off-diagonal product appears
// twice), but it is used almost exclusively inside the exponentiation ladder,
// whose cost is amortised across an entire batch by Montgomery's trick. Keeping
// it as a multiplication removes a second hundred-term expression from the set
// of things that can be wrong. `docs/06-performance.md` records whether
// specialising it ever became worthwhile.
__device__ __forceinline__ void fe_sq(fe h, const fe f) { fe_mul(h, f, f); }

// h = f * 121666, the constant used by the Montgomery ladder. Present for
// completeness of the field API; unused by the search kernel.
__device__ __forceinline__ void fe_mul121666(fe h, const fe f) {
    i64 t[FE_LIMBS];
#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) t[i] = (i64)f[i] * 121666;
    fe_carry_into(h, t);
}

// Load a field element from 32 little-endian bytes.
//
// The top bit of byte 31 is ignored: callers strip the sign bit of a compressed
// point before calling, and accepting a 256th bit here would silently admit
// values >= 2^255 that have no canonical representation.
//
// The limb slicing looks misaligned and is not. Each `t[i]` is loaded as a
// plain little-endian window scaled so that its contribution equals the bits it
// covers; consecutive windows deliberately overlap, and the carry pass at the
// end redistributes the overlap into the true radix-25.5 limbs. Loading exact
// bit fields directly would need ten separate shift-and-mask expressions and
// gains nothing, since the carry pass is required regardless.
__device__ __forceinline__ i64 fe_load3(const u8 *s) {
    return (i64)s[0] | ((i64)s[1] << 8) | ((i64)s[2] << 16);
}

__device__ __forceinline__ i64 fe_load4(const u8 *s) {
    return (i64)s[0] | ((i64)s[1] << 8) | ((i64)s[2] << 16) | ((i64)s[3] << 24);
}

__device__ __forceinline__ void fe_frombytes(fe h, const u8 *s) {
    i64 t[FE_LIMBS];
    t[0] = fe_load4(s);
    t[1] = fe_load3(s + 4) << 6;
    t[2] = fe_load3(s + 7) << 5;
    t[3] = fe_load3(s + 10) << 3;
    t[4] = fe_load3(s + 13) << 2;
    t[5] = fe_load4(s + 16);
    t[6] = fe_load3(s + 20) << 7;
    t[7] = fe_load3(s + 23) << 5;
    t[8] = fe_load3(s + 26) << 4;
    // Bit 255 is masked off here, not by the caller.
    t[9] = (fe_load3(s + 29) & 8388607) << 2;

    fe_carry_into(h, t);
}

// Store a field element as 32 little-endian bytes, fully reduced mod p.
//
// This is the only routine that must produce a *canonical* answer: the
// redundant representation means a value has many encodings, but an onion
// address has exactly one.
//
// It is deliberately **total** — it accepts any limb values the other routines
// can produce, including the un-normalised output of `fe_add` and `fe_sub`,
// and begins by carrying them itself. The reference implementation this
// derives from instead documents a precondition that callers must normalise
// first. That is a trap: the precondition is invisible at the call site, and
// violating it yields an answer that is wrong by exactly 19 — plausible enough
// to survive casual inspection. A function whose contract cannot be violated
// is worth one extra carry pass.
// Reduce to the canonical representative, as non-negative limbs.
//
// Split out of `fe_tobytes` because the search loop needs a canonical *value*
// far more often than it needs canonical *bytes*: it tests a 64-bit prefix of
// every candidate but only serialises the rare ones that match.
//
// Measured effect on throughput: none. The compiler was already eliminating the
// packing of bytes 8..31, because nothing read them. The split is kept because
// it states that intent in the code rather than depending on the optimiser to
// rediscover it, but it is documented here as neutral rather than as a win —
// see docs/07-benchmarks.md for the hypothesis this disproved.
__device__ __forceinline__ void fe_freeze(i32 t[FE_LIMBS], const fe h) {
    // Normalise first, so no caller can violate a precondition.
    i64 acc[FE_LIMBS];
#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) acc[i] = h[i];
    fe carried;
    fe_carry_into(carried, acc);

#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) t[i] = carried[i];

    // Compute q, the number of times p must be subtracted, by simulating the
    // carry chain of h + 19 (which overflows exactly when h >= p).
    i32 q = (19 * t[9] + (((i32)1) << 24)) >> 25;
#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) q = (t[i] + q) >> fe_limb_bits(i);

    // Now 0 <= h + 19q < 2^255, so the remaining carries are exact and the
    // final mask discards nothing.
    t[0] += 19 * q;

#pragma unroll
    for (int i = 0; i < FE_LIMBS - 1; i++) {
        i32 c = t[i] >> fe_limb_bits(i);
        t[i + 1] += c;
        t[i] -= c << fe_limb_bits(i);
    }
    t[9] &= 0x1ffffff;
}

// The first eight bytes of the canonical encoding, read big-endian.
//
// This is the hot path: it is evaluated for every candidate the search
// examines, and its result is the prefilter probe. Bytes 0..7 are bits 0..63,
// which lie entirely within limbs 0, 1 and 2 (spanning bits 0..76), so seven of
// the ten limbs are never touched and the bit-by-bit packing loop is skipped.
//
// The byte reversal is because a base32 address reads the key most significant
// *byte* first while the field element is little-endian; comparing as one
// big-endian 64-bit integer lets the prefilter be a single mask and compare.
// It must agree exactly with `honion_core::pattern::key_prefix_u64`.
__device__ __forceinline__ u64 fe_prefix_be64(const fe h) {
    i32 t[FE_LIMBS];
    fe_freeze(t, h);

    // Limb 0 holds bits 0..25, limb 1 bits 26..50, limb 2 bits 51..76.
    const u64 lo = (u64)(u32)t[0] | ((u64)(u32)t[1] << 26) | ((u64)(u32)t[2] << 51);

    // Byte-swap; the compiler lowers this to two PRMT instructions.
    return ((lo & 0x00000000000000ffULL) << 56) | ((lo & 0x000000000000ff00ULL) << 40)
         | ((lo & 0x0000000000ff0000ULL) << 24) | ((lo & 0x00000000ff000000ULL) << 8)
         | ((lo & 0x000000ff00000000ULL) >> 8)  | ((lo & 0x0000ff0000000000ULL) >> 24)
         | ((lo & 0x00ff000000000000ULL) >> 40) | ((lo & 0xff00000000000000ULL) >> 56);
}

// Store a field element as 32 little-endian bytes, fully reduced mod p.
//
// This is the only routine that must produce a *canonical* answer: the
// redundant representation means a value has many encodings, but an onion
// address has exactly one.
//
// It is deliberately **total** — it accepts any limb values the other routines
// can produce, including the un-normalised output of `fe_add` and `fe_sub`.
// The reference implementation this derives from instead documents a
// precondition that callers must normalise first. That is a trap: the
// precondition is invisible at the call site, and violating it yields an answer
// that is wrong by exactly 19 — plausible enough to survive casual inspection.
__device__ __forceinline__ void fe_tobytes(u8 *s, const fe h) {
    i32 t[FE_LIMBS];
    fe_freeze(t, h);

    // Pack the limbs; their bit offsets are the running sums of their widths.
#pragma unroll
    for (int i = 0; i < 32; i++) s[i] = 0;
    int bit = 0;
#pragma unroll
    for (int i = 0; i < FE_LIMBS; i++) {
        const int w = fe_limb_bits(i);
        u32 v = (u32)t[i];
        for (int b = 0; b < w; b++) {
            const int abs_bit = bit + b;
            s[abs_bit >> 3] |= (u8)(((v >> b) & 1u) << (abs_bit & 7));
        }
        bit += w;
    }
}

// Whether f is zero modulo p. Returns 1 or 0.
__device__ __forceinline__ u32 fe_isnonzero(const fe f) {
    u8 s[32];
    fe_tobytes(s, f);
    u8 acc = 0;
#pragma unroll
    for (int i = 0; i < 32; i++) acc |= s[i];
    return acc != 0;
}

// The least significant bit of the canonical representative — the "sign" bit
// that a compressed Edwards point carries for its x coordinate.
__device__ __forceinline__ u32 fe_isnegative(const fe f) {
    u8 s[32];
    fe_tobytes(s, f);
    return s[0] & 1;
}

// out = z^(2^252 - 3). Shared skeleton for inversion and square roots.
//
// Marked `__noinline__` deliberately. It expands to 261 field operations, each
// of which is itself a hundred-term multiplication; inlining it multiplied the
// kernel's PTX by roughly fifty. It runs once per batch of BATCH_SIZE
// candidates, so a real call costs nothing measurable and keeps the hot loop
// resident in the instruction cache.
//
// The addition chain below is the standard one for this exponent: it builds
// z^(2^k - 1) for k = 1, 2, 3, 5, 10, 20, 40, 50, 100, 200, 250 by repeated
// squaring and multiplication, then finishes with two squarings and a multiply.
// 250 squarings and 11 multiplications in total.
__device__ __noinline__ void fe_pow22523(fe out, const fe z) {
    fe t0, t1, t2;
    int i;

    fe_sq(t0, z);                                   // z^2
    fe_sq(t1, t0); fe_sq(t1, t1);                   // z^8
    fe_mul(t1, z, t1);                              // z^9
    fe_mul(t0, t0, t1);                             // z^11
    fe_sq(t0, t0);                                  // z^22
    fe_mul(t0, t1, t0);                             // z^(2^5 - 1)
    fe_sq(t1, t0); for (i = 1; i < 5; i++) fe_sq(t1, t1);
    fe_mul(t0, t1, t0);                             // z^(2^10 - 1)
    fe_sq(t1, t0); for (i = 1; i < 10; i++) fe_sq(t1, t1);
    fe_mul(t1, t1, t0);                             // z^(2^20 - 1)
    fe_sq(t2, t1); for (i = 1; i < 20; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);                             // z^(2^40 - 1)
    fe_sq(t1, t1); for (i = 1; i < 10; i++) fe_sq(t1, t1);
    fe_mul(t0, t1, t0);                             // z^(2^50 - 1)
    fe_sq(t1, t0); for (i = 1; i < 50; i++) fe_sq(t1, t1);
    fe_mul(t1, t1, t0);                             // z^(2^100 - 1)
    fe_sq(t2, t1); for (i = 1; i < 100; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);                             // z^(2^200 - 1)
    fe_sq(t1, t1); for (i = 1; i < 50; i++) fe_sq(t1, t1);
    fe_mul(t0, t1, t0);                             // z^(2^250 - 1)
    fe_sq(t0, t0); fe_sq(t0, t0);
    fe_mul(out, t0, z);                             // z^(2^252 - 3)
}

// out = 1/z, by Fermat's little theorem: z^(p-2) = z^(2^255 - 21).
//
// Note 1/0 evaluates to 0 in this construction, which is what Montgomery's
// batch trick relies on never happening — a zero Z coordinate would mean a
// point at infinity, which the incremental walk cannot produce.
__device__ __noinline__ void fe_invert(fe out, const fe z) {
    fe t0, t1, t2, t3;
    int i;

    fe_sq(t0, z);                                   // z^2
    fe_sq(t1, t0); fe_sq(t1, t1);                   // z^8
    fe_mul(t1, z, t1);                              // z^9
    fe_mul(t0, t0, t1);                             // z^11
    fe_sq(t2, t0);                                  // z^22
    fe_mul(t1, t1, t2);                             // z^(2^5 - 1)
    fe_sq(t2, t1); for (i = 1; i < 5; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);                             // z^(2^10 - 1)
    fe_sq(t2, t1); for (i = 1; i < 10; i++) fe_sq(t2, t2);
    fe_mul(t2, t2, t1);                             // z^(2^20 - 1)
    fe_sq(t3, t2); for (i = 1; i < 20; i++) fe_sq(t3, t3);
    fe_mul(t2, t3, t2);                             // z^(2^40 - 1)
    fe_sq(t2, t2); for (i = 1; i < 10; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);                             // z^(2^50 - 1)
    fe_sq(t2, t1); for (i = 1; i < 50; i++) fe_sq(t2, t2);
    fe_mul(t2, t2, t1);                             // z^(2^100 - 1)
    fe_sq(t3, t2); for (i = 1; i < 100; i++) fe_sq(t3, t3);
    fe_mul(t2, t3, t2);                             // z^(2^200 - 1)
    fe_sq(t2, t2); for (i = 1; i < 50; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);                             // z^(2^250 - 1)
    fe_sq(t1, t1); for (i = 1; i < 5; i++) fe_sq(t1, t1);
    fe_mul(out, t1, t0);                            // z^(2^255 - 21) = z^(p-2)
}
