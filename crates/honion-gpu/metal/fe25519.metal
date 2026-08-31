// Arithmetic in the field GF(2^255 - 19), for Metal.
//
// A direct port of `cuda/fe25519.cuh`: same radix-25.5 representation, same
// algorithms, same overflow bounds. The only differences are the ones the
// language forces — Metal has address spaces, so a field element is a `struct`
// value living in `thread` memory and passed by value rather than a bare
// `int[10]` passed by pointer, and byte I/O names the `device` address space of
// its buffer explicitly. The mathematics is unchanged, and the differential
// test drives this file against the same `num-bigint` reference the CUDA field
// is checked against, demanding exact agreement.
//
// ## Representation
//
// Ten signed 32-bit limbs, alternating radix 2^26 and 2^25 ("radix 25.5"), the
// ref10 layout:
//
//     h = v[0]
//       + v[1]*2^26  + v[2]*2^51  + v[3]*2^77  + v[4]*2^102
//       + v[5]*2^128 + v[6]*2^153 + v[7]*2^179 + v[8]*2^204
//       + v[9]*2^230
//
// Even limbs carry 26 bits, odd limbs 25. Limbs are signed and the
// representation is redundant; add/sub leave limbs un-normalised and
// normalisation happens inside multiplication and in `fe_tobytes`.

#ifndef HONION_FE25519
#define HONION_FE25519

#include <metal_stdlib>
using namespace metal;

typedef uchar u8;
typedef int   i32;
typedef uint  u32;
typedef long  i64;
typedef ulong u64;

constant int FE_LIMBS = 10;

// A field element. A value type in `thread` space; passed and returned by value.
struct fe {
    i32 v[FE_LIMBS];
};

// Bit width of limb `i`: 26 for even, 25 for odd.
inline constexpr int fe_limb_bits(int i) { return (i & 1) ? 25 : 26; }

inline fe fe_zero() {
    fe h;
    for (int i = 0; i < FE_LIMBS; i++) h.v[i] = 0;
    return h;
}

inline fe fe_one() {
    fe h = fe_zero();
    h.v[0] = 1;
    return h;
}

// h = f + g. Limbs left un-normalised.
inline fe fe_add(fe f, fe g) {
    fe h;
    for (int i = 0; i < FE_LIMBS; i++) h.v[i] = f.v[i] + g.v[i];
    return h;
}

// h = f - g. Likewise un-normalised.
inline fe fe_sub(fe f, fe g) {
    fe h;
    for (int i = 0; i < FE_LIMBS; i++) h.v[i] = f.v[i] - g.v[i];
    return h;
}

// h = -f.
inline fe fe_neg(fe f) {
    fe h;
    for (int i = 0; i < FE_LIMBS; i++) h.v[i] = -f.v[i];
    return h;
}

// Propagate carries from a 64-bit accumulator array into normalised limbs.
//
// Two interleaved chains (even limbs, then odd) keep the dependency graph two
// short chains rather than one ten-deep one. The final wrap folds limb 9's
// overflow into limb 0 times 19, because 2^255 ≡ 19 (mod p). Rounding form
// `(x + 2^(w-1)) >> w` keeps limbs centred on zero for headroom.
inline fe fe_carry(thread i64 *t) {
#define CARRY(i, w, next)                                          \
    do {                                                           \
        i64 c = (t[i] + ((i64)1 << ((w) - 1))) >> (w);             \
        t[next] += c;                                              \
        t[i] -= c << (w);                                          \
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

    {
        i64 c = (t[9] + ((i64)1 << 24)) >> 25;
        t[0] += c * 19;
        t[9] -= c << 25;
    }
    CARRY(0, 26, 1);
#undef CARRY

    fe h;
    for (int i = 0; i < FE_LIMBS; i++) h.v[i] = (i32)t[i];
    return h;
}

// h = f * g.
//
// term(i,j) folds by 2 when both i,j odd (radix parity) and by 19 when
// i+j >= 10 (the 2^255 ≡ 19 wrap). Both conditions depend only on loop indices,
// so the compiler resolves them when the fixed loops unroll. Overflow bound: with
// input limbs up to 1.1*2^27, the widest accumulator stays under 2^62.4, inside
// a signed 64-bit accumulator.
inline fe fe_mul(fe f, fe g) {
    i32 g19[FE_LIMBS];
    i32 f2[FE_LIMBS];
    for (int i = 0; i < FE_LIMBS; i++) {
        g19[i] = 19 * g.v[i];
        f2[i] = 2 * f.v[i];
    }

    i64 t[FE_LIMBS];
    for (int i = 0; i < FE_LIMBS; i++) t[i] = 0;

    for (int i = 0; i < FE_LIMBS; i++) {
        for (int j = 0; j < FE_LIMBS; j++) {
            const int k = i + j;
            const bool wraps = (k >= FE_LIMBS);
            const bool doubled = (i & 1) && (j & 1);
            const i32 lhs = doubled ? f2[i] : f.v[i];
            const i32 rhs = wraps ? g19[j] : g.v[j];
            t[wraps ? k - FE_LIMBS : k] += (i64)lhs * (i64)rhs;
        }
    }
    return fe_carry(t);
}

// h = f * f. Kept as a plain multiply for checkability; see the CUDA note.
inline fe fe_sq(fe f) { return fe_mul(f, f); }

inline i64 fe_load3(const device u8 *s) {
    return (i64)s[0] | ((i64)s[1] << 8) | ((i64)s[2] << 16);
}

inline i64 fe_load4(const device u8 *s) {
    return (i64)s[0] | ((i64)s[1] << 8) | ((i64)s[2] << 16) | ((i64)s[3] << 24);
}

// Load a field element from 32 little-endian bytes; bit 255 is masked here.
inline fe fe_frombytes(const device u8 *s) {
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
    t[9] = (fe_load3(s + 29) & 8388607) << 2;
    return fe_carry(t);
}

// Reduce to the canonical representative as non-negative limbs, into t[10].
//
// Total by construction: it carries its input first, so no caller can violate a
// normalisation precondition and get an answer wrong by exactly 19.
inline void fe_freeze(thread i32 *t, fe h) {
    i64 acc[FE_LIMBS];
    for (int i = 0; i < FE_LIMBS; i++) acc[i] = h.v[i];
    fe carried = fe_carry(acc);
    for (int i = 0; i < FE_LIMBS; i++) t[i] = carried.v[i];

    // q = number of times p must be subtracted, from simulating h + 19's carry.
    i32 q = (19 * t[9] + (((i32)1) << 24)) >> 25;
    for (int i = 0; i < FE_LIMBS; i++) q = (t[i] + q) >> fe_limb_bits(i);

    t[0] += 19 * q;
    for (int i = 0; i < FE_LIMBS - 1; i++) {
        i32 c = t[i] >> fe_limb_bits(i);
        t[i + 1] += c;
        t[i] -= c << fe_limb_bits(i);
    }
    t[9] &= 0x1ffffff;
}

// The first eight bytes of the canonical encoding, read big-endian — the
// prefilter probe. Must agree exactly with honion_core::pattern::key_prefix_u64.
inline u64 fe_prefix_be64(fe h) {
    i32 t[FE_LIMBS];
    fe_freeze(t, h);

    const u64 lo = (u64)(u32)t[0] | ((u64)(u32)t[1] << 26) | ((u64)(u32)t[2] << 51);

    return ((lo & 0x00000000000000ffUL) << 56) | ((lo & 0x000000000000ff00UL) << 40)
         | ((lo & 0x0000000000ff0000UL) << 24) | ((lo & 0x00000000ff000000UL) << 8)
         | ((lo & 0x000000ff00000000UL) >> 8)  | ((lo & 0x0000ff0000000000UL) >> 24)
         | ((lo & 0x00ff000000000000UL) >> 40) | ((lo & 0xff00000000000000UL) >> 56);
}

// Store a field element as 32 little-endian bytes, fully reduced mod p. Total,
// for the same reason as fe_freeze.
inline void fe_tobytes(device u8 *s, fe h) {
    i32 t[FE_LIMBS];
    fe_freeze(t, h);

    for (int i = 0; i < 32; i++) s[i] = 0;
    int bit = 0;
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

// Store a field element as 32 little-endian canonical bytes, into a thread
// buffer. The thread-space counterpart of fe_tobytes, needed where the bytes
// are consumed on-device (the search kernel's residual check) rather than
// written out to a device buffer.
inline void fe_tobytes_thread(thread u8 *s, fe h) {
    i32 t[FE_LIMBS];
    fe_freeze(t, h);
    for (int i = 0; i < 32; i++) s[i] = 0;
    int bit = 0;
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

// Whether f is zero modulo p. Returns 1 or 0. Writes into a thread scratch
// buffer so it needs no device pointer.
inline u32 fe_isnonzero(fe h) {
    i32 t[FE_LIMBS];
    fe_freeze(t, h);
    // Repack into bytes in thread space and OR them.
    u8 acc = 0;
    int bit = 0;
    u8 s[32];
    for (int i = 0; i < 32; i++) s[i] = 0;
    for (int i = 0; i < FE_LIMBS; i++) {
        const int w = fe_limb_bits(i);
        u32 v = (u32)t[i];
        for (int b = 0; b < w; b++) {
            const int abs_bit = bit + b;
            s[abs_bit >> 3] |= (u8)(((v >> b) & 1u) << (abs_bit & 7));
        }
        bit += w;
    }
    for (int i = 0; i < 32; i++) acc |= s[i];
    return acc != 0;
}

// The least significant bit of the canonical representative: the compressed
// point's x-sign bit.
inline u32 fe_isnegative(fe h) {
    i32 t[FE_LIMBS];
    fe_freeze(t, h);
    // Bit 0 is limb 0's low bit after freezing.
    return (u32)(t[0] & 1);
}

// out = z^(2^252 - 3). Shared skeleton for inversion and square roots.
//
// [[clang::noinline]] deliberately (DEC-METAL-006): on the CUDA side inlining
// this 261-operation routine multiplied the kernel's PTX ~50x, and it runs only
// once per batch. The equivalent Metal trap is rediscovered and neutralised
// here; the effect is confirmed when the search kernel that calls it is built
// in the search kernel, where the batch loop makes any inlining blow-up
// observable.
[[clang::noinline]] fe fe_pow22523(fe z) {
    fe t0, t1, t2;
    int i;

    t0 = fe_sq(z);                                  // z^2
    t1 = fe_sq(t0); t1 = fe_sq(t1);                 // z^8
    t1 = fe_mul(z, t1);                             // z^9
    t0 = fe_mul(t0, t1);                            // z^11
    t0 = fe_sq(t0);                                 // z^22
    t0 = fe_mul(t1, t0);                            // z^(2^5 - 1)
    t1 = fe_sq(t0); for (i = 1; i < 5; i++) t1 = fe_sq(t1);
    t0 = fe_mul(t1, t0);                            // z^(2^10 - 1)
    t1 = fe_sq(t0); for (i = 1; i < 10; i++) t1 = fe_sq(t1);
    t1 = fe_mul(t1, t0);                            // z^(2^20 - 1)
    t2 = fe_sq(t1); for (i = 1; i < 20; i++) t2 = fe_sq(t2);
    t1 = fe_mul(t2, t1);                            // z^(2^40 - 1)
    t1 = fe_sq(t1); for (i = 1; i < 10; i++) t1 = fe_sq(t1);
    t0 = fe_mul(t1, t0);                            // z^(2^50 - 1)
    t1 = fe_sq(t0); for (i = 1; i < 50; i++) t1 = fe_sq(t1);
    t1 = fe_mul(t1, t0);                            // z^(2^100 - 1)
    t2 = fe_sq(t1); for (i = 1; i < 100; i++) t2 = fe_sq(t2);
    t1 = fe_mul(t2, t1);                            // z^(2^200 - 1)
    t1 = fe_sq(t1); for (i = 1; i < 50; i++) t1 = fe_sq(t1);
    t0 = fe_mul(t1, t0);                            // z^(2^250 - 1)
    t0 = fe_sq(t0); t0 = fe_sq(t0);
    return fe_mul(t0, z);                           // z^(2^252 - 3)
}

// out = 1/z, by Fermat: z^(p-2) = z^(2^255 - 21). 1/0 = 0 in this construction.
[[clang::noinline]] fe fe_invert(fe z) {
    fe t0, t1, t2, t3;
    int i;

    t0 = fe_sq(z);                                  // z^2
    t1 = fe_sq(t0); t1 = fe_sq(t1);                 // z^8
    t1 = fe_mul(z, t1);                             // z^9
    t0 = fe_mul(t0, t1);                            // z^11
    t2 = fe_sq(t0);                                 // z^22
    t1 = fe_mul(t1, t2);                            // z^(2^5 - 1)
    t2 = fe_sq(t1); for (i = 1; i < 5; i++) t2 = fe_sq(t2);
    t1 = fe_mul(t2, t1);                            // z^(2^10 - 1)
    t2 = fe_sq(t1); for (i = 1; i < 10; i++) t2 = fe_sq(t2);
    t2 = fe_mul(t2, t1);                            // z^(2^20 - 1)
    t3 = fe_sq(t2); for (i = 1; i < 20; i++) t3 = fe_sq(t3);
    t2 = fe_mul(t3, t2);                            // z^(2^40 - 1)
    t2 = fe_sq(t2); for (i = 1; i < 10; i++) t2 = fe_sq(t2);
    t1 = fe_mul(t2, t1);                            // z^(2^50 - 1)
    t2 = fe_sq(t1); for (i = 1; i < 50; i++) t2 = fe_sq(t2);
    t2 = fe_mul(t2, t1);                            // z^(2^100 - 1)
    t3 = fe_sq(t2); for (i = 1; i < 100; i++) t3 = fe_sq(t3);
    t2 = fe_mul(t3, t2);                            // z^(2^200 - 1)
    t2 = fe_sq(t2); for (i = 1; i < 50; i++) t2 = fe_sq(t2);
    t1 = fe_mul(t2, t1);                            // z^(2^250 - 1)
    t1 = fe_sq(t1); for (i = 1; i < 5; i++) t1 = fe_sq(t1);
    return fe_mul(t1, t0);                          // z^(2^255 - 21) = z^(p-2)
}

#endif
