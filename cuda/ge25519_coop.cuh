// Cooperative group arithmetic on Ed25519, paralleling ge25519.cuh.
//
// Each group coordinate is a single u32 register (one limb per thread, 8
// threads per cooperative group). All fe_*_coop calls involve shuffles, so
// every lane in the group must participate even when only one lane uses the
// result.

#pragma once

#include "fe25519_coop.cuh"
#include "ge25519.cuh"

// r = p + q (mixed addition, q in niels form).
__device__ __forceinline__ void ge_madd_coop(
    u32 &rX, u32 &rY, u32 &rZ, u32 &rT,
    u32 pX, u32 pY, u32 pZ, u32 pT,
    u32 qypx, u32 qymx, u32 qxy2d
) {
    u32 t0;
    fe_add_coop(rX, pY, pX);
    fe_sub_coop(rY, pY, pX);
    fe_mul_coop(rZ, rX, qypx);
    fe_mul_coop(rY, rY, qymx);
    fe_mul_coop(rT, qxy2d, pT);
    fe_add_coop(t0, pZ, pZ);
    fe_sub_coop(rX, rZ, rY);
    fe_add_coop(rY, rZ, rY);
    fe_add_coop(rZ, t0, rT);
    fe_sub_coop(rT, t0, rT);
}

// p1p1 → extended coordinates.
__device__ __forceinline__ void ge_p1p1_to_p3_coop(
    u32 &rX, u32 &rY, u32 &rZ, u32 &rT,
    u32 pX, u32 pY, u32 pZ, u32 pT
) {
    u32 oX, oY, oZ, oT;
    fe_mul_coop(oX, pX, pT);
    fe_mul_coop(oY, pY, pZ);
    fe_mul_coop(oZ, pZ, pT);
    fe_mul_coop(oT, pX, pY);
    rX = oX; rY = oY; rZ = oZ; rT = oT;
}

// p += 8*B, in place.
__device__ __forceinline__ void ge_add_8b_coop(u32 &X, u32 &Y, u32 &Z, u32 &T) {
    u32 qypx  = fe_scatter(ge_8b_yplusx);
    u32 qymx  = fe_scatter(ge_8b_yminusx);
    u32 qxy2d = fe_scatter(ge_8b_xy2d);

    u32 tX, tY, tZ, tT;
    ge_madd_coop(tX, tY, tZ, tT, X, Y, Z, T, qypx, qymx, qxy2d);
    ge_p1p1_to_p3_coop(X, Y, Z, T, tX, tY, tZ, tT);
}

// Dual addition law: y-fractions for base±off, 2 multiplies + 4 adds/subs.
__device__ __forceinline__ void ge_dual_pair_coop(
    u32 &pnum, u32 &pden, u32 &mnum, u32 &mden,
    u32 bx, u32 by, u32 bxy,
    u32 ox, u32 oy, u32 oxy
) {
    u32 x1y2, y1x2;
    fe_mul_coop(x1y2, bx, oy);
    fe_mul_coop(y1x2, by, ox);
    fe_sub_coop(pnum, bxy, oxy);
    fe_sub_coop(pden, x1y2, y1x2);
    fe_add_coop(mnum, bxy, oxy);
    fe_add_coop(mden, x1y2, y1x2);
}

// Affine form from extended coordinates, given zinv = 1/Z.
__device__ __forceinline__ void ge_p3_to_affine_coop(
    u32 &ax, u32 &ay, u32 &axy,
    u32 pX, u32 pY, u32 zinv
) {
    fe_mul_coop(ax, pX, zinv);
    fe_mul_coop(ay, pY, zinv);
    fe_mul_coop(axy, ax, ay);
}

// Decompress a 32-byte public key into cooperative extended coordinates.
// Returns 1 on success (same value on all lanes), 0 on failure.
__device__ __noinline__ u32 ge_frombytes_coop(
    u32 &X, u32 &Y, u32 &Z, u32 &T, const u8 *s
) {
    u32 u, v, v3, vxx, check;

    Y = fe_frombytes_coop(s);
    fe_1_coop(Z);
    fe_sq_coop(u, Y);
    u32 d = fe_scatter(fe_d);
    fe_mul_coop(v, u, d);
    fe_sub_coop(u, u, Z);   // u = y^2 - 1
    fe_add_coop(v, v, Z);   // v = d*y^2 + 1

    fe_sq_coop(v3, v);
    fe_mul_coop(v3, v3, v);    // v^3
    fe_sq_coop(X, v3);
    fe_mul_coop(X, X, v);
    fe_mul_coop(X, X, u);      // u * v^7

    fe_pow22523_coop(X, X);
    fe_mul_coop(X, X, v3);
    fe_mul_coop(X, X, u);      // u * v^3 * (u * v^7)^((p-5)/8)

    fe_sq_coop(vxx, X);
    fe_mul_coop(vxx, vxx, v);
    fe_sub_coop(check, vxx, u);
    if (fe_isnonzero_coop(check)) {
        fe_add_coop(check, vxx, u);
        if (fe_isnonzero_coop(check)) {
            return 0;
        }
        u32 sqrtm1 = fe_scatter(fe_sqrtm1);
        fe_mul_coop(X, X, sqrtm1);
    }

    u32 sign_bit = (u32)(s[31] >> 7);
    if (fe_isnegative_coop(X) != sign_bit) {
        fe_neg_coop(X, X);
    }
    fe_mul_coop(T, X, Y);
    return 1;
}

// Niels-form conversion for an affine point.
__device__ __forceinline__ void ge_affine_to_precomp_coop(
    u32 &ypx, u32 &ymx, u32 &xy2d,
    u32 x, u32 y
) {
    fe_add_coop(ypx, y, x);
    fe_sub_coop(ymx, y, x);
    u32 t;
    fe_mul_coop(t, x, y);
    u32 d = fe_scatter(fe_d);
    fe_mul_coop(t, t, d);
    fe_add_coop(xy2d, t, t);
}
