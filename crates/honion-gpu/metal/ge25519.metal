// Group arithmetic on the Ed25519 curve, for Metal.
//
// A port of `cuda/ge25519.cuh`: the incremental +8B walk, point (de)compression,
// and the dual addition law that yields two candidate y-coordinates for the
// price of two multiplications. The mathematics and the constants are identical
// to the CUDA version; the port translates the CUDA field's pointer-mutation
// calls (`fe_add(r->X, ...)`) into this field's value-returning form
// (`r.X = fe_add(...)`), which is what Metal's address spaces make natural.
//
// See docs/01-ed25519-vanity-search.md for the walk, and the dual-law block
// below for why only y is produced.

#ifndef HONION_GE25519
#define HONION_GE25519

#include "fe25519.metal"

using namespace metal;

// Extended coordinates: x = X/Z, y = Y/Z, T = XY/Z.
struct ge_p3 { fe X, Y, Z, T; };

// The output of a mixed addition: cheaper to produce than a p3.
struct ge_p1p1 { fe X, Y, Z, T; };

// An affine point in "niels" form, the precomputed second operand of a mixed
// addition.
struct ge_precomp { fe yplusx, yminusx, xy2d; };

// An affine point with its x*y product precomputed (for the dual law).
struct ge_affine { fe x, y, xy; };

// Numerator and denominator of a candidate's y coordinate.
struct ge_yfrac { fe num, den; };

// The result of decompression: whether the bytes named a point, and the point.
struct ge_decode { u32 ok; ge_p3 p; };

// GENERATED constants (radix-25.5), from cuda/gen_constants.py.
// 8*B compresses to b4b937fca95b2f1e93e41e62fc3c78818ff38a66096fad6e7973e5c90006d321.

// d = -121665/121666, the Edwards curve parameter.
constant fe fe_d = {{56195235, 13857412, 51736253, 6949390, 114729, 24766616, 60832955, 30306712, 48412415, 21499315}};

// sqrt(-1), for choosing the decompression branch.
constant fe fe_sqrtm1 = {{34513072, 25610706, 9377949, 3500415, 12389472, 33281959, 41962654, 31548777, 326685, 11406482}};

// 8*B in affine form; the offset table is built by walking from here.
constant fe ge_8b_x = {{10847432, 33517314, 24323952, 31195980, 61664025, 5164947, 51818108, 3590224, 33127799, 27069317}};
constant fe ge_8b_y = {{3652020, 30861951, 9593797, 31658231, 33939699, 9106319, 45581491, 7286229, 826967, 8866840}};

// 8*B in niels form.
constant fe ge_8b_yplusx = {{14499471, 30824833, 33917750, 29299779, 28494861, 14271267, 30290735, 10876454, 33954766, 2381725}};
constant fe ge_8b_yminusx = {{59913433, 30899068, 52378708, 462250, 39384538, 3941371, 60872247, 3696004, 34808032, 15351954}};
constant fe ge_8b_xy2d = {{27431194, 8222322, 16448760, 29646437, 48401861, 11938354, 34147463, 30583916, 29551812, 10109425}};

// r = p + q, q affine in niels form. The "madd" formula for extended twisted
// Edwards coordinates with a = -1. Four multiplications here, four in
// ge_p1p1_to_p3: eight in total.
inline ge_p1p1 ge_madd(ge_p3 p, ge_precomp q) {
    ge_p1p1 r;
    fe t0;
    r.X = fe_add(p.Y, p.X);        // A = Y1 + X1
    r.Y = fe_sub(p.Y, p.X);        // B = Y1 - X1
    r.Z = fe_mul(r.X, q.yplusx);   // C = A * (y2 + x2)
    r.Y = fe_mul(r.Y, q.yminusx);  // D = B * (y2 - x2)
    r.T = fe_mul(q.xy2d, p.T);     // E = T1 * 2*d*x2*y2
    t0  = fe_add(p.Z, p.Z);        // F = 2*Z1
    fe C = r.Z, D = r.Y;
    r.X = fe_sub(C, D);            // X3 = C - D
    r.Y = fe_add(C, D);            // Y3 = C + D
    r.Z = fe_add(t0, r.T);         // Z3 = F + E
    r.T = fe_sub(t0, r.T);         // T3 = F - E
    return r;
}

// Convert a p1p1 back to extended coordinates. Four multiplications.
inline ge_p3 ge_p1p1_to_p3(ge_p1p1 p) {
    ge_p3 r;
    r.X = fe_mul(p.X, p.T);
    r.Y = fe_mul(p.Y, p.Z);
    r.Z = fe_mul(p.Z, p.T);
    r.T = fe_mul(p.X, p.Y);
    return r;
}

// The niels form of 8*B.
inline ge_precomp ge_precomp_8b() {
    ge_precomp q;
    q.yplusx = ge_8b_yplusx;
    q.yminusx = ge_8b_yminusx;
    q.xy2d = ge_8b_xy2d;
    return q;
}

// p += 8*B.
inline ge_p3 ge_add_8b(ge_p3 p) {
    return ge_p1p1_to_p3(ge_madd(p, ge_precomp_8b()));
}

// Compress a point to the 32-byte Ed25519 public-key encoding: affine y,
// little-endian, with the low bit of affine x in bit 255. Costs a modular
// inversion; noinline for the same reason as the CUDA version.
[[clang::noinline]] void ge_p3_tobytes(device u8 *s, ge_p3 p) {
    fe recip = fe_invert(p.Z);
    fe x = fe_mul(p.X, recip);
    fe y = fe_mul(p.Y, recip);
    fe_tobytes(s, y);
    s[31] ^= (u8)(fe_isnegative(x) << 7);
}

// Decompress a 32-byte public key into extended coordinates. ok=0 when the
// encoding names no curve point. The caller must check: a thread walking from a
// bad start point reports nothing, indistinguishable from bad luck, so failure
// here is surfaced as a decode failure rather than absorbed.
[[clang::noinline]] ge_decode ge_frombytes(const device u8 *s) {
    ge_decode out;
    ge_p3 h;
    fe u, v, v3, vxx, check;

    h.Y = fe_frombytes(s);
    h.Z = fe_one();
    u = fe_sq(h.Y);
    v = fe_mul(u, fe_d);
    u = fe_sub(u, h.Z);   // u = y^2 - 1
    v = fe_add(v, h.Z);   // v = d*y^2 + 1

    v3 = fe_sq(v);
    v3 = fe_mul(v3, v);   // v^3
    h.X = fe_sq(v3);
    h.X = fe_mul(h.X, v);
    h.X = fe_mul(h.X, u); // u * v^7

    h.X = fe_pow22523(h.X);
    h.X = fe_mul(h.X, v3);
    h.X = fe_mul(h.X, u); // x = u * v^3 * (u * v^7)^((p-5)/8)

    vxx = fe_sq(h.X);
    vxx = fe_mul(vxx, v);
    check = fe_sub(vxx, u); // v*x^2 - u
    if (fe_isnonzero(check)) {
        check = fe_add(vxx, u); // v*x^2 + u
        if (fe_isnonzero(check)) {
            out.ok = 0;
            out.p = h;
            return out;     // u/v is not a square: not a point
        }
        h.X = fe_mul(h.X, fe_sqrtm1);
    }

    if (fe_isnegative(h.X) != (u32)(s[31] >> 7)) {
        h.X = fe_neg(h.X);
    }
    h.T = fe_mul(h.X, h.Y);
    out.ok = 1;
    out.p = h;
    return out;
}

// Whether a point satisfies -x^2 + y^2 = 1 + d*x^2*y^2. Returns 1/0.
[[clang::noinline]] u32 ge_p3_is_on_curve(ge_p3 p) {
    fe recip = fe_invert(p.Z);
    fe x = fe_mul(p.X, recip);
    fe y = fe_mul(p.Y, recip);
    fe x2 = fe_sq(x);
    fe y2 = fe_sq(y);
    fe lhs = fe_sub(y2, x2);            // y^2 - x^2
    fe t = fe_mul(x2, y2);
    fe rhs = fe_mul(t, fe_d);
    fe one = fe_one();
    rhs = fe_add(rhs, one);            // 1 + d*x^2*y^2
    t = fe_sub(lhs, rhs);
    return fe_isnonzero(t) ? 0u : 1u;
}

// ---------------------------------------------------------------------------
// The dual addition law
// ---------------------------------------------------------------------------
//
// For a = -1, y(P±Q) can be written without the curve constant d and without
// projective coordinates for the result:
//
//     y(P + Q) = (x1*y1 - x2*y2) / (x1*y2 - y1*x2)
//     y(P - Q) = (x1*y1 + x2*y2) / (x1*y2 + y1*x2)
//
// (Verified against the standard law in cuda/verify_dual_law.py.) Only y is
// produced — all an address needs — and the two results share x1*y2 and y1*x2,
// differing only in a + versus a -: two candidates for two multiplications. The
// results are fractions; the single batch inversion divides them all at once.

// The y coordinates of base+off and base-off, as unreduced fractions. Two
// multiplications for both.
inline void ge_dual_pair(thread ge_yfrac &plus, thread ge_yfrac &minus,
                         ge_affine base, fe off_x, fe off_y, fe off_xy) {
    fe x1y2 = fe_mul(base.x, off_y);   // x1 * y2
    fe y1x2 = fe_mul(base.y, off_x);   // y1 * x2

    plus.num  = fe_sub(base.xy, off_xy);  // x1*y1 - x2*y2
    plus.den  = fe_sub(x1y2, y1x2);       // x1*y2 - y1*x2
    minus.num = fe_add(base.xy, off_xy);  // x1*y1 + x2*y2
    minus.den = fe_add(x1y2, y1x2);       // x1*y2 + y1*x2
}

// Affine form of an extended point, given 1/Z.
inline ge_affine ge_p3_to_affine(ge_p3 p, fe zinv) {
    ge_affine a;
    a.x = fe_mul(p.X, zinv);
    a.y = fe_mul(p.Y, zinv);
    a.xy = fe_mul(a.x, a.y);
    return a;
}

// 8*B as an extended-coordinate point.
inline ge_p3 ge_p3_8b() {
    ge_p3 p;
    p.X = ge_8b_x;
    p.Y = ge_8b_y;
    p.Z = fe_one();
    p.T = fe_mul(ge_8b_x, ge_8b_y);
    return p;
}

// Affine point to niels form.
inline ge_precomp ge_affine_to_precomp(fe x, fe y) {
    ge_precomp q;
    q.yplusx = fe_add(y, x);
    q.yminusx = fe_sub(y, x);
    fe t = fe_mul(x, y);
    t = fe_mul(t, fe_d);
    q.xy2d = fe_add(t, t);   // 2*d*x*y
    return q;
}

#endif
