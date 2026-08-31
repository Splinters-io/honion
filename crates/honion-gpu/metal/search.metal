// The vanity search kernel, for Metal.
//
// A port of `cuda/search.cu`. The algorithm is identical — the incremental
// +8B walk, the dual addition law giving two candidates per pair, and one
// batched modular inversion via Montgomery's trick — and the host builds every
// table it reads, so the device parses nothing (langsec rule 4). The port
// tracks the CUDA source closely; the differences are noted where they occur.
//
// ## Two deliberate deviations from the CUDA kernel
//
// 1. The offset table is read straight from device memory rather than staged in
//    threadgroup memory. On the CUDA side the table lives in __shared__ memory;
//    at the useful HALF=512 it is ~60 KB, which exceeds an Apple GPU's
//    threadgroup allocation (~32 KB), so staging it is not possible at the sizes
//    that matter. Staging is a bandwidth optimisation, not a correctness
//    requirement — every thread reads the same immutable table — so it is
//    dropped here and revisited as tuning work. Correctness is
//    unaffected and the set-equality test proves it.
//
// 2. The batch and HALF loops carry `#pragma clang loop unroll(disable)`, the
//    Metal analog of the CUDA `#pragma unroll 1`. HALF is a compile-time
//    constant (for array sizing), so without this the compiler could fully
//    unroll a 512-iteration loop of hundred-term multiplies — the second of the
//    two documented compiler traps (DEC-METAL-006). The effect is verified by
//    the search kernel's compile time and code size staying bounded across HALF.

#include "ge25519.metal"

using namespace metal;

#ifndef HALF
#define HALF 128
#endif

#define CANDS_PER_BATCH (2 * HALF + 1)
#define OFF_STRIDE (3 * FE_LIMBS)

#define STATUS_BAD_START_POINT 1u
#define STATUS_HIT_OVERFLOW    2u
#define STATUS_SINGULAR        4u

// A reported candidate. Layout matches honion_gpu::search::Hit (repr(C)).
struct Hit {
    u32 thread_id;
    int offset;      // signed: a match may lie below the starting scalar
    u32 pattern_id;
    u32 reserved;
};

// The 5-bit value base32 character `index` encodes, read from a raw key.
// Must agree exactly with honion_core::pattern::char_value.
inline u32 key_char_value(const thread u8 *key, u32 index) {
    const u32 bit = index * 5u;
    const u32 byte = bit >> 3;
    const u32 off = bit & 7u;
    const u32 b0 = key[byte];
    const u32 b1 = (byte + 1u < 32u) ? key[byte + 1u] : 0u;
    return (((b0 << 8) | b1) >> (11u - off)) & 0x1fu;
}

// Whether `key` satisfies pattern `pid`'s residual constraints (the parts the
// 64-bit prefilter cannot express). Most patterns have none.
inline bool residuals_hold(const thread u8 *key, u32 pid,
                           const device u32 *res_off, const device u64 *res) {
    const u32 end = res_off[pid + 1];
    for (u32 i = res_off[pid]; i < end; i++) {
        const u64 entry = res[i];
        if (((((u32)entry) >> key_char_value(key, (u32)(entry >> 32))) & 1u) == 0u) {
            return false;
        }
    }
    return true;
}

// Build the offset table: affine (x, y, x*y) for 1*8B .. HALF*8B, plus the
// giant step (2*HALF+1)*8B in niels form. One thread, once per search.
kernel void honion_build_offsets(device u32 *table          [[buffer(0)]],
                                 device u32 *giant          [[buffer(1)]],
                                 device atomic_uint *status [[buffer(2)]],
                                 uint gid                   [[thread_position_in_grid]]) {
    if (gid != 0) return;

    ge_precomp step = ge_precomp_8b();
    ge_p3 p = ge_p3_8b();   // p = 1 * 8B

    for (u32 j = 0; j < HALF; j++) {
        if (!fe_isnonzero(p.Z)) {
            atomic_fetch_or_explicit(status, STATUS_SINGULAR, memory_order_relaxed);
            return;
        }
        fe zinv = fe_invert(p.Z);
        ge_affine a = ge_p3_to_affine(p, zinv);

        device u32 *slot = table + (ulong)j * OFF_STRIDE;
        for (int k = 0; k < FE_LIMBS; k++) {
            slot[k] = (u32)a.x.v[k];
            slot[FE_LIMBS + k] = (u32)a.y.v[k];
            slot[2 * FE_LIMBS + k] = (u32)a.xy.v[k];
        }

        p = ge_p1p1_to_p3(ge_madd(p, step));   // p = (j+2) * 8B
    }

    for (u32 j = HALF + 1; j < CANDS_PER_BATCH; j++) {
        p = ge_p1p1_to_p3(ge_madd(p, step));
    }

    if (!fe_isnonzero(p.Z)) {
        atomic_fetch_or_explicit(status, STATUS_SINGULAR, memory_order_relaxed);
        return;
    }
    fe zinv = fe_invert(p.Z);
    fe gx = fe_mul(p.X, zinv);
    fe gy = fe_mul(p.Y, zinv);
    ge_precomp g = ge_affine_to_precomp(gx, gy);
    for (int k = 0; k < FE_LIMBS; k++) {
        giant[k] = (u32)g.yplusx.v[k];
        giant[FE_LIMBS + k] = (u32)g.yminusx.v[k];
        giant[2 * FE_LIMBS + k] = (u32)g.xy2d.v[k];
    }
}

// Test one candidate's y against the pattern tables, recording a hit if it
// matches. A macro so the batched path and the base-point path share it without
// a function call in the inner loop.
#define HONION_CHECK(Y, OFFSET)                                                 \
    do {                                                                        \
        const u64 probe_base = fe_prefix_be64(Y);                              \
        for (u32 gg = 0; gg < num_groups; gg++) {                              \
            const u64 probe = probe_base & group_mask[gg];                     \
            u32 lo = group_off[gg];                                            \
            u32 hi = group_off[gg + 1];                                        \
            while (lo < hi) {                                                  \
                const u32 mid = lo + ((hi - lo) >> 1);                         \
                if (target[mid] < probe) lo = mid + 1; else hi = mid;          \
            }                                                                  \
            u8 key[32];                                                        \
            bool key_ready = false;                                            \
            for (u32 tt = lo; tt < group_off[gg + 1] && target[tt] == probe; tt++) { \
                const u32 pid = target_pat[tt];                                \
                if (res_off[pid + 1] != res_off[pid]) {                        \
                    if (!key_ready) { fe_tobytes_thread(key, Y); key_ready = true; } \
                    if (!residuals_hold(key, pid, res_off, res)) continue;     \
                }                                                              \
                const u32 slot = atomic_fetch_add_explicit(hit_count, 1u, memory_order_relaxed); \
                if (slot < max_hits) {                                         \
                    hits[slot].thread_id = tid;                                \
                    hits[slot].offset = (OFFSET);                              \
                    hits[slot].pattern_id = pid;                               \
                    hits[slot].reserved = 0;                                   \
                } else {                                                       \
                    atomic_fetch_or_explicit(status, STATUS_HIT_OVERFLOW, memory_order_relaxed); \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

// The search. All tables are host-built (langsec rule 4).
[[max_total_threads_per_threadgroup(256)]]
// `walk_points` is read-write and is the *walk* buffer, not the caller's
// starting points: the host copies the starting points into it before the first
// dispatch of a launch. A launch is split across several dispatches so that no
// single command buffer runs long enough for macOS's GPU watchdog to kill it
// (DEC-METAL-009), and each dispatch leaves the point its thread reached here
// for the next one to resume from. `batch_base` is the absolute index of this
// dispatch's first batch, so a reported offset means the same thing whether the
// launch ran in one dispatch or twenty.
kernel void honion_search(
    device u8 *walk_points            [[buffer(0)]],
    constant uint &num_threads        [[buffer(1)]],
    constant uint &num_batches        [[buffer(2)]],
    const device u32 *off_table       [[buffer(3)]],
    const device u32 *giant_niels     [[buffer(4)]],
    constant uint &num_groups         [[buffer(5)]],
    const device u64 *group_mask      [[buffer(6)]],
    const device u32 *group_off       [[buffer(7)]],
    const device u64 *target          [[buffer(8)]],
    const device u32 *target_pat      [[buffer(9)]],
    const device u32 *res_off         [[buffer(10)]],
    const device u64 *res             [[buffer(11)]],
    device Hit *hits                  [[buffer(12)]],
    device atomic_uint *hit_count     [[buffer(13)]],
    constant uint &max_hits           [[buffer(14)]],
    device atomic_uint *status        [[buffer(15)]],
    constant uint &batch_base         [[buffer(16)]],
    uint tid                          [[thread_position_in_grid]]) {

    if (tid >= num_threads) return;

    ge_precomp giant;
    for (int k = 0; k < FE_LIMBS; k++) {
        giant.yplusx.v[k]  = (int)giant_niels[k];
        giant.yminusx.v[k] = (int)giant_niels[FE_LIMBS + k];
        giant.xy2d.v[k]    = (int)giant_niels[2 * FE_LIMBS + k];
    }

    ge_decode dec = ge_frombytes(walk_points + 32 * (ulong)tid);
    if (!dec.ok) {
        atomic_fetch_or_explicit(status, STATUS_BAD_START_POINT, memory_order_relaxed);
        return;
    }
    ge_p3 point = dec.p;

    // One field element per candidate — only the numerator; denominators are
    // recomputed in the backward pass (see the CUDA comment on why).
    thread fe ynum[2 * HALF];

    if (!fe_isnonzero(point.Z)) {
        atomic_fetch_or_explicit(status, STATUS_SINGULAR, memory_order_relaxed);
        return;
    }
    fe zinv = fe_invert(point.Z);
    ge_affine base = ge_p3_to_affine(point, zinv);

    #pragma clang loop unroll(disable)
    for (u32 batch = 0; batch < num_batches; batch++) {
        const int centre = (int)(batch_base + batch) * CANDS_PER_BATCH;

        // The base point is candidate zero; its y is already affine.
        HONION_CHECK(base.y, centre);

        // Forward pass, with the running product split in two independent
        // chains (one for + candidates, one for -) to halve the dependency
        // chain, exactly as the CUDA kernel does.
        fe run_p = fe_one();
        fe run_m = fe_one();
        #pragma clang loop unroll(disable)
        for (u32 j = 0; j < HALF; j++) {
            const device u32 *slot = off_table + (ulong)j * OFF_STRIDE;
            fe ox, oy, oxy;
            for (int k = 0; k < FE_LIMBS; k++) {
                ox.v[k]  = (int)slot[k];
                oy.v[k]  = (int)slot[FE_LIMBS + k];
                oxy.v[k] = (int)slot[2 * FE_LIMBS + k];
            }

            ge_yfrac plus, minus;
            ge_dual_pair(plus, minus, base, ox, oy, oxy);

            const u32 i0 = 2 * j, i1 = 2 * j + 1;
            ynum[i0] = fe_mul(run_p, plus.num);
            run_p    = fe_mul(run_p, plus.den);
            ynum[i1] = fe_mul(run_m, minus.num);
            run_m    = fe_mul(run_m, minus.den);
        }

        // Advance to the next batch's base while still projective, folding its Z
        // into the same product so one inversion serves the whole batch.
        point = ge_p1p1_to_p3(ge_madd(point, giant));

        fe prod = fe_mul(run_p, run_m);
        fe run = fe_mul(prod, point.Z);

        if (!fe_isnonzero(run)) {
            atomic_fetch_or_explicit(status, STATUS_SINGULAR, memory_order_relaxed);
            return;
        }

        fe inv = fe_invert(run);

        fe next_zinv = fe_mul(inv, prod);
        fe inv_prod = fe_mul(inv, point.Z);
        fe acc_p = fe_mul(inv_prod, run_m);
        fe acc_m = fe_mul(inv_prod, run_p);

        // Backward pass, pairs in reverse so candidate indices descend.
        #pragma clang loop unroll(disable)
        for (int j = HALF - 1; j >= 0; j--) {
            const device u32 *slot = off_table + (ulong)j * OFF_STRIDE;
            fe ox, oy, oxy;
            for (int k = 0; k < FE_LIMBS; k++) {
                ox.v[k]  = (int)slot[k];
                oy.v[k]  = (int)slot[FE_LIMBS + k];
                oxy.v[k] = (int)slot[2 * FE_LIMBS + k];
            }
            ge_yfrac plus, minus;
            ge_dual_pair(plus, minus, base, ox, oy, oxy);

            const int step = j + 1;
            fe y_m = fe_mul(acc_m, ynum[2 * j + 1]);   // base - (j+1)*8B
            fe y_p = fe_mul(acc_p, ynum[2 * j]);       // base + (j+1)*8B
            acc_m = fe_mul(acc_m, minus.den);
            acc_p = fe_mul(acc_p, plus.den);
            HONION_CHECK(y_m, centre - step);
            HONION_CHECK(y_p, centre + step);
        }

        base = ge_p3_to_affine(point, next_zinv);
    }

    // Hand the walk to the next dispatch. The loop body already advanced
    // `point` past the batch it just finished, so this is exactly the base
    // point batch `batch_base + num_batches` starts from — splitting a launch
    // into several dispatches computes the same thing as running it in one.
    // A thread that returned early above deliberately does not write: it left a
    // status flag, and the host fails the whole launch on that.
    ge_p3_tobytes(walk_points + 32 * (ulong)tid, point);
}

// Report the compressed keys a thread visits, for testing. Simple +8B walk,
// pinning "the key at offset k" independently of the batched enumeration.
kernel void honion_walk_dump(const device u8 *start_points  [[buffer(0)]],
                             constant uint &num_threads     [[buffer(1)]],
                             constant uint &iterations      [[buffer(2)]],
                             device u8 *out                 [[buffer(3)]],
                             uint tid                       [[thread_position_in_grid]]) {
    if (tid >= num_threads) return;

    ge_decode dec = ge_frombytes(start_points + 32 * (ulong)tid);
    if (!dec.ok) return;
    ge_p3 point = dec.p;

    for (u32 k = 0; k < iterations; k++) {
        ge_p3_tobytes(out + 32 * ((ulong)tid * iterations + k), point);
        point = ge_add_8b(point);
    }
}
