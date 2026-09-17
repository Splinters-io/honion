// Warp-cooperative vanity search kernel.
//
// Same algorithm as search.cu, but each walk uses 8 threads (one cooperative
// group) instead of 1. Each thread holds one limb of each field element,
// cutting register demand from ~128 to ~20 per thread and enabling 3-4x
// occupancy.
//
// Requires CC 7.0+ for __shfl_sync.

#include "ge25519_coop.cuh"

#ifndef HALF
#define HALF 128
#endif

#define CANDS_PER_BATCH (2 * HALF + 1)
#define OFF_STRIDE (3 * FE_LIMBS)

#define STATUS_BAD_START_POINT 1u
#define STATUS_HIT_OVERFLOW    2u
#define STATUS_SINGULAR        4u

struct Hit {
    u32 thread_id;
    i32 offset;
    u32 pattern_id;
    u32 reserved;
};

__device__ __forceinline__ u32 key_char_value(const u8 *key, u32 index) {
    const u32 bit = index * 5u;
    const u32 byte = bit >> 3;
    const u32 off = bit & 7u;
    const u32 b0 = key[byte];
    const u32 b1 = (byte + 1u < 32u) ? key[byte + 1u] : 0u;
    return (((b0 << 8) | b1) >> (11u - off)) & 0x1fu;
}

__device__ __forceinline__ bool residuals_hold(const u8 *key, u32 pid,
                                               const u32 *__restrict__ res_off,
                                               const u64 *__restrict__ res) {
    const u32 end = res_off[pid + 1];
    for (u32 i = res_off[pid]; i < end; i++) {
        const u64 entry = res[i];
        if (((((u32)entry) >> key_char_value(key, (u32)(entry >> 32))) & 1u) == 0u) return false;
    }
    return true;
}

// Check a cooperative y coordinate against pattern tables.
// All lanes compute the same probe and run the same branches.
#define HONION_CHECK_COOP(Y_LIMB, OFFSET)                                      \
    do {                                                                       \
        const u64 probe_base = fe_prefix_be64_coop(Y_LIMB);                    \
        for (u32 g = 0; g < num_groups; g++) {                                 \
            const u64 probe = probe_base & group_mask[g];                      \
            u32 lo = group_off[g];                                             \
            u32 hi = group_off[g + 1];                                         \
            while (lo < hi) {                                                  \
                const u32 mid = lo + ((hi - lo) >> 1);                         \
                if (target[mid] < probe) lo = mid + 1; else hi = mid;          \
            }                                                                  \
            u8 key[32];                                                        \
            bool key_ready = false;                                            \
            for (u32 t = lo; t < group_off[g + 1] && target[t] == probe; t++) { \
                const u32 pid = target_pat[t];                                 \
                if (res_off[pid + 1] != res_off[pid]) {                        \
                    if (!key_ready) {                                           \
                        fe_tobytes_coop(key, Y_LIMB);                          \
                        key_ready = true;                                       \
                    }                                                          \
                    if (!residuals_hold(key, pid, res_off, res)) continue;     \
                }                                                              \
                if (lane == 0) {                                               \
                    const u32 slot = atomicAdd(hit_count, 1u);                 \
                    if (slot < max_hits) {                                     \
                        hits[slot].thread_id = walk_id;                        \
                        hits[slot].offset = (OFFSET);                          \
                        hits[slot].pattern_id = pid;                           \
                        hits[slot].reserved = 0;                               \
                    } else {                                                   \
                        atomicOr(status, STATUS_HIT_OVERFLOW);                 \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

// Build the offset table. Runs single-threaded, same as scalar version.
// This is NOT cooperative — it uses scalar field ops from ge25519.cuh.
extern "C" __global__ void honion_build_offsets(u32 *__restrict__ table,
                                                u32 *__restrict__ giant,
                                                u32 *__restrict__ status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    ge_precomp step;
    ge_precomp_8b(&step);

    ge_p3 p;
    ge_p3_8b(&p);

    for (u32 j = 0; j < HALF; j++) {
        fe zinv;
        if (!fe_isnonzero(p.Z)) { atomicOr(status, STATUS_SINGULAR); return; }
        fe_invert(zinv, p.Z);
        ge_affine a;
        ge_p3_to_affine(&a, &p, zinv);

        u32 *slot = table + (size_t)j * OFF_STRIDE;
#pragma unroll
        for (int k = 0; k < FE_LIMBS; k++) {
            slot[k] = (u32)a.x[k];
            slot[FE_LIMBS + k] = (u32)a.y[k];
            slot[2 * FE_LIMBS + k] = (u32)a.xy[k];
        }

        ge_p1p1 t;
        ge_madd(&t, &p, &step);
        ge_p1p1_to_p3(&p, &t);
    }

    for (u32 j = HALF + 1; j < CANDS_PER_BATCH; j++) {
        ge_p1p1 t;
        ge_madd(&t, &p, &step);
        ge_p1p1_to_p3(&p, &t);
    }

    fe zinv;
    if (!fe_isnonzero(p.Z)) { atomicOr(status, STATUS_SINGULAR); return; }
    fe_invert(zinv, p.Z);
    fe gx, gy;
    fe_mul(gx, p.X, zinv);
    fe_mul(gy, p.Y, zinv);
    ge_precomp g;
    ge_affine_to_precomp(&g, gx, gy);
#pragma unroll
    for (int k = 0; k < FE_LIMBS; k++) {
        giant[k] = (u32)g.yplusx[k];
        giant[FE_LIMBS + k] = (u32)g.yminusx[k];
        giant[2 * FE_LIMBS + k] = (u32)g.xy2d[k];
    }
}

// The cooperative search kernel.
//
// Each cooperative group of COOP_WIDTH (8) threads walks one scalar
// region. `num_walks` is the number of independent walks (equivalent to
// `num_threads` in the scalar kernel). The total thread count launched
// must be num_walks * COOP_WIDTH.
extern "C" __global__ __launch_bounds__(256) void honion_search_coop(
    const u8 *__restrict__ start_points,
    u32 num_walks,
    u32 num_batches,
    const u32 *__restrict__ off_table,
    const u32 *__restrict__ giant_niels,
    u32 num_groups,
    const u64 *__restrict__ group_mask,
    const u32 *__restrict__ group_off,
    const u64 *__restrict__ target,
    const u32 *__restrict__ target_pat,
    const u32 *__restrict__ res_off,
    const u64 *__restrict__ res,
    Hit *__restrict__ hits,
    u32 *__restrict__ hit_count,
    u32 max_hits,
    u32 *__restrict__ status) {

    const int lane = threadIdx.x & (COOP_WIDTH - 1);
    const u32 global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 walk_id = global_thread / COOP_WIDTH;

    // Load offset table into shared memory. Each group of 8 threads
    // cooperates on loading, but the data is shared across the block.
    __shared__ u32 s_off[HALF * OFF_STRIDE];
    for (u32 i = threadIdx.x; i < HALF * OFF_STRIDE; i += blockDim.x) {
        s_off[i] = off_table[i];
    }
    __syncthreads();

    // All threads must participate in shuffles. Threads past num_walks
    // skip memory ops but still execute cooperative arithmetic.
    const bool active = (walk_id < num_walks);

    // Load the giant step precomp into cooperative layout.
    // Each lane reads its limb from the giant_niels table.
    u32 giant_ypx = giant_niels[lane];
    u32 giant_ymx = giant_niels[FE_LIMBS + lane];
    u32 giant_xy2d = giant_niels[2 * FE_LIMBS + lane];

    // Decompress starting point cooperatively.
    u32 pX, pY, pZ, pT;
    u32 decompress_ok = 1;
    if (active) {
        decompress_ok = ge_frombytes_coop(pX, pY, pZ, pT,
                                          start_points + 32 * (size_t)walk_id);
    } else {
        fe_1_coop(pY);
        fe_1_coop(pZ);
        pX = 0;
        pT = 0;
    }
    if (active && !decompress_ok) {
        if (lane == 0) atomicOr(status, STATUS_BAD_START_POINT);
        return;
    }

    // Per-walk numerator array. In cooperative mode each thread holds one
    // limb, so this is u32[2*HALF] per thread instead of fe[2*HALF].
    u32 ynum[2 * HALF];

    // First affine conversion: one inversion for the initial base point.
    u32 z_nz = fe_isnonzero_coop(pZ);
    if (active && !z_nz) {
        if (lane == 0) atomicOr(status, STATUS_SINGULAR);
        return;
    }
    u32 zinv;
    fe_invert_coop(zinv, pZ);
    u32 base_x, base_y, base_xy;
    ge_p3_to_affine_coop(base_x, base_y, base_xy, pX, pY, zinv);

#pragma unroll 1
    for (u32 batch = 0; batch < num_batches; batch++) {
        const i32 centre = (i32)batch * CANDS_PER_BATCH;

        // The base point is itself a candidate.
        if (active) {
            HONION_CHECK_COOP(base_y, centre);
        }

        // Forward pass: build numerators pre-multiplied by running product.
        u32 run_p, run_m;
        fe_1_coop(run_p);
        fe_1_coop(run_m);

#pragma unroll 1
        for (u32 j = 0; j < HALF; j++) {
            const u32 *slot = s_off + (size_t)j * OFF_STRIDE;
            // Load offset into cooperative layout.
            u32 ox = slot[lane];
            u32 oy = slot[FE_LIMBS + lane];
            u32 oxy = slot[2 * FE_LIMBS + lane];

            // Dual addition law: 2 fe_mul + 4 fe_add/sub.
            u32 pnum, pden, mnum, mden;
            ge_dual_pair_coop(pnum, pden, mnum, mden,
                              base_x, base_y, base_xy, ox, oy, oxy);

            const u32 i0 = 2 * j, i1 = 2 * j + 1;
            fe_mul_coop(ynum[i0], run_p, pnum);
            fe_mul_coop(run_p, run_p, pden);
            fe_mul_coop(ynum[i1], run_m, mnum);
            fe_mul_coop(run_m, run_m, mden);
        }

        // Advance to next batch's base point.
        u32 tX, tY, tZ, tT;
        ge_madd_coop(tX, tY, tZ, tT, pX, pY, pZ, pT,
                     giant_ypx, giant_ymx, giant_xy2d);
        ge_p1p1_to_p3_coop(pX, pY, pZ, pT, tX, tY, tZ, tT);

        // Combine running products + next Z into single inversion.
        u32 prod;
        fe_mul_coop(prod, run_p, run_m);
        u32 run;
        fe_mul_coop(run, prod, pZ);

        u32 run_nz = fe_isnonzero_coop(run);
        if (active && !run_nz) {
            if (lane == 0) atomicOr(status, STATUS_SINGULAR);
            return;
        }

        u32 inv;
        fe_invert_coop(inv, run);

        // Peel apart: 1/Z for next base, then one inverse per chain.
        u32 next_zinv, inv_prod, acc_p, acc_m;
        fe_mul_coop(next_zinv, inv, prod);
        fe_mul_coop(inv_prod, inv, pZ);
        fe_mul_coop(acc_p, inv_prod, run_m);
        fe_mul_coop(acc_m, inv_prod, run_p);

        // Backward pass: recover y values and check patterns.
#pragma unroll 1
        for (i32 j = HALF - 1; j >= 0; j--) {
            const u32 *slot = s_off + (size_t)j * OFF_STRIDE;
            u32 ox = slot[lane];
            u32 oy = slot[FE_LIMBS + lane];
            u32 oxy = slot[2 * FE_LIMBS + lane];

            u32 pnum, pden, mnum, mden;
            ge_dual_pair_coop(pnum, pden, mnum, mden,
                              base_x, base_y, base_xy, ox, oy, oxy);

            const i32 step = j + 1;
            u32 y_m, y_p;
            fe_mul_coop(y_m, acc_m, ynum[2 * j + 1]);
            fe_mul_coop(y_p, acc_p, ynum[2 * j]);
            fe_mul_coop(acc_m, acc_m, mden);
            fe_mul_coop(acc_p, acc_p, pden);

            if (active) {
                HONION_CHECK_COOP(y_m, centre - step);
                HONION_CHECK_COOP(y_p, centre + step);
            }
        }

        ge_p3_to_affine_coop(base_x, base_y, base_xy, pX, pY, next_zinv);
    }
}

// Cooperative walk dump for testing.
extern "C" __global__ void honion_walk_dump_coop(
    const u8 *__restrict__ start_points,
    u32 num_walks, u32 iterations,
    u8 *__restrict__ out) {

    const int lane = threadIdx.x & (COOP_WIDTH - 1);
    const u32 walk_id = (blockIdx.x * blockDim.x + threadIdx.x) / COOP_WIDTH;
    const bool active = (walk_id < num_walks);

    u32 pX, pY, pZ, pT;
    if (active) {
        if (!ge_frombytes_coop(pX, pY, pZ, pT,
                               start_points + 32 * (size_t)walk_id))
            return;
    } else {
        fe_1_coop(pY);
        fe_1_coop(pZ);
        pX = 0;
        pT = 0;
    }

#pragma unroll 1
    for (u32 k = 0; k < iterations; k++) {
        if (active) {
            // Compress: y coordinate with sign bit of x in bit 255.
            u32 recip;
            fe_invert_coop(recip, pZ);
            u32 x, y;
            fe_mul_coop(x, pX, recip);
            fe_mul_coop(y, pY, recip);
            u8 *dst = out + 32 * ((size_t)walk_id * iterations + k);
            fe_tobytes_coop(dst, y);
            u32 x_neg = fe_isnegative_coop(x);
            if (lane == 0) dst[31] ^= (u8)(x_neg << 7);
        }
        ge_add_8b_coop(pX, pY, pZ, pT);
    }
}
