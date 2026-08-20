// The vanity search kernel.
//
// Each thread walks its own region of scalar space and reports keys whose
// base32 encoding begins the way the caller asked. Four observations make it
// fast; each is explained where it is used.
//
//   1. Consecutive valid scalars differ by 8, so neighbouring public keys
//      differ by a point addition rather than a scalar multiplication.
//   2. The first 51 characters of an address depend only on the public key,
//      never on the checksum, so no hashing is needed to test a prefix.
//   3. The dual addition law yields y directly, needs no curve constant, and
//      produces `base + off` and `base - off` from two multiplications —
//      *two* candidates for the price of one. (`ge25519.cuh`)
//   4. Those y values are fractions, and one modular inversion serves an entire
//      batch of them via Montgomery's trick, with the base point's own Z folded
//      into the same product so that a batch costs exactly one inversion.
//
// Together these cost about six field multiplications per candidate. A
// straightforward implementation — stepping one point at a time in extended
// coordinates — costs twelve; see docs/07-benchmarks.md for the measurement
// that prompted the change.
//
// ## What this kernel is trusted with
//
// Nothing. It holds no secret: it is given public starting points and returns
// (thread, offset) pairs that *might* be matches. The host reconstructs the
// secret from its own memory, re-derives the key with an independent
// implementation, and re-checks the address before writing anything.

#include "ge25519.cuh"

// Number of positive offsets in the table. Each yields two candidates, so a
// batch covers 2*HALF + 1 candidates: the base point and HALF pairs either
// side of it. Overridable at compile time; see docs/06-performance.md.
#ifndef HALF
#define HALF 128
#endif

// Candidates produced per batch, per thread.
#define CANDS_PER_BATCH (2 * HALF + 1)

// Field-element limbs, and the stride of one offset table entry (x, y, xy).
#define OFF_STRIDE (3 * FE_LIMBS)

// Bits set in the status word to report conditions the host must know about.
#define STATUS_BAD_START_POINT 1u
#define STATUS_HIT_OVERFLOW    2u
#define STATUS_SINGULAR        4u

// A reported candidate.
//
// `offset` is signed: the search covers a symmetric range either side of each
// thread's starting scalar, so a match may lie below it. The host reconstructs
// the secret as `a0 + 8 * offset`.
struct Hit {
    u32 thread_id;
    i32 offset;
    u32 pattern_id;
    u32 reserved;
};

// The 5-bit value that base32 character `index` encodes, read from a raw key.
//
// Characters run most-significant-bit first, so character i occupies bits
// [5i, 5i+5) counting from the top bit of byte 0. Must agree exactly with
// `honion_core::pattern::char_value`.
__device__ __forceinline__ u32 key_char_value(const u8 *key, u32 index) {
    const u32 bit = index * 5u;
    const u32 byte = bit >> 3;
    const u32 off = bit & 7u;
    const u32 b0 = key[byte];
    const u32 b1 = (byte + 1u < 32u) ? key[byte + 1u] : 0u;
    return (((b0 << 8) | b1) >> (11u - off)) & 0x1fu;
}

// Whether `key` satisfies the residual constraints of pattern `pid`.
//
// Residuals are the parts of a pattern the 64-bit prefilter cannot express:
// multi-character classes, and any position past character 12. Most patterns
// have none, in which case this is never called.
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

// Build the offset table: affine (x, y, x*y) for 1*8B .. HALF*8B, plus the
// giant step (2*HALF+1)*8B in niels form.
//
// Run once per search, by a single thread. It costs HALF+1 modular inversions
// — a few milliseconds — against a search that runs for seconds at minimum, so
// it is not worth parallelising. Doing it on the device rather than embedding a
// generated table keeps HALF tunable without regenerating source, and reuses
// the same arithmetic the differential tests already cover.
extern "C" __global__ void honion_build_offsets(u32 *__restrict__ table,
                                                u32 *__restrict__ giant,
                                                u32 *__restrict__ status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    ge_precomp step;
    ge_precomp_8b(&step);

    ge_p3 p;
    ge_p3_8b(&p);   // p = 1 * 8B

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
        ge_p1p1_to_p3(&p, &t);   // p = (j+2) * 8B
    }

    // Continue to (2*HALF + 1) * 8B: the stride between consecutive batches,
    // chosen so that the ranges [-HALF, +HALF] around successive base points
    // tile the integers exactly, with no gap and no overlap.
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

// Test one candidate's y coordinate against the pattern tables, recording a
// hit if it matches. Defined as a macro so that it can be used from both the
// batched path and the base-point path without duplicating the table walk or
// paying for a function call in the inner loop.
#define HONION_CHECK(Y, OFFSET)                                                \
    do {                                                                       \
        const u64 probe_base = fe_prefix_be64(Y);                              \
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
                    if (!key_ready) { fe_tobytes(key, Y); key_ready = true; }  \
                    if (!residuals_hold(key, pid, res_off, res)) continue;     \
                }                                                              \
                const u32 slot = atomicAdd(hit_count, 1u);                     \
                if (slot < max_hits) {                                         \
                    hits[slot].thread_id = tid;                                \
                    hits[slot].offset = (OFFSET);                              \
                    hits[slot].pattern_id = pid;                               \
                    hits[slot].reserved = 0;                                   \
                } else {                                                       \
                    atomicOr(status, STATUS_HIT_OVERFLOW);                     \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

// The search.
//
// Table layout, all built and validated on the host (langsec rule 4 — the
// device parses nothing and reads no length out of data).
extern "C" __global__ __launch_bounds__(256) void honion_search(
    const u8 *__restrict__ start_points,
    u32 num_threads,
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

    // The offset table is identical for every thread, so it is staged in shared
    // memory once per block rather than re-read from global memory by each of
    // the 2*HALF multiplications every batch performs.
    __shared__ u32 s_off[HALF * OFF_STRIDE];
    for (u32 i = threadIdx.x; i < HALF * OFF_STRIDE; i += blockDim.x) {
        s_off[i] = off_table[i];
    }
    __syncthreads();

    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    ge_precomp giant;
#pragma unroll
    for (int k = 0; k < FE_LIMBS; k++) {
        giant.yplusx[k] = (fe_limb)giant_niels[k];
        giant.yminusx[k] = (fe_limb)giant_niels[FE_LIMBS + k];
        giant.xy2d[k] = (fe_limb)giant_niels[2 * FE_LIMBS + k];
    }

    ge_p3 point;
    if (!ge_frombytes(&point, start_points + 32 * (size_t)tid)) {
        // The host only ever sends real public keys, so this means the transfer
        // or the arithmetic is broken. Reporting beats walking meaningless
        // values at full speed, which looks exactly like bad luck.
        atomicOr(status, STATUS_BAD_START_POINT);
        return;
    }

    // One field element per candidate, and only the numerator.
    //
    // The denominators are *not* stored. Profiling showed this kernel is bound
    // by local-memory bandwidth, not arithmetic: ablating a fifth of the
    // multiplies changed throughput by 0.1%, while DRAM sat at 61% of peak
    // moving about 106 bytes per candidate. Storing both halves of each
    // fraction was most of that traffic.
    //
    // So the backward pass recomputes each denominator from the base point and
    // the offset table — two multiplications per pair — instead of reading it
    // back. That is one more multiplication per candidate and half the memory
    // traffic, which on a memory-bound kernel is a trade worth making.
    fe ynum[2 * HALF];

    // The first batch needs an affine base point, which costs one inversion.
    // Every later batch gets 1/Z out of its own batch inversion for free — see
    // the fold below — so this is paid once per launch, not once per batch.
    fe zinv;
    if (!fe_isnonzero(point.Z)) { atomicOr(status, STATUS_SINGULAR); return; }
    fe_invert(zinv, point.Z);
    ge_affine base;
    ge_p3_to_affine(&base, &point, zinv);

#pragma unroll 1
    for (u32 batch = 0; batch < num_batches; batch++) {
        const i32 centre = (i32)batch * CANDS_PER_BATCH;

        // The base point is itself a candidate, at offset zero within the
        // batch. Its y is already affine, so it needs no division.
        HONION_CHECK(base.y, centre);

        // Forward pass. Candidate 2j is base + (j+1)*8B, candidate 2j+1 is
        // base - (j+1)*8B. Each numerator is pre-multiplied by the product of
        // all denominators before it, which lets the backward pass recover y
        // with two multiplications instead of three.
        //
        // The running product is split in two — one accumulator for the `+`
        // candidates and one for the `-` candidates — because a single product
        // is a dependency chain 2*HALF multiplications long, and with roughly
        // two warps per scheduler there is nothing to interleave with it.
        // Profiling showed the kernel bound by that latency rather than by
        // instruction throughput: ablating a fifth of the multiplies changed
        // nothing measurable. Two chains halve the critical path and give the
        // scheduler two independent streams per thread. They are recombined
        // below at a cost of a few multiplications per batch.
        fe run_p, run_m;
        fe_1(run_p);
        fe_1(run_m);
#pragma unroll 1
        for (u32 j = 0; j < HALF; j++) {
            const u32 *slot = s_off + (size_t)j * OFF_STRIDE;
            fe ox, oy, oxy;
#pragma unroll
            for (int k = 0; k < FE_LIMBS; k++) {
                ox[k] = (fe_limb)slot[k];
                oy[k] = (fe_limb)slot[FE_LIMBS + k];
                oxy[k] = (fe_limb)slot[2 * FE_LIMBS + k];
            }

            ge_yfrac plus, minus;
            ge_dual_pair(&plus, &minus, &base, ox, oy, oxy);

            const u32 i0 = 2 * j, i1 = 2 * j + 1;
            fe_mul(ynum[i0], run_p, plus.num);
            fe_mul(run_p, run_p, plus.den);
            fe_mul(ynum[i1], run_m, minus.num);
            fe_mul(run_m, run_m, minus.den);
        }

        // Advance to the next batch's base point while still projective, and
        // fold its Z into the same product. One inversion then serves both the
        // 2*HALF candidates and the next affine conversion.
        ge_p1p1 t;
        ge_madd(&t, &point, &giant);
        ge_p1p1_to_p3(&point, &t);

        // Combine the two chains and the next base point's Z into one product,
        // so the batch still costs exactly one inversion.
        fe prod;
        fe_mul(prod, run_p, run_m);
        fe run;
        fe_mul(run, prod, point.Z);

        if (!fe_isnonzero(run)) {
            // A denominator vanished, which for the dual law means the base
            // point coincided with an offset. The walk cannot produce that from
            // a valid start, so it is a fault rather than an ordinary case.
            atomicOr(status, STATUS_SINGULAR);
            return;
        }

        fe inv;
        fe_invert(inv, run);

        // Peel the pieces apart: 1/Z for the next base point, then one inverse
        // per chain. `inv_prod` is 1/(run_p * run_m), so multiplying by the
        // other chain's product isolates each.
        fe next_zinv, inv_prod, acc_p, acc_m;
        fe_mul(next_zinv, inv, prod);
        fe_mul(inv_prod, inv, point.Z);
        fe_mul(acc_p, inv_prod, run_m);
        fe_mul(acc_m, inv_prod, run_p);

        // Backward pass, walking pairs in reverse so candidate indices still
        // descend: 2j+1 then 2j. Each pair's two denominators are rebuilt from
        // the same two products the forward pass used, rather than read back
        // from memory.
#pragma unroll 1
        for (i32 j = HALF - 1; j >= 0; j--) {
            const u32 *slot = s_off + (size_t)j * OFF_STRIDE;
            fe ox, oy, oxy;
#pragma unroll
            for (int k = 0; k < FE_LIMBS; k++) {
                ox[k] = (fe_limb)slot[k];
                oy[k] = (fe_limb)slot[FE_LIMBS + k];
                oxy[k] = (fe_limb)slot[2 * FE_LIMBS + k];
            }
            ge_yfrac plus, minus;
            ge_dual_pair(&plus, &minus, &base, ox, oy, oxy);

            const i32 step = j + 1;
            // The two chains unwind independently and interleaved, which is the
            // whole point: neither multiplication below waits on the other.
            fe y_m, y_p;
            fe_mul(y_m, acc_m, ynum[2 * j + 1]);   // base - (j+1)*8B
            fe_mul(y_p, acc_p, ynum[2 * j]);       // base + (j+1)*8B
            fe_mul(acc_m, acc_m, minus.den);
            fe_mul(acc_p, acc_p, plus.den);
            HONION_CHECK(y_m, centre - step);
            HONION_CHECK(y_p, centre + step);
        }

        ge_p3_to_affine(&base, &point, next_zinv);
    }
}

// Report the compressed keys a thread visits, for testing.
//
// Walks by single additions of 8B, which is the simple enumeration the search
// used before the dual-law rewrite. It is retained because it pins the meaning
// of "the key at offset k" independently of the batched enumeration, so the
// tests can check that the search visits the keys it claims to.
extern "C" __global__ void honion_walk_dump(const u8 *__restrict__ start_points,
                                            u32 num_threads, u32 iterations,
                                            u8 *__restrict__ out) {
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    ge_p3 point;
    if (!ge_frombytes(&point, start_points + 32 * (size_t)tid)) return;

#pragma unroll 1
    for (u32 k = 0; k < iterations; k++) {
        ge_p3_tobytes(out + 32 * ((size_t)tid * iterations + k), &point);
        ge_add_8b(&point);
    }
}
