# Performance: measurements and what they mean for you

All figures measured on the machine this was developed on:

- **GPU** NVIDIA RTX PRO 6000 Blackwell Workstation Edition — sm_120 (compute
  capability 12.0), 188 SMs, 24 064 CUDA cores, 96 GB, 3.09 GHz max SM clock
- **Driver** 595.84, CUDA 13.1 via NVRTC
- **CPU** 48 cores, **OS** Ubuntu 26.04

## Headline

**13.3 × 10⁹ candidate addresses per second** in the kernel, about **12.5 × 10⁹**
end to end at the default settings.

Four rewrites got it there. Every one came from a measurement, and each measured
a *different* bottleneck — which is the point: the thing limiting this kernel
changed three times, and optimising for the previous one would have been wasted
work each time.

| version | kernel | bottleneck it removed |
|---|---|---|
| first working kernel | 4.9 G/s | — |
| dual addition law | 8.3 G/s | work per candidate: 12 field multiplications → ~5.5 |
| 8×32-limb field arithmetic | 10.9 G/s | 64-bit ops: 56% of PTX → 2% |
| denominators recomputed, not stored | 12.5 G/s | DRAM: 1.05 TB/s (61% of peak) → 577 GB/s |
| running product split in two | **13.3 G/s** | a dependency chain 1025 multiplications long |

The first is algorithmic
([01-ed25519-vanity-search.md](01-ed25519-vanity-search.md)); the rest are
covered in [02-gpu-architecture.md](02-gpu-architecture.md), along with the five
things that were tried and did not work.

Note the shape of the last two. Once the arithmetic was cheap, removing further
arithmetic stopped helping entirely — ablating a fifth of the multiplications
changed throughput by 0.13%. The wins came from *memory* and then from
*latency*, and both were found by ablation rather than by reading the code.

Reproduce with `cargo run --release -p honion-gpu --example bench`, or just
watch the rate line during a real search:

```
$ honion search --prefix carroll --out ./keys --count 1
  combined difficulty 35.0 bits, prefilter 35.0 bits
  device compute capability 12.0, 131072 threads, 256 offsets (513 candidates per inversion)
     32.47G examined     7.56G/s  elapsed       4.3s  expected       4.5s  P(found)  61.1%  hits 0
     …
carrolliuz67cgsdlpeu5e2hcrcg3rzaowhrj2dps4pwwgrkxd7xrtqd.onion
```

The rate a real search reports is a little below the benchmark figure because it
also draws fresh secret scalars and derives their public points on the CPU
between launches — about 2.5% at the default four-second launches.

For scale, `mkp224o` — the established CPU generator, using the same algorithm —
reaches 2.6 × 10⁸/s on this machine's 24-core Threadripper. The difference is
24 064 cores and batched inversion, not a better method. Measured side by side
in [07-benchmarks.md](07-benchmarks.md).

## How long will my prefix take?

Expected time at 8.4 G/s. Each character multiplies the work by 32.

| characters | expected trials | expected time |
|---|---|---|
| 6 | 1.07 × 10⁹ | 0.13 s |
| 7 | 3.44 × 10¹⁰ | 4 s |
| 8 | 1.10 × 10¹² | 2.2 min |
| 9 | 3.52 × 10¹³ | 1.2 hours |
| 10 | 1.13 × 10¹⁵ | **1.6 days** |
| 11 | 3.60 × 10¹⁶ | 50 days |
| 12 | 1.15 × 10¹⁸ | 4.3 years |

**Ten characters is the practical ceiling on one card.** DuckDuckGo's
`duckduckgogg42…` is a ten-character vanity address, which gives a sense of what
is worth doing.

`honion estimate` reports this for any pattern without searching.

### These are means, not deadlines

The search is memoryless — every key is an independent trial — so the expected
time is where the *average* run lands, not where yours will. The chance of
having found something is:

| elapsed | probability |
|---|---|
| 0.5 × expected | 39% |
| 1 × expected | 63% |
| 2 × expected | 86% |
| 3 × expected | 95% |
| 5 × expected | 99.3% |

A run at twice the expected time is unremarkable; it happens to one run in
seven. The progress display reports `P(found)` so you can tell "unlucky" from
"broken".

Searching several patterns at once is nearly free and is the cheapest way to
improve your odds: two patterns halve the expected time.

## Tuning

`--offsets` sets the number of precomputed offsets (each yields two candidates,
so a batch covers `2N+1` keys per inversion); `--threads` sets concurrent walks
and defaults to a value chosen from free device memory.

Measured on an RTX PRO 6000 Blackwell, median of three sustained runs per point:

| `--offsets` | `--threads` | kernel | local memory / thread | device memory |
|---|---|---|---|---|
| 256 | 262 144 | 12.02 G/s | 16 KB | 4.3 GB |
| 512 | 262 144 | 12.74 G/s | 32 KB | 8.6 GB |
| 1024 | 262 144 | 12.86 G/s | 64 KB | 17.2 GB |
| 512 | 393 216 | 12.72 G/s | 32 KB | 12.9 GB |
| **512** | **524 288** | **13.28 G/s** | **32 KB** | **17.2 GB** |

Defaults are 512 offsets, with the thread count sized to the card. On this
machine that lands on 524 288 threads and 17.2 GB.

### End-to-end disagrees with the kernel, and end-to-end wins

More concurrent walks keep raising kernel throughput, but the host must draw a
fresh secret scalar and derive a public key for every thread each launch, and
that cost grows linearly with the count. Tuning on the kernel number alone once
picked a million threads and made the tool 9% *slower*.

That is why host preparation now runs *while* the GPU works — the CLI enqueues a
launch, draws the next epoch's scalars, and only then waits. With the overlap in
place the two figures agree closely again, and the remaining end-to-end gap is
startup plus per-key verification.

### Overlapping the host

Host preparation runs *while* the GPU works: the CLI enqueues a launch, draws
the next epoch's scalars, and only then waits. Without that overlap the same
work sits in front of every launch as dead time — about 2% at 262 144 threads,
and 17% at a million. It is what makes a large thread count merely unhelpful
rather than actively harmful.

## Where the time goes

Measured with Nsight Compute, at 256 offsets and 131 072 threads:

| | before the field rewrite | after |
|---|---|---|
| instructions per candidate | 2595 | **2019** |
| registers per thread | 254 | **128** |
| static shared memory per block | 30.7 KB | **24.6 KB** |
| achieved occupancy | 16.6% | **28.4%** |
| SM throughput | 60.7% | 62.1% |
| 64-bit ops as share of PTX | 56% | **2%** |

Two things happened at once. Fewer instructions, obviously — but also, because a
32-bit representation needs far fewer live registers than 64-bit accumulators,
the compiler dropped from 254 registers per thread (one short of the
architectural maximum) to 128, which let a second block fit on each SM and
nearly doubled occupancy.

That second effect was not planned. An earlier attempt to buy occupancy directly
— capping registers with `__launch_bounds__(256, 2)` — made things *slower*,
because the spilling cost more than the extra warps were worth. Occupancy was a
symptom; the register pressure was the cause.

## Startup

| step | time |
|---|---|
| NVRTC compile | 2.9 s |
| PTX load and JIT | ~0.1 s |
| scalar generation and point derivation, 131 072 threads, 48 cores | ~0.1 s |

The 2.9 s compile is paid once per run. Caching PTX on disk, keyed by source
hash and architecture, would remove it and is the obvious next improvement.

Fresh scalars are drawn every launch rather than continuing the previous walk.
At the default 4-second launches that is ~2.5% overhead, bought in exchange for
each launch being an independent random sample and for there being no
cross-launch state to get wrong.

## Two compiler lessons, kept because they cost real time

**Cold routines must be `__noinline__`.** `fe_invert` and `fe_pow22523` are 261
field operations each; inlined, they multiplied the PTX by ~50×. They run once
per batch.

**Batch loops must be `#pragma unroll 1`.** NVRTC's unrolling heuristic flipped
unpredictably with `BATCH_SIZE`:

| `BATCH_SIZE` | compile | PTX | rate |
|---|---|---|---|
| 32 | 44 ms | 4.9 MB | 0.99 G/s |
| 64 | **284 s** | 9.6 MB | — |
| 128 | 11 ms | 876 KB | 4.35 G/s |

The compact version was 4× faster, and the cliff at 64 made the build unusable.
Pinning unrolling off gives predictable code size and compile time at every
setting, and made throughput monotonic in the batch size as the arithmetic says
it should be. This was the first real speedup the project got, and it was one
line — worth remembering before reaching for a rewrite.

## What was not done

- **A specialised `fe_sq`.** Squaring can exploit symmetry for roughly a third
  fewer products, but it is used almost only inside the inversion, whose cost is
  already amortised to about half a multiplication per candidate.
- **Block-wide batch inversion.** Sharing one inversion across a whole block via
  a parallel scan would cut per-thread local memory. Since memory is no longer
  what limits the useful thread count — host preparation is — this would buy
  something the current bottleneck does not want.
- **Cutting host preparation further.** It is overlapped, so it is nearly free
  at the default thread count, but it is what stops larger thread counts from
  helping. Deriving public keys on the GPU would break the property that no
  secret is ever derived near the device, so it is not obviously worth it.
- **A specialised `fe_sq`.** Squaring can exploit symmetry, but it is used
  almost only inside the inversion, whose cost is already amortised to ~1
  multiplication per candidate.
- **Block-wide batch inversion.** Sharing one inversion across a whole block via
  a parallel scan would cut per-thread local memory. Since 4 GB at the default
  settings is not a constraint, it would buy occupancy that the flat part of the
  curve says is not the bottleneck.
