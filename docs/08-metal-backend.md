# The Metal backend: honion on Apple Silicon

honion has two GPU backends behind one interface. The reference backend is CUDA
([docs/02](02-gpu-architecture.md)); the second is native Metal, for Apple
Silicon. They are selected at build time by a Cargo feature — `cuda` or `metal`
— and `honion-cli` picks the right one per platform, so the source is
backend-agnostic and names `honion_gpu::Searcher` without knowing which device
is underneath.

This document covers what is specific to the Metal backend. The address format,
the vanity-search idea, the field and curve arithmetic, and the security model
are backend-independent and live in [00](00-onion-v3-address-spec.md),
[01](01-ed25519-vanity-search.md), and [05](05-security-model.md).

## The same algorithm, a different compiler

The Metal kernels are a port, not a rewrite. `metal/fe25519.metal`,
`metal/ge25519.metal` and `metal/search.metal` implement exactly the algorithms
[docs/01](01-ed25519-vanity-search.md) and [docs/02](02-gpu-architecture.md)
describe — the radix-25.5 field, the dual addition law, the batched inversion —
and they are gated by the same differential tests, run against the same
independent references (`num-bigint`, `curve25519-dalek`) demanding the same
exact agreement. A field element is a `struct` value in Metal rather than a bare
array, because Metal's address spaces make value-returning functions natural
where CUDA passed pointers; the mathematics is unchanged.

The field representation is the radix-2^25.5 one, not the 8×32-limb variant the
CUDA backend settled on. Metal has no add-with-carry intrinsic and runs 64-bit
integer arithmetic well below 32-bit, so the carry-deferred reduced-radix form —
which needs no carry flag and keeps every intermediate in a 64-bit accumulator
with provable headroom — is the right fit. (DEC-METAL-004.)

## Runtime compilation, and the closed include set

Like NVRTC on the CUDA side, Metal compiles device code from a source string at
run time (`newLibraryWithSource`). honion uses this the same way: the sources
are embedded in the binary with `include_str!`, `HALF` is baked in per run as a
compile constant, and `#include "..."` directives resolve against a closed set
of embedded headers — an unknown include is an error, not a filesystem lookup.
The set of text that can reach the device compiler is therefore fixed when the
binary is built (langsec rule 4, [docs/03](03-langsec-design.md)), enforced by
honion's own code rather than by platform behaviour. (DEC-METAL-005.)

## The two compiler traps, rediscovered

[docs/06](06-performance.md) records two NVRTC-specific lessons that cost real
time: cold routines must be `__noinline__`, and batch loops must not be
unrolled. Neither transfers to a different compiler, so both were rediscovered
on Metal's:

- **Cold routines** (`fe_invert`, `fe_pow22523`) carry `[[clang::noinline]]`.
  Each is ~261 field operations; inlined into the search kernel's batch loop
  they would multiply the code size the way they did on CUDA.

- **Batch loops** carry `#pragma clang loop unroll(disable)`. `HALF` is a
  compile-time constant (it sizes per-thread arrays), so without this the
  compiler could fully unroll a 512-iteration loop of hundred-term multiplies.
  Clang has historically ignored this pragma, so the effect is *verified* rather
  than assumed: the search kernel compiles in about half a second and its
  compile time stays bounded across `HALF` from 1 to 512. Were the pragma
  ignored, the compile would blow up as NVRTC's did (docs/06 records 284 s and
  9.6 MB at one setting). (DEC-METAL-006.)

## Unified memory removes the copy

On a discrete GPU the host must copy start points to the device and hits back,
and honion overlaps that transfer with host work to hide it (docs/06,
"Overlapping the host"). Apple Silicon shares system memory, so there is no
separate device address space and no copy to hide: buffers are `storageModeShared`,
the CPU writes start points straight into the buffer the GPU reads, and reads
hits straight out of the buffer the GPU wrote. The overlap machinery is simply
not needed here, and the Metal `Searcher` runs its dispatch synchronously.
(DEC-METAL-007.)

One consequence: the offset table is read from device memory rather than staged
in threadgroup memory. At the useful `HALF=512` the table is ~60 KB, past an
Apple GPU's ~32 KB threadgroup allocation, so staging is not possible at the
sizes that matter. Staging is a bandwidth optimisation, not a correctness
requirement — every thread reads the same immutable table — so it is left as
future tuning.

## The trust boundary is identical

No secret ever reaches the GPU on either backend. `honion-gpu` does not depend
on `honion-keyfile`; the device receives only public points and host-built
integer tables, and returns `(thread, offset)` claims the host re-derives and
re-verifies before anything is written. The Metal backend changes none of this —
see [docs/05](05-security-model.md).

## Performance

Measured on an Apple M4 Max, the Metal backend reaches about **0.09 × 10⁹
candidate addresses per second** end to end, rising slightly with `HALF` as the
single batch inversion amortises across more candidates. That is roughly 130×
below the RTX PRO 6000 CUDA figure of 12.5 × 10⁹/s — the difference is core
count (an M4 Max GPU is tens of cores against the Blackwell's 24 064), not
method. Reproduce the sweep the figure comes from with:

```
HALF=512 THREADS=262144 ITERS=8192 REPS=2 \
  cargo run --release -p honion-gpu --features metal --example bench_metal
```

Those are the parameters the CSV records. The example's bare defaults are
`HALF=512 ITERS=4096`, a different and shorter measurement, so its output is not
comparable to the figures above. As with every measurement in this project, run
it on an otherwise idle machine.

The raw sweep is in [`bench/results-metal-m4max.csv`](../bench/results-metal-m4max.csv);
the method is in [`bench/README.md`](../bench/README.md). These numbers are
measured, never asserted — the same discipline as the rest of the project
([docs/07](07-benchmarks.md)).

Note that `honion estimate`'s default rate is the CUDA-measured figure, so its
time estimates are optimistic on Apple Silicon by the ratio above until a
Metal-measured rate is wired in. The live progress line during a real search
reports the true rate regardless.
