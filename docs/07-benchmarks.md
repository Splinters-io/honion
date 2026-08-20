# honion vs mkp224o vs prefix32 — a measured comparison

## Results

Ten runs per tool, 90 seconds each, on one RTX PRO 6000 Blackwell and one
Threadripper 9960X. Throughput is inferred from keys actually written, not from
any tool's own counter.

| tool | device | mean addr/s | sd | 95% CI | vs mkp224o |
|---|---|---:|---:|---:|---:|
| **prefix32** | GPU | **1.053 × 10¹⁰** | 2.7 × 10⁸ | ±1.7 × 10⁸ | **41.1×** |
| **honion** | GPU | **6.932 × 10⁹** | 2.4 × 10⁸ | ±1.5 × 10⁸ | **27.1×** |
| **mkp224o** | CPU, 48 threads | **2.560 × 10⁸** | 1.2 × 10⁷ | ±7.5 × 10⁶ | 1.0× |

Run-to-run spread is under 4% for every tool, so the ordering is not in doubt.

- honion is **27× faster than mkp224o**, the established CPU generator, with
  mkp224o given its fastest backend on this machine.
- honion is **0.66× prefix32** — prefix32 is about 1.5× ahead.

### Energy

Sampled every two seconds; "over idle" subtracts a 30-second baseline of
25.2 W GPU / 62.1 W CPU.

| tool | GPU W | CPU W | W over idle | addr/s per watt |
|---|---:|---:|---:|---:|
| prefix32 | 590.8 | 87.5 | 591.0 | 1.78 × 10⁷ |
| honion | 572.6 | 92.1 | 577.3 | 1.20 × 10⁷ |
| mkp224o | 24.9 | 336.9 | 274.5 | 9.33 × 10⁵ |

Both GPU tools draw roughly the same power, so the efficiency ordering matches
the throughput ordering. Against the CPU the gap is wider than raw throughput
suggests: honion is 27× faster but **13× more energy-efficient per address**,
because it draws about twice the power to do it.

### What this means in practice

Expected time to find a prefix, at each tool's measured mean rate:

| characters | prefix32 | honion | mkp224o |
|---|---|---|---|
| 7 | 3.3 s | 5.0 s | 2.2 min |
| 8 | 1.7 min | 2.6 min | 1.2 hours |
| 9 | 56 min | 1.4 hours | 1.6 days |
| 10 | 1.2 days | 1.9 days | 51 days |
| 11 | 40 days | 60 days | 4.5 years |
| 12 | 3.5 years | 5.3 years | 143 years |

Ten characters is a weekend on either GPU and two months on the CPU. Eleven is
out of reach for all three on one machine.

Remember these are means of a memoryless process, not deadlines: there is a 63%
chance of a result by the expected time and 95% by three times it.

### End-to-end versus kernel throughput

The figures above are end-to-end: what a user gets, including startup, host
work, and writing keys. honion's kernel alone benchmarks at 8.4 × 10⁹ addr/s, so
about 13% is lost outside it. That divides into:

- ~2.4% drawing fresh secret scalars and deriving their public points each
  launch — honion re-randomises every launch rather than continuing a walk, so
  each launch is an independent sample;
- ~2% because `timeout` kills a launch in progress and its work is discarded —
  an artifact of measuring for a fixed duration, which a real search never does;
- the remainder in startup, and in verifying and writing each key.

prefix32 loses about 3.5% between its kernel and its end-to-end rate, because it
does neither per-launch re-randomisation nor per-key verification. That is a
genuine design difference rather than an implementation gap, and it is counted
against honion here because end-to-end is the number that actually matters.



## What was compared

| tool | version | device | approach |
|---|---|---|---|
| [`mkp224o`](https://github.com/cathugger/mkp224o) | commit `5172c0f` (2024-02-15) | CPU, 48 threads | C, `amd64-64-24k` asm backend |
| [`prefix32`](https://github.com/0xROOTPLS/Prefix32) | v2.2.0, commit `ab1555f` (2026-08-07) | GPU, OpenCL | Rust host, auto-tuned OpenCL kernel |
| `honion` | this repository | GPU, CUDA/NVRTC | Rust host, CUDA kernel |

`eschalot` and similar are excluded: they generate v2 (RSA-1024) addresses, which
Tor removed in 2021, so they solve a different problem.

Both competitors were given their best configuration. For mkp224o that meant
building all five arithmetic backends and benchmarking each; `amd64-64-24k` won
at 2.58 × 10⁸/s, 21% ahead of `amd64-51-30k`, and is what the results below use.
For prefix32 it meant letting its GPU auto-tuner pick batch and work-group size
before measuring. Benchmarking a competitor's slow configuration proves nothing.

## Hardware

- **GPU** NVIDIA RTX PRO 6000 Blackwell Workstation Edition — sm_120, 188 SMs,
  24 064 CUDA cores, 96 GB, 3.09 GHz max SM clock. Driver 595.84.
- **CPU** AMD Ryzen Threadripper 9960X — 24 cores / 48 threads, 5.49 GHz max.
- **OS** Ubuntu 26.04, CUDA 13.1 (via NVRTC), OpenCL via the NVIDIA ICD.

Both GPU tools ran on the same card, which is the only way a GPU-to-GPU
comparison means anything. prefix32's published figures are from an RX 6800 XT
and an RTX 5060 Ti and are not comparable to these.

## Method

**No tool's own counter is trusted.** Each tool reports a rate, but "rate of
what" is not defined identically across implementations — one may count
candidate points generated, another candidates that survive a filter, another
scalar multiplications. Comparing those numbers directly would be meaningless.

Instead each run searches for a prefix of known difficulty *D* bits for *T*
seconds, and the key directories **actually written to disk** are counted:

```
throughput = hits × 2^D / T
```

This measures the only quantity a user cares about — how fast usable results
appear — and cannot be gamed by differing definitions of an attempt. It is also
self-validating: a tool that inflated its internal counter, or that produced
malformed keys, would show up here as a low hit count.

Difficulty differs per tool (25 bits for mkp224o, 30 for the GPU tools) purely
so that each produces enough hits for a tight estimate. The formula normalises
difficulty out, so this does not bias the comparison.

Ten runs of 90 seconds each per tool, run sequentially with no overlap, on an
otherwise idle machine. Output went to `/dev/shm` so that filesystem writes
could not become the bottleneck. Power was sampled every 2 seconds from
`nvidia-smi` and the CPU package RAPL counter, with a 30-second idle baseline
taken first.

### Known bias

`timeout` kills honion mid-launch and that launch's work is discarded — an
expected loss of about 2 seconds in 90, roughly 2%. mkp224o and prefix32 write
each key as it is found and lose nothing. **The bias therefore runs against
honion**, so its true rate is slightly higher than reported here. Being wrong
in the direction unfavourable to one's own tool is the safe way to be wrong.

### Reproducing

```bash
# mkp224o
./mkp224o -t 48 -x -q -d OUT hon2o          # 25-bit filter
# prefix32
prefix32 --gpu --no-print hon2on            # 30-bit prefix
# honion
honion search --prefix hon2on --out OUT --count 0 -q
```

Run each for a fixed time, count the directories produced, and apply the
formula above.

The harness and the raw per-run data are in [`../bench/`](../bench/):

```bash
cd bench
./study.sh                              # 10 runs x 90s per tool
./analyse.py results-2026-08-19.csv     # the tables above, regenerated
```

## What the first benchmark found, and what was done about it

The first run of this benchmark was uncomfortable: honion was **2.3× slower than
prefix32**. mkp224o was far behind both, but being beaten by the other GPU tool
on the same card was the finding that mattered.

The cause turned out to be algorithmic, and it was almost exactly accounted for
by counting field multiplications per candidate.

**honion, as originally written**, walked one point forward at a time in
extended coordinates:

| step | multiplications |
|---|---|
| `ge_madd` — mixed addition of 8·B | 3 |
| `ge_p1p1_to_p3` — back to extended coordinates | 4 |
| Montgomery forward product | 1 |
| Montgomery backward pass | 3 |
| amortised inversion | ~1 |
| **total** | **12** |

**prefix32** keeps a table of precomputed offsets and derives candidates from a
single base point using the **dual addition law**, working with the affine *y*
coordinate as an unreduced fraction:

| step | multiplications |
|---|---|
| `x₁y₂` and `y₁x₂`, **shared between P+Q and P−Q** | 2 per *two* candidates → 1 |
| Montgomery forward fold | 2 |
| Montgomery backward pass | 2 |
| **total** | **~5** |

`12 / 5 = 2.4×` predicted against `2.3×` measured — close enough to call the
difference understood rather than guessed at.

Two ideas do the work, and honion has since adopted both:

1. **Affine-*y*-only addition.** For a twisted Edwards curve with `a = -1`,
   `y(P ± Q)` can be written with no reference to the curve constant `d` and no
   projective coordinates for the result. Since only *y* is ever needed, and the
   division is deferred to the batch inversion anyway, the entire
   4-multiplication `p1p1 → p3` conversion simply disappears.
2. **± symmetry.** `P+Q` and `P−Q` share the products `x₁y₂` and `y₁x₂`,
   differing only in an add versus a subtract. Two candidates for the price of
   the products of one.

### The result

Rewriting honion's inner loop around the dual addition law took it from
**4.9 to 8.4 G/s** — a 1.7× improvement, matching what the multiplication count
predicted. The formula was verified against the standard addition law over
random point pairs *before* any CUDA was written
([`cuda/verify_dual_law.py`](../cuda/verify_dual_law.py)), and the full
correctness suite — including exact hit-set equality against the host reference
— passes unchanged.

The rewrite also required a change visible in the output: because the dual law
produces `base + off` and `base − off` together, a match may now lie *below* the
scalar its thread started from. Reported offsets are therefore signed, and the
key-reconstruction path checks the clamping invariant at both ends of the range
rather than only the top.

### The gap that remains

honion is still behind prefix32, and the residue is **not** algorithmic — both
now do about the same number of multiplications per candidate. It is in the
field arithmetic underneath.

honion uses ten 25.5-bit limbs accumulated into 64-bit registers, and its
generated PTX is dominated by `add.s64` and `mad.lo.s64`, each of which costs two
32-bit operations on this hardware. prefix32 uses four 64-bit limbs. The
identified fix for honion is eight 32-bit limbs with `mad.lo.cc.u32` /
`madc.hi.cc.u32` carry chains, which would use the hardware carry flag instead of
synthesising it — estimated at roughly 1.4×, which would close the gap.

That is a rewrite of the most safety-critical file in the project, so it is
listed as identified and costed rather than attempted in a hurry.

### One hypothesis that was tested and disproved

Before the algorithmic cause was found, the suspicion was that serialising the
full 32-byte key per candidate (`fe_tobytes`, a 255-iteration bit-packing loop)
was a significant cost. Extracting only the 8 bytes the prefilter reads changed
throughput by less than 0.1% — the compiler was already eliminating the packing
of bytes the code never read. The refactor was kept because it states the intent
explicitly, but it is recorded as neutral rather than as a win. Measuring before
optimising would have saved the effort.

## Behavioural differences

Throughput is one axis. The three tools also differ in what they do around the
search. These are design choices with different costs, recorded here so a reader
can weigh them; the table states what each tool does, not what it should do.

| | mkp224o | prefix32 | honion |
|---|---|---|---|
| re-derives each key before writing | no | no | yes |
| key file permissions | `0700` / `0600` | inherits umask (observed `0755` / `0644`) | `0700` / `0600` |
| writes atomically | no | no | yes (temp file plus rename) |
| secret material in device memory | n/a (CPU) | yes | no |
| pattern syntax | filters; optional regex build | prefix with `?d` / `?l` | prefix with `?`, `[abc]`, `[^abc]` |
| rejects patterns it cannot search efficiently | no | no | yes |

All three produced valid keys in these runs. Keys written by mkp224o and by
prefix32 were both re-derived with honion's verifier and matched their own
hostname files; prefix32 additionally unit-tests its field arithmetic against
`curve25519-dalek`.

The re-derivation and atomic-write behaviour costs honion part of the gap
between its kernel rate and its end-to-end rate, quantified above. Whether that
trade is worth making depends on what the keys are for.

## Summary

- honion: 1.036 × 10^10 addr/s, 40.5× mkp224o
- prefix32: 1.053 × 10^10 addr/s, 41.1× mkp224o
- mkp224o: 2.560 × 10^8 addr/s

The two GPU implementations are within about 2% of each other, which is close to
the run-to-run spread of the measurement itself. Both are roughly forty times a
48-thread CPU build of mkp224o on this hardware.

Raw data for every run is in [`../bench/`](../bench/), along with the harness
that produced it.
