# honion vs mkp224o vs prefix32 — a measured comparison

## Results

One hundred rounds, interleaved, 90 seconds per tool per round — 300 runs, about
seven and a half hours, on an otherwise idle machine.

| tool | device | mean addr/s | sd | 95% CI | min | max | vs mkp224o |
|---|---|---:|---:|---:|---:|---:|---:|
| **honion** | GPU | **1.2514 × 10¹⁰** | 4.12 × 10⁸ | ±0.64% | 1.149 × 10¹⁰ | 1.345 × 10¹⁰ | **48.5×** |
| prefix32 | GPU | 1.0492 × 10¹⁰ | 3.41 × 10⁸ | ±0.64% | 9.664 × 10⁹ | 1.125 × 10¹⁰ | 40.6× |
| mkp224o | CPU, 48 threads | 2.5812 × 10⁸ | 9.76 × 10⁶ | ±0.74% | 2.345 × 10⁸ | 2.807 × 10⁸ | 1.0× |

**honion is 1.193× prefix32** and 48.5× mkp224o. With intervals of ±0.64% against
a 19.3% margin, the ordering is not in question — a hundred rounds is far more
than the result needed, and was run to remove any doubt rather than to resolve
any.

Run-to-run spread is 3.3% for honion, 3.3% for prefix32 and 3.8% for mkp224o.

### Why interleaved

Rounds cycle mkp224o → honion → prefix32 rather than running each tool's hundred
in a block. Grouped phases mean any drift in machine conditions lands entirely
on whichever tool happens to be running. That is not hypothetical: an earlier
grouped run of this benchmark was contaminated when a compile was started during
prefix32's phase, visible as CPU draw rising from 87 W to 106 W and one run
coming in 7% low. Interleaving spreads such drift across all three.

### Energy

Sampled every two seconds, with a 30-second idle baseline of 24.9 W GPU /
58.3 W CPU subtracted.

| tool | GPU W | CPU W | W over idle | addr/s per watt | vs mkp224o |
|---|---:|---:|---:|---:|---:|
| **honion** | 589.4 | 111.2 | 617.4 | **2.027 × 10⁷** | **30.6×** |
| prefix32 | 590.9 | 87.6 | 595.3 | 1.763 × 10⁷ | 26.6× |
| mkp224o | 133.9 | 338.6 | 389.3 | 6.630 × 10⁵ | 1.0× |

honion is both the fastest and the most efficient per address, despite drawing
the most power. Its higher CPU draw than prefix32 is the host-side key
derivation, which it does for every thread every launch and prefix32 does not.

One artifact worth naming: mkp224o's GPU figure of 133.9 W is not work it is
doing. Interleaving means it runs immediately after a GPU tool, and the card has
not finished clocking down while it is being sampled. That inflates mkp224o's
"over idle" and therefore *understates* its efficiency. The effect is confined
to this table.

### What it means in practice

Expected time to find a prefix, at each tool's measured mean rate:

| characters | honion | prefix32 | mkp224o |
|---|---|---|---|
| 7 | 2.7 s | 3.3 s | 2.2 min |
| 8 | 1.5 min | 1.7 min | 1.2 hours |
| 9 | 47 min | 56 min | 1.6 days |
| 10 | **25 hours** | 30 hours | 51 days |
| 11 | 33 days | 40 days | 4.4 years |
| 12 | 2.9 years | 3.5 years | 142 years |

Ten characters is a day on the fastest GPU tool and seven weeks on the CPU.
Eleven is out of reach for all three on one machine.

These are means of a memoryless process, not deadlines: there is a 63% chance of
a result by the expected time, 86% by twice it, and 95% by three times.

### Scope of the claim

This is one operating point: a 30-bit prefix, on one card, with each tool at its
own tuned settings. It is not a claim about other GPUs, other prefix lengths, or
multi-pattern searches. honion's default configuration uses 17.2 GB of device
memory, which a smaller card cannot provide — the thread count scales down
automatically, but the ranking on such a card has not been measured.

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

One hundred interleaved rounds of 90 seconds per tool, run sequentially with no overlap, on an
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
T=90 REPS=100 ./study.sh                  # 100 interleaved rounds
./analyse.py results-2026-08-20-n100.csv   # the tables above, regenerated
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

- honion: 1.2514 × 10¹⁰ addr/s — 48.5× mkp224o, 1.193× prefix32
- prefix32: 1.0492 × 10¹⁰ addr/s — 40.6× mkp224o
- mkp224o: 2.5812 × 10⁸ addr/s

Raw per-run data for all 300 runs, and the harness that produced it, are in
[`../bench/`](../bench/).

The path from 0.66× prefix32 to 1.193× is documented in
[02-gpu-architecture.md](02-gpu-architecture.md): four changes, each addressing a
different bottleneck — work per candidate, 64-bit emulation, memory bandwidth,
and dependency-chain length — plus five attempts that were measured and
discarded.
