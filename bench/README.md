# Benchmark harness

Reproduces the comparison in [`../docs/07-benchmarks.md`](../docs/07-benchmarks.md).

```
T=90 REPS=100 ./study.sh              # 100 interleaved rounds, writes results.csv
./analyse.py results-2026-08-20-n100.csv
```

Rounds are interleaved (mkp224o, honion, prefix32, mkp224o, ...) rather than
grouped by tool, so drift in machine conditions is spread across all three
rather than landing on whichever one happens to be running.

`study.sh` compares against `mkp224o` and `prefix32`, which are not vendored
here: build them, then point the script at them. Defaults resolve against this
checkout (`bench/tools/` for the comparison tools, `target/release/honion` for
honion), and `TOOLS`, `MK`, `P32`, `HO`, `CSV`, `T`, `REPS` and `RAPL` all
override from the environment:

```
TOOLS=/path/to/comparison-tools ./study.sh
```

It refuses to start if any of the three binaries is missing, rather than
measuring a tool that is not there — that would find no keys and write a rate of
zero, which is indistinguishable from a real measurement once it is in the CSV.
It needs passwordless `sudo` to read the CPU package RAPL energy counter, and
writes tool output to `/dev/shm` so the filesystem cannot become the bottleneck.

`results-2026-08-20-n100.csv` is the raw per-run data behind the published
figures: 300 rows, plus an idle-power baseline row.

## The measurement

No tool's own counter is trusted — "rate of what" is not defined identically
across implementations. Each run searches a prefix of known difficulty `D` bits
for `T` seconds, and the key directories *actually written* are counted:

```
throughput = hits * 2^D / T
```

Difficulty differs per tool only so that each produces enough hits for a tight
estimate; the formula normalises it out.

## Pitfalls

Both of these produced wrong numbers here before they were understood.

**Rebuild the whole workspace before measuring.** `cargo build -p honion-gpu`
builds the library only and leaves `target/release/honion` stale. Measuring the
CLI after changing the kernel that way times the *previous* build — which once
looked exactly like a 25% regression and took a while to attribute.

```
cargo build --release --workspace     # not -p honion-gpu
```

**Do not touch the machine while a study runs.** A compile started during one
tool's phase raised CPU draw from 87 W to 106 W and pulled one run 7% low. That
is why rounds are interleaved rather than grouped: grouping puts any such drift
entirely onto whichever tool happens to be running.

**Single runs are noisy.** At a 30-bit prefix a 90-second run finds around a
thousand keys, so Poisson alone is about ±3%. A single run once read 3.7σ above
the distribution of ten later runs and was briefly believed. Take medians, and
interleave anything being compared.

## The Metal backend

`study.sh` above compares CPU generators against the CUDA build on an NVIDIA
card. The Metal (Apple Silicon) backend is measured separately, because the
comparison tools it would sit beside do not run on that hardware and the numbers
are two orders of magnitude apart. The self-contained throughput sweep is:

```
for half in 256 512 1024; do
  HALF=$half THREADS=262144 ITERS=8192 REPS=2 \
    cargo run --release -p honion-gpu --features metal --example bench_metal
done
```

Those are the parameters `results-metal-m4max.csv` records. The example's bare
defaults are `HALF=512 ITERS=4096` — a shorter measurement that is not
comparable to the CSV.

Run it on an otherwise idle machine, and treat that as a hard requirement rather
than tidiness. Re-running this sweep on a busy M4 Max measured 0.035 G addr/s
where the same binary on the same machine, idle, measured 0.108 — a threefold
spread from host load alone, since the GPU shares its power and memory-bandwidth
budget with the CPU. Throughput also decays across back-to-back reps as the part
heats, so a rep taken late in a long session reads lower than the same rep taken
first. Both effects are far larger than the differences between `HALF` settings
the sweep is trying to resolve; interleave the configurations rather than running
each to completion in turn if you need to compare them.

It warms up, then times back-to-back launches over a 12-character (unfindable)
pattern, so the rate is pure search throughput undisturbed by hits — the same
measured-not-asserted discipline as the study above, just against the host's own
`examined` count rather than keys written.

`results-metal-m4max.csv` is the raw data behind the Metal figures in
[`../docs/06-performance.md`](../docs/06-performance.md): an Apple M4 Max reaches
about **0.09 G addr/s** end to end, rising slightly with `HALF` as the single
batch inversion amortises further. That is roughly 130× below the RTX PRO 6000
CUDA figure — the difference is core count (an M4 Max GPU is tens of cores to the
Blackwell's 24 064), not method. The MSL kernel compiles in about half a second
and stays bounded across `HALF`, which is the evidence that the unrolling trap
documented in `docs/06` is neutralised on Metal's compiler too.
