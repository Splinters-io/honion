# Benchmark harness

Reproduces the comparison in [`../docs/07-benchmarks.md`](../docs/07-benchmarks.md).

```
T=90 REPS=100 ./study.sh              # 100 interleaved rounds, writes results.csv
./analyse.py results-2026-08-20-n100.csv
```

Rounds are interleaved (mkp224o, honion, prefix32, mkp224o, ...) rather than
grouped by tool, so drift in machine conditions is spread across all three
rather than landing on whichever one happens to be running.

`study.sh` expects `mkp224o` and `prefix32` checked out and built beside it;
edit the paths at the top. It needs passwordless `sudo` to read the CPU package
RAPL energy counter, and writes tool output to `/dev/shm` so the filesystem
cannot become the bottleneck.

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
