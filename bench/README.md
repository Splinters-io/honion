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
