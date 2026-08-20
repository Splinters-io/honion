# Benchmark harness

Reproduces the comparison in [`../docs/07-benchmarks.md`](../docs/07-benchmarks.md).

```
./study.sh                 # 10 runs x 90s per tool, writes results.csv
./analyse.py results.csv   # tables: throughput, energy, time-to-find
```

`study.sh` expects `mkp224o` and `prefix32` checked out and built beside it;
edit the paths at the top. It needs passwordless `sudo` to read the CPU package
RAPL energy counter, and writes tool output to `/dev/shm` so the filesystem
cannot become the bottleneck.

`results-2026-08-19.csv` is the raw per-run data behind the published figures:
one row per run, plus an idle-power baseline row.

## The measurement

No tool's own counter is trusted — "rate of what" is not defined identically
across implementations. Each run searches a prefix of known difficulty `D` bits
for `T` seconds, and the key directories *actually written* are counted:

```
throughput = hits * 2^D / T
```

Difficulty differs per tool only so that each produces enough hits for a tight
estimate; the formula normalises it out.
