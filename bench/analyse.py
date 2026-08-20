#!/usr/bin/env python3
"""Turn results.csv into the tables used by docs/07-benchmarks.md."""
import csv, statistics as st, sys, math

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
rows = [r for r in csv.DictReader(open(path))]
idle = next((r for r in rows if r["tool"] == "idle"), None)
idle_gpu = float(idle["gpu_watts"]) if idle else 0.0
idle_cpu = float(idle["cpu_watts"]) if idle else 0.0

tools = {}
for r in rows:
    if r["tool"] == "idle":
        continue
    tools.setdefault(r["tool"], []).append(r)

def stats(vals):
    n = len(vals)
    m = st.mean(vals)
    sd = st.stdev(vals) if n > 1 else 0.0
    # 95% CI on the mean, normal approximation.
    ci = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return n, m, sd, ci, min(vals), max(vals)

print(f"idle baseline: GPU {idle_gpu:.1f} W, CPU {idle_cpu:.1f} W\n")
print(f"{'tool':<10} {'n':>3} {'mean addr/s':>13} {'sd':>10} {'95% CI':>10} {'min':>12} {'max':>12}")
summary = {}
for tool, rs in tools.items():
    v = [float(r["addr_per_sec"]) for r in rs]
    n, m, sd, ci, lo, hi = stats(v)
    summary[tool] = dict(n=n, mean=m, sd=sd, ci=ci, lo=lo, hi=hi,
                         gpu=st.mean([float(r["gpu_watts"]) for r in rs]),
                         cpu=st.mean([float(r["cpu_watts"]) for r in rs]),
                         hits=sum(int(r["hits"]) for r in rs))
    print(f"{tool:<10} {n:>3} {m:>13.4e} {sd:>10.2e} {ci:>10.2e} {lo:>12.4e} {hi:>12.4e}")

print()
print(f"{'tool':<10} {'total hits':>11} {'GPU W':>8} {'CPU W':>8} {'W over idle':>12} {'addr/s/W':>12}")
for tool, s in summary.items():
    over = (s["gpu"] - idle_gpu) + (s["cpu"] - idle_cpu)
    perw = s["mean"] / over if over > 1 else float("nan")
    print(f"{tool:<10} {s['hits']:>11} {s['gpu']:>8.1f} {s['cpu']:>8.1f} {over:>12.1f} {perw:>12.3e}")

print("\nrelative throughput:")
names = sorted(summary, key=lambda t: -summary[t]["mean"])
top = summary[names[0]]["mean"]
for t in names:
    print(f"  {t:<10} {summary[t]['mean']/top:>6.3f}x of fastest   "
          f"{summary[t]['mean']/summary['mkp224o']['mean']:>7.1f}x mkp224o")

print("\nexpected time to find, by prefix length (mean rate):")
hdr = "  chars " + "".join(f"{t:>14}" for t in names)
print(hdr)
def human(s):
    if s < 1: return f"{s*1000:.0f}ms"
    if s < 90: return f"{s:.1f}s"
    if s < 5400: return f"{s/60:.1f}min"
    if s < 172800: return f"{s/3600:.1f}h"
    if s < 365.25*86400: return f"{s/86400:.1f}d"
    return f"{s/(365.25*86400):.1f}y"
for c in range(7, 13):
    line = f"  {c:>5} "
    for t in names:
        line += f"{human(32**c / summary[t]['mean']):>14}"
    print(line)
