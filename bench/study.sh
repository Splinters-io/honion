#!/usr/bin/env bash
# honion vs mkp224o: 10-run throughput and efficiency study.
#
# Methodology
# -----------
# Neither tool's internal counter is trusted. Each run searches for a filter of
# known difficulty D bits for T seconds and the keys ACTUALLY WRITTEN are
# counted; throughput is then  hits * 2^D / elapsed.  This measures the only
# quantity a user cares about -- how fast usable results appear -- and is immune
# to the two tools defining "attempt" differently.
#
# Difficulty differs per tool (25 bits for mkp224o, 30 for honion) purely so
# that both produce enough hits for a tight estimate. The formula normalises
# difficulty out, so this does not bias the comparison.
#
# Residual bias, AGAINST honion: `timeout` kills it mid-launch and that
# launch's work is discarded (expected loss ~2s of 90s, about 2%). mkp224o
# writes each key as it is found and loses nothing.
set -u
S=/tmp/claude-1000/-home-carroll-honion/f014c0db-6a57-48fe-b1ba-2bd618bddc06/scratchpad
MK=$S/mkp224o/mkp224o
HO=/home/carroll/honion/target/release/honion
P32=$S/Prefix32/target/release/prefix32
T=${T:-90}
REPS=${REPS:-10}
CSV=$S/results.csv
RAPL=/sys/class/powercap/intel-rapl:0

echo "tool,rep,filter,bits,hits,seconds,addr_per_sec,gpu_watts,cpu_watts" > $CSV

rapl_uj() { sudo -n cat $RAPL/energy_uj 2>/dev/null || echo 0; }
RAPL_MAX=$(sudo -n cat $RAPL/max_energy_range_uj 2>/dev/null || echo 0)

# Sample power for the duration of a background job, printing "gpu_w cpu_w".
sample_power() {
  local pid=$1 gsum=0 gn=0 e_prev e_now d cpu_j=0 t0 t1
  e_prev=$(rapl_uj); t0=$(date +%s.%N)
  while kill -0 $pid 2>/dev/null; do
    sleep 2
    g=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "${g:-}" ] && { gsum=$(awk -v a=$gsum -v b=$g 'BEGIN{print a+b}'); gn=$((gn+1)); }
    e_now=$(rapl_uj)
    # RAPL counters wrap; fold the wrap rather than emitting a negative delta.
    d=$(awk -v a="$e_prev" -v b="$e_now" -v m="$RAPL_MAX" 'BEGIN{d=b-a; if(d<0) d+=m; print d}')
    cpu_j=$(awk -v a="$cpu_j" -v b="$d" 'BEGIN{print a+b/1000000}')
    e_prev=$e_now
  done
  t1=$(date +%s.%N)
  awk -v gs=$gsum -v gn=$gn -v cj=$cpu_j -v a=$t0 -v b=$t1 \
    'BEGIN{printf "%.1f %.1f", (gn?gs/gn:0), cj/(b-a)}'
}

run_one() {  # tool filter bits rep
  local tool=$1 filt=$2 bits=$3 rep=$4 out=/dev/shm/study_$$ hits pw
  rm -rf $out; mkdir -p $out
  case $tool in
    mkp224o) timeout $T $MK -t 48 -x -q -d $out "$filt" >/dev/null 2>&1 & ;;
    honion)  timeout $T $HO search --prefix "$filt" --out $out --count 0 -q >/dev/null 2>&1 & ;;
    # prefix32 writes into the working directory, so it is run from there.
    prefix32) ( cd $out && timeout $T $P32 --gpu --no-print "$filt" >/dev/null 2>&1 ) & ;;
  esac
  local pid=$!
  pw=$(sample_power $pid)
  wait $pid 2>/dev/null
  hits=$(ls $out 2>/dev/null | wc -l); rm -rf $out
  local gw=$(echo $pw | cut -d' ' -f1) cw=$(echo $pw | cut -d' ' -f2)
  local rate=$(awk -v h=$hits -v t=$T -v b=$bits 'BEGIN{printf "%.6e", h*2^b/t}')
  echo "$tool,$rep,$filt,$bits,$hits,$T,$rate,$gw,$cw" | tee -a $CSV
}

echo "### idle baseline (30s)"
sleep 1 & sample_power $! >/dev/null
(sleep 30) & IDLE=$(sample_power $!); echo "idle: gpu=$(echo $IDLE|cut -d' ' -f1)W cpu=$(echo $IDLE|cut -d' ' -f2)W"
echo "idle,0,-,0,0,30,0,$(echo $IDLE|cut -d' ' -f1),$(echo $IDLE|cut -d' ' -f2)" >> $CSV

echo "### interleaved: $REPS rounds of (mkp224o, honion, prefix32) x ${T}s each"
for r in $(seq 1 $REPS); do
  echo "--- round $r/$REPS ---"
  run_one mkp224o  hon2o  25 $r
  run_one honion   hon2on 30 $r
  run_one prefix32 hon2on 30 $r
done
echo "### DONE"
