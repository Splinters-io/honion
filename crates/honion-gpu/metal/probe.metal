// A trivial compute kernel that exercises the Metal driver (msl.rs) end to end.
//
// It has nothing to do with the vanity search — the real field, curve and
// search kernels arrive in Waves 3-5. Its only job is to give the driver
// something to compile, bind a `storageModeShared` buffer to, dispatch, and
// read back, so the runtime-compile / pipeline / buffer / launch substrate can
// be tested on its own before any arithmetic depends on it.
//
// It deliberately touches every mechanism the driver must get right:
//   * a closed `#include` (probe_common.metal), so the include resolver is
//     exercised, not just asserted;
//   * a `-D`-injected compile constant (PROBE_BASE), so define injection is
//     proven the same way `HALF` will be for the search kernel;
//   * a bounds check against a thread count passed as a buffer, so an
//     over-dispatched grid cannot write out of range.

#include "probe_common.metal"

#include <metal_stdlib>
using namespace metal;

// Overridable at compile time, exactly as HALF will be for the search kernel.
// The default exists so the source compiles stand-alone; the driver always
// pins it explicitly.
#ifndef PROBE_BASE
#define PROBE_BASE 0u
#endif

// out[i] = i + PROBE_BASE + PROBE_TAG, for i < count.
//
// Every term is observable from the host: the index proves per-thread
// dispatch, PROBE_BASE proves define injection, PROBE_TAG proves the header
// was included.
kernel void probe_fill(device uint *out          [[buffer(0)]],
                       constant uint &count      [[buffer(1)]],
                       uint gid                  [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    out[gid] = gid + PROBE_BASE + PROBE_TAG;
}
