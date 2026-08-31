// Shared definitions for the driver-probe kernel.
//
// This header exists so the MSL driver has a real closed-include edge to
// resolve: `probe.metal` includes it by name, `msl::known_headers` is the only
// place that name resolves, and an include of anything else is a hard error.
// It is not part of the search kernel; Waves 3-5 add the real field, curve and
// search sources alongside it.

#ifndef HONION_PROBE_COMMON
#define HONION_PROBE_COMMON

// Folded into every value the probe writes, so a test can prove the header was
// actually included and its contents reached the compiler — not silently
// dropped.
constant uint PROBE_TAG = 0x1000u;

#endif
