//! The Metal runtime driver compiles, dispatches, and reads back correctly,
//! and closes its include set.
//!
//! Runs only under the `metal` feature (see Cargo.toml). The include-set tests
//! need no GPU and always run; the compile/dispatch tests skip cleanly with a
//! printed note when no Metal device is present, matching how the CUDA
//! integration tests skip without a card.

// @decision DEC-METAL-005 (test side)
// @title The driver is proven on a trivial kernel before any arithmetic depends on it
// @status accepted
// @rationale Waves 3-5 port ~1,300 lines of field/curve/search MSL onto this
//   driver. If the driver itself were unproven, a failure there would be
//   indistinguishable from a porting bug in that arithmetic. The probe kernel
//   lets the substrate — runtime compile, define injection, closed includes,
//   shared-buffer readback, bounds-checked dispatch — be verified in isolation,
//   so later waves can trust it. The include-set assertions run without a GPU
//   because the langsec boundary is a host-side property, not a device one.

use honion_gpu::msl::{self, MslError, MslKernel};

/// PROBE_TAG from metal/probe_common.metal — the value the header contributes.
const PROBE_TAG: u32 = 0x1000;

fn skip_without_device(err: &MslError) -> bool {
    matches!(err, MslError::NoDevice)
}

#[test]
fn expand_includes_resolves_the_closed_set() {
    // probe.metal includes probe_common.metal (quoted) and metal_stdlib
    // (angle-bracket). The quoted one is resolved and its contents inlined; the
    // angle-bracket one is passed through untouched for the Metal compiler.
    let expanded = msl::expand_includes(msl::sources::PROBE).expect("probe includes resolve");
    assert!(
        expanded.contains("PROBE_TAG"),
        "the quoted header's contents should be inlined"
    );
    assert!(
        !expanded.contains("#include \"probe_common.metal\""),
        "the quoted include line should be replaced, not left in place"
    );
    assert!(
        expanded.contains("#include <metal_stdlib>"),
        "angle-bracket system includes are left for the Metal compiler"
    );
}

#[test]
fn an_unknown_quoted_include_is_rejected() {
    let source = "#include \"not_a_real_header.metal\"\nkernel void k() {}\n";
    match msl::expand_includes(source) {
        Err(MslError::UnknownInclude { name }) => assert_eq!(name, "not_a_real_header.metal"),
        other => panic!("expected UnknownInclude, got {other:?}"),
    }
}

#[test]
fn each_header_is_emitted_at_most_once() {
    // Two includes of the same header must not double-define its contents.
    let source = "#include \"probe_common.metal\"\n#include \"probe_common.metal\"\n";
    let expanded = msl::expand_includes(source).expect("resolves");
    assert_eq!(
        expanded.matches("PROBE_TAG").count(),
        1,
        "a header reached twice should still be emitted once"
    );
}

#[test]
fn compile_dispatch_and_readback() {
    const BASE: u32 = 500;
    let kernel = match MslKernel::compile(
        msl::sources::PROBE,
        "probe_fill",
        &[("PROBE_BASE", BASE.to_string())],
    ) {
        Ok(k) => k,
        Err(e) if skip_without_device(&e) => {
            eprintln!("skipping: no Metal device present");
            return;
        }
        Err(e) => panic!("probe kernel failed to build: {e}"),
    };

    let n: usize = 4096;
    let tg = kernel.max_threads_per_threadgroup().min(256);

    // Not `mut`: the GPU writes this through shared memory, not Rust.
    let out = kernel
        .new_shared_buffer(n * std::mem::size_of::<u32>())
        .expect("output buffer");
    let mut count = kernel
        .new_shared_buffer(std::mem::size_of::<u32>())
        .expect("count buffer");
    count.as_mut_slice::<u32>()[0] = n as u32;

    kernel
        .dispatch(n, tg, &[&out, &count])
        .expect("dispatch succeeds");

    // out[i] == i + BASE + PROBE_TAG proves per-thread dispatch (i), define
    // injection (BASE), and that the header was included (PROBE_TAG).
    let got = out.as_slice::<u32>();
    for (i, &v) in got.iter().enumerate() {
        assert_eq!(v, i as u32 + BASE + PROBE_TAG, "mismatch at index {i}");
    }
}

#[test]
fn dispatch_bounds_check_leaves_the_tail_untouched() {
    let kernel = match MslKernel::compile(msl::sources::PROBE, "probe_fill", &[]) {
        Ok(k) => k,
        Err(e) if skip_without_device(&e) => {
            eprintln!("skipping: no Metal device present");
            return;
        }
        Err(e) => panic!("probe kernel failed to build: {e}"),
    };

    // Allocate more slots than we tell the kernel to fill; the kernel's own
    // `gid >= count` guard must leave the tail at its zero-initialised value.
    let slots: usize = 4096;
    let filled: usize = 1000;
    let tg = kernel.max_threads_per_threadgroup().min(256);

    // Not `mut`: the GPU writes this through shared memory, not Rust.
    let out = kernel
        .new_shared_buffer(slots * std::mem::size_of::<u32>())
        .expect("output buffer");
    let mut count = kernel
        .new_shared_buffer(std::mem::size_of::<u32>())
        .expect("count buffer");
    count.as_mut_slice::<u32>()[0] = filled as u32;

    // Dispatch the full slot count so threads beyond `filled` really run and
    // must be stopped by the in-kernel bound, not by an undersized grid.
    kernel.dispatch(slots, tg, &[&out, &count]).expect("dispatch");

    let got = out.as_slice::<u32>();
    assert_eq!(got[filled - 1], (filled - 1) as u32 + PROBE_TAG, "last filled slot");
    assert!(
        got[filled..].iter().all(|&v| v == 0),
        "slots past the count must stay zero (buffer was zero-initialised)"
    );
}

#[test]
fn a_zero_width_threadgroup_is_rejected() {
    let kernel = match MslKernel::compile(msl::sources::PROBE, "probe_fill", &[]) {
        Ok(k) => k,
        Err(e) if skip_without_device(&e) => {
            eprintln!("skipping: no Metal device present");
            return;
        }
        Err(e) => panic!("build: {e}"),
    };
    let out = kernel.new_shared_buffer(16).expect("buffer");
    match kernel.dispatch(16, 0, &[&out]) {
        Err(MslError::BadParameter(_)) => {}
        other => panic!("expected BadParameter for zero threadgroup, got {other:?}"),
    }
}
