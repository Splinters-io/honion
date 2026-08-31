//! The backend-agnostic facade compiles and behaves the same under either
//! feature.
//!
//! Deliberately has no `required-features`: this file must build with `cuda`,
//! with `metal`, or with both, because it exercises exactly the names a
//! consumer like `honion-cli` is allowed to touch. Nothing here needs a
//! device — it checks the seam, not the kernel.

// @decision DEC-BACKEND-001 (test side)
// @title One feature-agnostic smoke test guards the facade surface
// @status accepted
// @rationale Every other integration test in this crate is gated behind a
//   backend feature, so nothing else would catch a regression where a shared
//   name (Searcher, DeviceInfo, candidates_per_batch) quietly stopped being
//   exported on one platform. This file compiles under any feature set the
//   crate accepts, which makes "the seam is intact" a CI property on both the
//   Linux and macOS jobs rather than an assumption.

use honion_gpu::{DEFAULT_HALF, DeviceInfo, Hit, candidates_per_batch};

#[test]
fn batch_math_is_backend_neutral() {
    // 2*half + 1: the base point plus `half` pairs either side of it.
    assert_eq!(candidates_per_batch(1), 3);
    assert_eq!(candidates_per_batch(DEFAULT_HALF), 2 * DEFAULT_HALF + 1);
}

#[test]
fn device_info_description_shapes() {
    let with_detail = DeviceInfo {
        backend: "CUDA",
        name: "Example GPU".into(),
        detail: "compute capability 12.0".into(),
    };
    assert_eq!(
        with_detail.description(),
        "Example GPU (CUDA, compute capability 12.0)"
    );

    let bare = DeviceInfo {
        backend: "Metal",
        name: "Apple M4 Max".into(),
        detail: String::new(),
    };
    assert_eq!(bare.description(), "Apple M4 Max (Metal)");

    // Display and description agree; the CLI uses Display.
    assert_eq!(format!("{bare}"), bare.description());

    let unknown = DeviceInfo::unknown();
    assert!(unknown.description().contains("unknown"));
}

#[test]
fn hit_layout_matches_the_device_struct() {
    // Four u32-sized fields, repr(C): the readback in collect() chunks by 4.
    assert_eq!(std::mem::size_of::<Hit>(), 16);
}

#[test]
fn searcher_type_is_nameable() {
    // The seam's whole point: this name resolves regardless of backend.
    fn _takes_searcher(_s: &honion_gpu::Searcher) {}
}
