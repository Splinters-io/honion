# MASTER_PLAN: honion

## Identity

**Type:** CLI + library workspace (GPU vanity-address generator)
**Languages:** Rust (host, primary), CUDA C++ / MSL (device), Python (device-code generators/verifiers)
**Root:** /Users/zephr/tools/honion
**Created:** 2026-08-25
**Last updated:** 2026-08-25

honion generates vanity Tor v3 (.onion) hidden-service addresses by brute-forcing Ed25519 keypairs whose base32 address begins with a user-specified pattern. The search runs on the GPU; the host holds all secret material, uploads only public points, and re-verifies every device claim before writing a key. Today the only search backend is NVIDIA CUDA (JIT-compiled at runtime via NVRTC through `cudarc`).

## Architecture

    crates/honion-core/     — pattern grammar, base32, address derivation, langsec recogniser (no GPU)
    crates/honion-gpu/      — device search kernel + host driver + differential tests (the backend layer)
    crates/honion-keyfile/  — SecretScalar, VerifiedKey, service-dir writing (holds secrets; GPU never depends on it)
    crates/honion-cli/      — `honion` binary: search / estimate / verify subcommands
    cuda/                   — device sources: fe25519(.u32).cuh, ge25519.cuh, search.cu, testkernels.cu + Python generators
    docs/                   — numbered design docs (00..07): address spec, curve math, GPU arch, langsec, security, perf, benchmarks
    bench/                  — measured-not-asserted benchmark harness (study.sh, analyse.py, CSV results)

## Original Intent

> This is `honion`, a Rust vanity .onion-address generator for Tor v3 hidden services. It currently has exactly one search backend: NVIDIA CUDA, JIT-compiled at runtime via NVRTC through the `cudarc` crate. Goal: design and plan a **second, sibling GPU backend using native Metal (via `metal-rs`), targeting Apple Silicon (specifically Apple M4 Max) on macOS** — no MLX, no Python, pure Rust — [...] This is an ADDITION (second backend), not a replacement — the CUDA/Linux path must keep working.

## Principles

These are the project's enduring design principles. They do not change between initiatives.

1. **No secret ever reaches the GPU** — Scalars are drawn host-side; only public points are uploaded. The device returns unverified `(thread, offset)` claims that the host re-derives and re-checks with `honion-core` before anything touches disk. A miscompiled, bit-flipped, or malicious kernel can waste time but can never emit a wrong key. `honion-gpu` must never depend on `honion-keyfile`.
2. **The device is not a parser (langsec rule 4)** — The GPU sees only fixed-size integers built on the host from an already-recognised pattern set. Device source is embedded at build time and its includes resolve against a closed list, so the text reaching the device compiler is fixed when the binary is built.
3. **Differential testing is the gate, not a formality** — Every arithmetic layer (field, curve, search) is tested against an independent implementation (`num-bigint`, `curve25519-dalek`) and a host reference matcher demanding exact set equality. A wrong coefficient runs at full speed finding nothing forever, so correctness cannot be assumed — it must be differentially demonstrated. New backends earn the same bar, not a reduced one.
4. **Every performance claim is a measured, reproducible number** — Throughput is proven by the `bench/` harness, never asserted. No G/s figure enters docs, README, or the estimator's default rate until it has been measured on the hardware it describes.
5. **Code is truth; capture decisions where they are made** — Significant decisions carry `@decision` annotations at the point of implementation. When device code and plan diverge on *how* (algorithm, compiler workaround), code wins and the annotation records why.

---

## Decision Log

Append-only record of significant decisions across all initiatives. This log persists across initiative boundaries — it is the project's institutional memory.

| Date | DEC-ID | Initiative | Decision | Rationale |
|------|--------|-----------|----------|-----------|
| 2026-08-25 | DEC-BACKEND-001 | metal-backend | Backend-agnostic `Searcher` facade with cfg/feature-selected backend modules | Keeps `honion-cli` and the public API surface stable while swapping the device layer; the Searcher is not in the per-candidate hot path, so the seam costs nothing at runtime. |
| 2026-08-25 | DEC-BACKEND-002 | metal-backend | `cuda` and `metal` as optional, non-default Cargo features; both device bindings optional deps | Lets the crate build on Linux+NVIDIA and macOS+Apple-Silicon without requiring the other platform's toolchain. `cudarc` stops being a hard dep (it cannot build on macOS today). |
| 2026-08-25 | DEC-BACKEND-003 | metal-backend | Use `objc2-metal` as the Rust Metal binding, not `metal-rs` | Research (2026-08-25): `metal-rs` is officially deprecated and its own repo recommends `objc2-metal`; the selectors are identical so porting guidance is binding-agnostic. **Overrides the originally-named crate — pending user confirmation (Alternatives Gate).** |
| 2026-08-25 | DEC-METAL-004 | metal-backend | Port the radix-2^25.5 10-limb field arithmetic (`fe25519.cuh`), not the 8x32 carry-chain variant (`fe25519_u32.cuh`) | MSL exposes no add-with-carry intrinsic and 64-bit int runs ~1/32-1/64 throughput on Apple GPUs. The carry-deferred reduced-radix representation maps onto Apple's int32 MAC datapath and needs no carry flag. |
| 2026-08-25 | DEC-METAL-005 | metal-backend | Runtime MSL compilation via `new_library_with_source`; preserve the closed-include property by host-side concatenation | Structural analog of the NVRTC path; keeps HALF/params baked in per run and needs no precompiled toolchain. Runtime MSL compile does not resolve user `#include` at all, so the langsec property holds — but keep explicit host concatenation so it is enforced by our code, not merely platform behaviour. |
| 2026-08-25 | DEC-METAL-006 | metal-backend | Independently rediscover the two device-compiler traps on Metal's compiler | The NVRTC `__noinline__` (cold-routine PTX bloat) and `#pragma unroll 1` (284s compile blowup) lessons do not transfer. MSL has `[[clang::noinline]]` and `#pragma clang loop unroll(disable)`, but Clang has historically ignored the unroll-disable pragma — effect must be verified in AIR / compile time, not assumed. |
| 2026-08-25 | DEC-METAL-007 | metal-backend | Use `storageModeShared` (unified-memory) buffers; keep the `set_start_points`/`launch_async`/`collect` API shape | Apple Silicon shares system memory, so there is no H2D/D2H copy — the PCIe-transfer-overlap machinery (docs/06 "Overlapping the host") is a discrete-GPU workaround and is not needed. Keeping the API shape means `honion-cli`'s launch loop is unchanged. |
| 2026-08-25 | DEC-BACKEND-008 | metal-backend | Replace `Searcher::compute_capability() -> (i32,i32)` with a backend-agnostic `DeviceInfo`/`device_description()` | Compute capability is a CUDA-only concept that currently leaks into CLI output. Metal reports a GPU family/name instead; the CLI prints a generic descriptor string. |
| 2026-08-27 | DEC-METAL-009 | metal-backend | Split a Metal launch into dispatches sized by a measured wall-clock target, resuming the walk through the point buffer | macOS kills a command buffer that holds the GPU too long (`kIOGPUCommandBufferCallbackErrorHang`), and one launch was one command buffer — reachable by ordinary use, since `--launch-seconds` defaults to 4. Batch counts cannot bound this because per-batch time is a property of the device, so the searcher measures its rate and chunks to `DISPATCH_TARGET_SECS`. The kernel writes back the point its thread reached and takes a `batch_base`, so reported offsets stay absolute and a split launch returns what a single dispatch returned. CUDA is deliberately unchanged: its Linux compute devices have no such watchdog, and DEC-BACKEND-001 keeps that file a verbatim move. |

---

## Active Initiatives

### Initiative: Native Metal GPU backend for Apple Silicon
**Status:** completed (2026-08-26 — all 7 waves merged; `honion search` runs end to end on an Apple M4 Max, ~0.09 G addr/s measured, differential suites at CUDA parity)
**Started:** 2026-08-25
**Goal:** Add a second, sibling GPU search backend using native Metal (MSL device code, pure-Rust host) targeting Apple Silicon (M4 Max), without regressing the existing CUDA/Linux path.

> honion's search is fast but NVIDIA-only: `cudarc` is a hard dependency and does not build on macOS. Apple Silicon has capable GPUs and unified memory, and a large audience runs macOS. This initiative makes `honion-gpu` backend-agnostic and adds a Metal backend that ports the ~1,300 lines of hand-written field/curve/search device arithmetic to MSL, earning the same differential-test and measured-benchmark bar the CUDA path already meets. The core safety property — no secret ever reaches the GPU — holds identically.

**Dominant Constraint:** correctness (the arithmetic must be provably identical to the CUDA path and to independent references; a silent field-arithmetic bug wastes hours finding nothing). Security (the trust boundary) is a hard invariant, not a trade-off; performance is measured, not targeted.

#### Goals
- REQ-GOAL-001: `honion search` runs end-to-end on an Apple M4 Max and writes a key whose `.onion` address matches the requested pattern and passes `honion verify`.
- REQ-GOAL-002: The Metal kernel passes differential field, group, and search-set-equality tests at exact parity with the CUDA suite (no reduced test bar).
- REQ-GOAL-003: The CUDA/Linux path continues to build and pass its full test suite unchanged; `honion-gpu` builds on macOS without a CUDA toolkit and on Linux without a Metal toolchain.
- REQ-GOAL-004: A reproducible `bench/`-discipline measurement produces real M4 Max throughput numbers before any Metal performance claim enters docs/README.
- REQ-GOAL-005: The "no secret reaches the GPU" trust boundary and the closed-include langsec property hold identically on the Metal backend.

#### Non-Goals
- REQ-NOGO-001: Beating or matching CUDA throughput — Apple's GPU core count is orders of magnitude below an RTX PRO 6000; no G/s target is promised, only a measurement.
- REQ-NOGO-002: MLX, wgpu/WGSL, or any Python in the device path — settled scope; raw MSL + pure-Rust host only.
- REQ-NOGO-003: A runtime backend selector (choosing CUDA vs Metal at run time in one binary) — the backend is chosen at build time by platform; cross-compiling both into one artifact is out of scope.
- REQ-NOGO-004: Replacing or deprecating the CUDA backend — this is a sibling addition; both are first-class.
- REQ-NOGO-005: Porting the NVRTC-specific tooling examples (`attrib.rs`, `ptxmix.rs`) to Metal — PTX attribution has no direct MSL analog; deferred to a P2.

#### Requirements

**Must-Have (P0)**

- REQ-P0-001: `honion-gpu` restructured so `cudarc` and the Metal binding are optional deps behind `cuda`/`metal` features; the crate builds with exactly one backend on the matching platform.
  Acceptance: Given a macOS host with no CUDA toolkit, When `cargo build -p honion-gpu --features metal` runs, Then it compiles; and `cargo build -p honion-gpu --features cuda` still compiles on Linux.
- REQ-P0-002: A backend-agnostic `Searcher` facade + `DeviceInfo` such that `honion-cli` compiles and runs against either backend with no CUDA-specific types in its source.
  Acceptance: Given the metal feature, When `honion search --prefix ...` runs, Then it uses the Metal `Searcher` and prints a generic device descriptor (no `compute_capability` leak).
- REQ-P0-003: A Metal host driver that compiles MSL at runtime (`new_library_with_source`), injects `HALF` as a compile constant, resolves the closed include set by host concatenation, builds a compute pipeline, and manages `storageModeShared` buffers.
  Acceptance: Given the embedded MSL sources, When the driver compiles the search kernel for a given HALF, Then an unknown `#include` is rejected before submission and a valid kernel yields a usable pipeline state.
- REQ-P0-004: MSL port of field arithmetic (`metal/fe25519.metal`, radix 2^25.5) passing the differential field suite against `num-bigint` on a live Metal device.
  Acceptance: Given random field elements, When each device routine runs on Metal, Then results equal the `num-bigint` reference for every operation (test skips cleanly with a message when no Metal device is present).
- REQ-P0-005: MSL port of Edwards-curve arithmetic (`metal/ge25519.metal`, dual addition law preserved) passing the differential group suite against `curve25519-dalek`.
  Acceptance: Given random points/scalars, When the device group ops run on Metal, Then `y(P±Q)` and the walk match `curve25519-dalek`.
- REQ-P0-006: MSL port of the search kernel (`metal/search.metal`: per-thread +8 scalar walk, prefilter, Montgomery-batched inversion, atomic hit reporting) passing the search-kernel set-equality test at exact agreement with the host reference matcher.
  Acceptance: Given a search space and pattern, When the Metal kernel runs, Then its hit set equals the host reference set exactly (no missing, no spurious hits) and `walk_matches_scalar_arithmetic` holds over the full visited sequence.
- REQ-P0-007: Both Metal compiler traps independently rediscovered and neutralised, with the effect verified (AIR dump / compile-time / code-size), and the workaround + numbers recorded via `@decision` and in docs.
  Acceptance: Given the search kernel, When compiled for Metal, Then compile time and kernel code size are bounded and stable across HALF values (no unroll/inline blowup), demonstrated by measurement.
- REQ-P0-008: A reproducible M4 Max benchmark (bench example + `bench/` harness path) reporting compile/JIT time and measured G/s, following the measured-not-asserted discipline.
  Acceptance: Given an M4 Max, When the Metal bench runs, Then it prints measured throughput and back-to-back launch decay, and the numbers are reproducible.
- REQ-P0-009: Docs/README updated so GPU requirements are no longer "NVIDIA-only", with a new Metal backend doc and measured numbers folded into perf/benchmark docs.
  Acceptance: Given the README, When a macOS user reads GPU requirements, Then Apple Silicon is documented as supported with the measured (not asserted) Metal figures.
- REQ-P0-010: CI builds and runs non-GPU tests on both a Linux (cuda) and a macOS (metal) job; GPU-requiring tests skip cleanly when no device is present.
  Acceptance: Given a push, When CI runs, Then the macOS job builds the metal feature and the Linux job builds the cuda feature, both green without a GPU attached.

**Nice-to-Have (P1)**

- REQ-P1-001: Compiled-pipeline caching analog to the NVRTC PTX disk cache (e.g. `MTLBinaryArchive`), so repeated runs skip recompilation.
- REQ-P1-002: Metal `auto_threads` sizing tuned from end-to-end measurement on unified memory (mirroring the CUDA cap-from-measurement approach), not just free-memory division.
- REQ-P1-003: A generated-vs-checked-in diff test for any Python-generated MSL (if a `gen_fe32.py` analog is introduced) mirroring the existing staleness gate.

**Future Consideration (P2)**

- REQ-P2-001: MSL instruction-attribution / disassembly tooling analog to `attrib.rs`/`ptxmix.rs` (AIR/GPU-assembly inspection) — design the driver so a debug-options path exists.
- REQ-P2-002: Broader Apple GPU family coverage (M1–M3, A-series) once M4 Max is validated.

#### Definition of Done

The Metal backend builds on macOS+Apple-Silicon behind the `metal` feature; the CUDA path is unchanged and green (REQ-GOAL-003, REQ-P0-001/010); the three differential suites pass at exact parity on a live Metal device (REQ-P0-004/005/006, REQ-GOAL-002); `honion search` produces a `honion verify`-passing key on an M4 Max (REQ-GOAL-001); both compiler traps are neutralised with verified effect (REQ-P0-007); a reproducible measured benchmark exists and its numbers — and only measured numbers — appear in updated docs/README (REQ-GOAL-004, REQ-P0-008/009); the trust boundary and closed-include property are demonstrably preserved (REQ-GOAL-005).

#### Architectural Decisions

- DEC-BACKEND-001: Backend-agnostic `Searcher` facade with feature-selected backend modules.
  Addresses: REQ-P0-002. Rationale: stable public API; the seam is outside the per-candidate hot path.
- DEC-BACKEND-002: `cuda`/`metal` optional non-default features; both device bindings optional deps.
  Addresses: REQ-P0-001, REQ-P0-010. Rationale: each platform builds with only its own toolchain; `cudarc` stops being a hard dep.
- DEC-BACKEND-003: Rust Metal binding = `objc2-metal` (not the deprecated `metal-rs`).
  Addresses: REQ-P0-003. Rationale: research shows metal-rs is deprecated and self-recommends objc2-metal; selectors identical. **Pending user confirmation.**
- DEC-METAL-004: Port the radix-2^25.5 field representation, not the 8x32 carry-chain variant.
  Addresses: REQ-P0-004. Rationale: MSL has no carry flag; 64-bit int is slow on Apple GPUs; carry-deferred limbs fit the int32 MAC datapath.
- DEC-METAL-005: Runtime MSL compilation + host-side closed-include concatenation.
  Addresses: REQ-P0-003, REQ-GOAL-005. Rationale: NVRTC-analog specialisation per run; langsec property enforced by our code.
- DEC-METAL-006: Independently rediscover and verify both device-compiler traps on Metal.
  Addresses: REQ-P0-007. Rationale: NVRTC lessons do not transfer; the unroll-disable pragma may be ignored by Clang, so effect must be measured.
- DEC-METAL-007: `storageModeShared` unified-memory buffers; keep the launch/collect API shape.
  Addresses: REQ-P0-003, REQ-P0-006. Rationale: no H2D/D2H copy on Apple Silicon; CLI launch loop unchanged.
- DEC-BACKEND-008: `DeviceInfo`/`device_description()` replaces `compute_capability()`.
  Addresses: REQ-P0-002. Rationale: compute capability is CUDA-only and leaks into CLI output.
- DEC-METAL-009: Chunk a launch into watchdog-sized dispatches, measured at run time.
  Addresses: REQ-P0-003, REQ-P0-006. Rationale: one launch was one command buffer, which macOS kills past a few seconds; `--launch-seconds` defaults to 4, so the limit was reachable in ordinary use. Only wall-clock can be bounded portably, so the rate is measured and the walk resumes through the point buffer.

#### Waves

##### Initiative Summary
- **Total items:** 7
- **Critical path:** 7 waves (W1-1 → W2-1 → W3-1 → W4-1 → W5-1 → W6-1 → W7-1)
- **Max width:** 1 — this initiative is deliberately sequential. The field → curve → search arithmetic stack must be validated bottom-up: a curve port cannot be trusted before its field arithmetic is differentially proven, and the search kernel cannot be trusted before the curve. Manufacturing parallelism across these layers would create false independence between things that must be verified in order. The genuine leaf-level parallelism (splitting field-mul vs field-add, or drafting docs skeleton early) is not worth the coordination cost and risks documenting unmeasured claims.
- **Gates:** 4 review (W1, W2, W5, W6), 1 approve (W7 — public claims + measured numbers)

##### Wave 1 (no dependencies)

**W1-1: Restructure `honion-gpu` for multi-backend + feature flags + CI (#issue)** — Weight: L, Gate: review
- Make `cudarc` optional under a `cuda` feature; add a `metal` feature with the `objc2-metal` (pending DEC-BACKEND-003) optional dep. Neither feature default.
- cfg-gate `nvrtc.rs`/`search.rs`/`tables.rs` (tables is backend-neutral and stays shared) so the crate compiles with exactly one backend.
- Introduce the backend-agnostic public facade: keep the `honion_gpu::Searcher`, `Hit`, `LaunchOutcome`, `SearchError`, `DeviceTables`, `auto_threads`, `candidates_per_batch`, `local_bytes_per_thread` names; add `DeviceInfo`/`device_description()` and remove `compute_capability()` from the public surface (or make it cuda-only).
- Update `honion-cli/src/search.rs` to print the generic device descriptor instead of `compute_capability`; select the backend feature in `honion-cli/Cargo.toml` via `[target.'cfg(target_os="macos")']` → metal and `cfg(target_os="linux")` → cuda.
- Add a GitHub Actions CI matrix: a Linux job building `--features cuda` + running non-GPU tests, a macOS job building `--features metal` + running non-GPU tests. Both green without a GPU.
- **Integration:** `honion-cli/Cargo.toml` (feature selection), `honion-cli/src/search.rs` (device-descriptor print), `honion-gpu/Cargo.toml` (optional deps + features), `honion-gpu/src/lib.rs` (cfg module gating + re-exports). New `.github/workflows/ci.yml`.

##### Wave 2
**Blocked by:** W1-1

**W2-1: Metal host driver `msl.rs` (runtime compile + launch substrate) + Metal test harness scaffold (#issue)** — Weight: L, Gate: review, Deps: W1-1
- `msl.rs`: NVRTC analog. `new_library_with_source` with `CompileOptions` (language version, fast-math off), `-D HALF=…` via prepended `#define`/macro injection, `new_function_with_name`, `new_compute_pipeline_state_with_function`.
- Port `known_headers()`/`expand_includes()` from `nvrtc.rs` to concatenate the closed MSL include set host-side (reject unknown includes before submission) — preserves langsec rule 4 explicitly. Add `msl::sources` embedding the `metal/*.metal` files via `include_str!`.
- Compile cache (in-memory + on-disk) keyed on source+params, mirroring `nvrtc::compile_cached`.
- `storageModeShared` buffer allocation helpers (start points, tables, hit buffer, hit counter, status) visible to CPU+GPU with no copy; command-buffer submit + completion sync for `launch_async`/`collect`.
- Metal `auto_threads`/`DeviceInfo` using `recommendedMaxWorkingSetSize`/`currentAllocatedSize` under unified memory.
- Shared Rust test `Harness` mirroring the CUDA `tests/` harness, with a clean skip when no Metal device is present.
- **Integration:** new `crates/honion-gpu/src/msl.rs` (re-exported from `lib.rs` under `cfg(feature="metal")`); new `crates/honion-gpu/metal/` dir; shared test-harness module under `tests/`.

##### Wave 3
**Blocked by:** W2-1

**W3-1: MSL field arithmetic `metal/fe25519.metal` (radix 2^25.5) + differential field tests (#issue)** — Weight: XL, Deps: W2-1
- Port `cuda/fe25519.cuh` (the radix-25.5 header, NOT `fe25519_u32.cuh`) to MSL: add/sub/mul/square/reduce/invert/pow22523/frombytes/tobytes on 10 carry-deferred limbs. No carry-flag intrinsics; bound limb growth and insert periodic widen/reduce.
- Port the field test kernels from `testkernels.cu` to `metal/testkernels.metal`.
- Add `tests/field_arithmetic_metal.rs` (cfg metal) reusing the `num-bigint` reference; demand exact agreement; skip cleanly with no device.
- Watch for the first compiler trap here (cold `fe_invert`/`fe_pow22523` inlining bloat) — apply `[[clang::noinline]]` and verify effect.
- **Integration:** `metal/fe25519.metal` + `metal/testkernels.metal` added to `msl::known_headers`/`sources`; new test file wired into the crate's test set.

##### Wave 4
**Blocked by:** W3-1

**W4-1: MSL Edwards-curve arithmetic `metal/ge25519.metal` + differential group tests (#issue)** — Weight: L, Deps: W3-1
- Port `cuda/ge25519.cuh`: the dual-addition-law trick (`y(P±Q)` from two multiplications, no curve constant, no projective coords) is the core win — preserve it exactly.
- Extend `metal/testkernels.metal` with group kernels; add `tests/group_arithmetic_metal.rs` differentially testing against `curve25519-dalek`.
- **Integration:** `metal/ge25519.metal` added to `msl` include set; test file wired in. Builds on the field layer proven in W3-1.

##### Wave 5
**Blocked by:** W4-1

**W5-1: MSL search kernel `metal/search.metal` + Metal `Searcher` end-to-end + set-equality tests (#issue)** — Weight: XL, Gate: review, Deps: W4-1
- Port `cuda/search.cu`: per-thread +8 scalar walk (clamping clears low 3 bits), pattern prefilter, Montgomery-batched inversion (513 candidates/inversion at default HALF), atomic hit reporting into a `storageModeShared` hit buffer via `atomic_fetch_add_explicit`. Map `thread_position_in_grid`, threadgroup size 256 (`maxTotalThreadsPerThreadgroup`), `device const*` tables, status flags.
- Wire the Metal `Searcher` (`new`/`set_start_points`/`launch_async`/`collect`) on top of `msl.rs`, keeping the DEC-METAL-007 API shape so `honion-cli`'s loop is unchanged.
- Rediscover the second compiler trap here (batch-loop unrolling blowup): apply `#pragma clang loop unroll(disable)`/`#pragma nounroll` and **verify** the effect via compile-time/AIR (Clang may ignore it — measure, do not assume).
- Add `tests/search_kernel_metal.rs`: exact set equality vs the host reference matcher + `walk_matches_scalar_arithmetic` over the full visited sequence.
- Prove `honion search --prefix …` writes a `honion verify`-passing key on an M4 Max.
- **Integration:** `metal/search.metal` added to `msl` sources; Metal `Searcher` becomes the `metal`-feature implementation behind the facade from W1-1; consumed unchanged by `honion-cli/src/search.rs`.

##### Wave 6
**Blocked by:** W5-1

**W6-1: Measured M4 Max benchmark (#issue)** — Weight: M, Gate: review, Deps: W5-1
- Metal analog of `examples/bench.rs` (compile/JIT timing + G/s + back-to-back launch decay), and a `bench/`-harness path that records M4 Max results as CSV following the existing discipline.
- Produce real, reproducible numbers. No number is asserted; every figure is measured and reproducible.
- **Integration:** new `examples/bench_metal.rs` (or cfg'd `bench.rs`); `bench/` harness updated to run the Metal path; results committed as data, not prose.

##### Wave 7
**Blocked by:** W6-1

**W7-1: Docs + README update (#issue)** — Weight: M, Gate: approve, Deps: W6-1
- README GPU requirements: replace "An NVIDIA GPU, compute capability 5.0 or later" hard requirement with per-backend requirements (NVIDIA CC 5.0+ *or* Apple Silicon + macOS).
- New `docs/08-metal-backend.md`: the MSL port, the two rediscovered compiler traps with their measured numbers, unified-memory simplification vs docs/06 "Overlapping the host", and the preserved trust boundary/closed-include property.
- Fold measured M4 Max numbers (from W6-1) into `docs/06-performance.md`/`docs/07-benchmarks.md`; note the estimator's default `--rate` is CUDA-measured and document the Metal-measured rate.
- **Integration:** `README.md`, new `docs/08-metal-backend.md`, edits to `docs/06`/`docs/07`; only measured numbers enter these files (Principle 4).

##### Critical Files
- `crates/honion-gpu/src/lib.rs` — the module gating + re-export surface that defines the backend seam.
- `crates/honion-gpu/src/nvrtc.rs` — the pattern (`known_headers`/`expand_includes`/`compile_cached`) the Metal `msl.rs` driver mirrors; the langsec closed-include reference.
- `crates/honion-gpu/src/search.rs` — the host driver shape (`Searcher`/`launch_async`/`collect`/`auto_threads`) the Metal backend must match behind the facade.
- `cuda/fe25519.cuh`, `cuda/ge25519.cuh`, `cuda/search.cu`, `cuda/testkernels.cu` — the ~1,300 lines of device arithmetic being ported to `metal/*.metal`.
- `crates/honion-cli/src/search.rs` — the sole GPU consumer; must stay backend-agnostic except the device-descriptor print.
- `crates/honion-gpu/tests/{field,group,search}_*.rs` — the differential-test bar the Metal suites must match exactly.

##### Decision Log

#### Native Metal GPU backend Worktree Strategy

Each wave was developed in an isolated worktree off `main`; because the initiative is sequential, waves ran one at a time and merged before the next began:
- **Wave 1:** `{project_root}/.worktrees/metal-w1-backend-seam` on branch `metal/w1-backend-seam`
- **Wave 2:** `{project_root}/.worktrees/metal-w2-msl-driver` on branch `metal/w2-msl-driver`
- **Wave 3:** `{project_root}/.worktrees/metal-w3-field` on branch `metal/w3-field`
- **Wave 4:** `{project_root}/.worktrees/metal-w4-curve` on branch `metal/w4-curve`
- **Wave 5:** `{project_root}/.worktrees/metal-w5-search` on branch `metal/w5-search`
- **Wave 6:** `{project_root}/.worktrees/metal-w6-bench` on branch `metal/w6-bench`
- **Wave 7:** `{project_root}/.worktrees/metal-w7-docs` on branch `metal/w7-docs`

Note: Waves 2–6 require a live Apple M4 Max to run their differential/GPU tests. Off Apple Silicon these waves can be implemented and built, but must have their tests run on an M4 Max before merge — verification is hardware-gated.

#### Native Metal GPU backend References

- Research notes (2026-08-25): binding choice, field representation, compiler traps, unified memory.
- `objc2-metal` — https://crates.io/crates/objc2-metal ; metal-rs deprecation — https://github.com/gfx-rs/metal-rs/issues/339
- `new_library_with_source` — https://developer.apple.com/documentation/metal/mtldevice/1433431-newlibrarywithsource
- Apple int32/64 GPU throughput — https://github.com/philipturner/metal-benchmarks ; radix 2^25.5 — Sandy2x / Cambridge Curve25519 tutorial.
- Local: `docs/02-gpu-architecture.md` (dual addition law), `docs/03-langsec-design.md` (rule 4), `docs/05-security-model.md` (trust boundary), `docs/06-performance.md` ("Two compiler lessons", "Overlapping the host"), `docs/07-benchmarks.md` (measurement discipline).

---

## Completed Initiatives

| Initiative | Period | Phases | Key Decisions | Archived |
|-----------|--------|--------|---------------|----------|
| Native Metal GPU backend for Apple Silicon | 2026-08-25 → 2026-08-26 | W1 backend seam · W2 msl.rs driver · W3 field · W4 curve · W5 search kernel + Searcher · W6 measured M4 Max bench · W7 docs | DEC-BACKEND-001/002/003/008 (feature-gated seam, objc2-metal), DEC-METAL-004 (radix-25.5), DEC-METAL-005 (runtime MSL + closed includes), DEC-METAL-006 (both compiler traps rediscovered + verified), DEC-METAL-007 (unified-memory synchronous Searcher) | 2026-08-26 |

---

## Parked Issues

Issues not belonging to any active initiative. Tracked for future consideration.

| Issue | Description | Reason Parked |
|-------|-------------|---------------|
<!-- Empty at project start -->
