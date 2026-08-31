//! The CUDA backend: NVRTC-compiled kernel driven through `cudarc`.
//!
//! This is the reference backend — the one the differential test suites gate
//! and the benchmark numbers in `docs/` describe. Backend-neutral types (what
//! a [`Hit`] is, what a launch produces) live in the parent module; this file
//! holds everything that actually needs a CUDA driver under it.

// @decision DEC-BACKEND-001 (see search/mod.rs)
// @title CUDA driver code moves verbatim behind the backend seam
// @status accepted
// @rationale This file is `search.rs` as it stood before the Metal initiative,
//   with three deliberate changes and no others: shared types (`Hit`,
//   `LaunchOutcome`, `SearchError`, `DEFAULT_HALF`, `candidates_per_batch`)
//   are imported from the parent instead of defined here; `device_info()` is
//   added so the CLI can print a backend-neutral descriptor (DEC-BACKEND-008);
//   and `FE_LIMBS`/`BLOCK_SIZE`/`local_bytes_per_thread` stay here because
//   they describe *this* kernel's representation, which the Metal backend is
//   free to choose differently (DEC-METAL-004). Keeping the move verbatim
//   makes the seam auditable: `git log --follow` shows a rename, not a rewrite.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use super::{DeviceInfo, Hit, LaunchOutcome, SearchError, candidates_per_batch};
use crate::nvrtc;
use crate::tables::DeviceTables;

/// Limbs per field element in the device's field representation.
///
/// Eight 32-bit limbs. The host only needs this to size the offset-table
/// buffer; it never interprets a limb.
pub const FE_LIMBS: usize = 8;

/// Compile the kernel with the 8x32-limb field arithmetic.
///
/// Gated by `crates/honion-gpu/tests/field_arithmetic.rs`, which runs the same
/// differential suite against both implementations.
const FE_RADIX32: &str = "1";

/// Threads per block. 256 matches the kernel's `__launch_bounds__`.
pub const BLOCK_SIZE: u32 = 256;

/// Whether the CUDA driver library is present and loadable.
///
/// `cudarc`'s `dynamic-loading` feature `dlopen`s libcuda on first use and
/// **panics** when it is absent — there is no `Result` path for "no NVIDIA
/// driver installed", because the load happens inside a `OnceLock` initialiser
/// (`cudarc::panic_no_lib_found`). So `CudaContext::new` cannot report that
/// case as an error: it aborts the process instead. Every entry point that is
/// about to touch the driver has to ask this first.
///
/// Note what this does *not* mean: a true answer says the driver library
/// loaded, not that a usable GPU is attached. Context creation still has to be
/// tried, and it can still fail normally afterwards.
#[must_use]
pub fn driver_present() -> bool {
    // Safety: `is_culib_present` only attempts `dlopen` on a fixed list of
    // library names and reports whether one resolved. It dereferences no
    // pointer and calls no driver entry point.
    unsafe { cudarc::driver::sys::is_culib_present() }
}

/// Create a context on device 0, reporting a missing driver as an error.
///
/// The pre-check in [`driver_present`] is what keeps this a `Result` rather
/// than a panic on a machine with no NVIDIA driver.
fn context() -> Result<Arc<CudaContext>, SearchError> {
    if !driver_present() {
        return Err(SearchError::Driver(
            "no CUDA driver library could be loaded; is the NVIDIA driver installed?".into(),
        ));
    }
    CudaContext::new(0).map_err(|e| SearchError::Driver(format!("{e:?}")))
}

/// Bytes of device-local memory each thread needs, for a given `half`.
///
/// One field element per candidate — the numerator — for `2 * half`
/// candidates. Denominators are recomputed in the backward pass rather than
/// stored, which halved this and the memory traffic with it.
#[must_use]
pub const fn local_bytes_per_thread(half: u32) -> u64 {
    (2 * half as u64) * (FE_LIMBS as u64) * 4
}

/// Choose a thread count that fits comfortably in free device memory.
///
/// More concurrent walks raise *kernel* throughput until the card runs out of
/// room for their local memory — but the host must draw a fresh secret scalar
/// and derive a public key for every thread each launch, and that cost grows
/// linearly with the count. Past a few hundred thousand threads the kernel
/// gains a fraction of a percent while the host work grows by hundreds of
/// milliseconds, so end-to-end throughput falls even though the benchmark
/// number rises. The cap below is set from end-to-end measurement, not from
/// how much memory happens to be free.
///
/// # Errors
///
/// If the device cannot be queried.
pub fn auto_threads(half: u32) -> Result<u32, SearchError> {
    let ctx = context()?;
    let (free, _total) = ctx
        .mem_get_info()
        .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
    let budget = free as u64 / 4;
    let per_thread = local_bytes_per_thread(half).max(1);
    let fit = budget / per_thread;
    // 64 blocks keeps a small card busy. The upper clamp is where measured
    // throughput stops improving; beyond it the extra walks add host work
    // without adding kernel throughput.
    let blocks = (fit / u64::from(BLOCK_SIZE)).clamp(64, 2048);
    Ok((blocks as u32) * BLOCK_SIZE)
}

// Status bits the kernel raises. Kept in sync with `cuda/search.cu` by
// `status_bits_match_the_kernel` in the integration tests.
const STATUS_BAD_START_POINT: u32 = 1;
const STATUS_HIT_OVERFLOW: u32 = 2;
const STATUS_SINGULAR: u32 = 4;

/// A configured search: compiled kernel, uploaded tables, allocated buffers.
pub struct Searcher {
    ctx: Arc<CudaContext>,
    func: CudaFunction,
    half: u32,
    num_threads: u32,

    // Precomputed offsets, built once on the device at construction.
    d_off_table: CudaSlice<u32>,
    d_giant: CudaSlice<u32>,

    // Pattern tables, uploaded once.
    d_group_mask: CudaSlice<u64>,
    d_group_off: CudaSlice<u32>,
    d_target: CudaSlice<u64>,
    d_target_pat: CudaSlice<u32>,
    d_res_off: CudaSlice<u32>,
    d_res: CudaSlice<u64>,
    num_groups: u32,

    // Reused across launches.
    d_points: CudaSlice<u8>,
    d_hits: CudaSlice<u32>,
    d_hit_count: CudaSlice<u32>,
    d_status: CudaSlice<u32>,
    max_hits: u32,
}

impl Searcher {
    /// Compile the kernel for the present device and upload `tables`.
    ///
    /// `num_threads` is how many independent walks run concurrently; it should
    /// be a large multiple of [`BLOCK_SIZE`]. `max_hits` bounds the per-launch
    /// hit buffer.
    ///
    /// # Errors
    ///
    /// If no CUDA device is present, the kernel fails to compile, or a device
    /// allocation fails.
    pub fn new(
        tables: &DeviceTables,
        num_threads: u32,
        half: u32,
        max_hits: u32,
    ) -> Result<Self, SearchError> {
        if half == 0 {
            return Err(SearchError::BadParameter(
                "half must be positive; it is the number of precomputed offsets".into(),
            ));
        }
        let ctx = context()?;
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;

        // Compile with the batch size baked in, so the kernel's local-memory
        // arrays and unrolling are fixed at compile time rather than being
        // dynamic. This is the payoff for compiling at run time.
        let ptx = nvrtc::compile_cached(
            nvrtc::sources::SEARCH,
            (major.max(0) as u32, minor.max(0) as u32),
            &[("HALF", half.to_string()), ("FE_RADIX32", FE_RADIX32.to_owned())],
        )?;
        let module = ctx
            .load_module(ptx.into())
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
        let func = module
            .load_function("honion_search")
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
        let build = module
            .load_function("honion_build_offsets")
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;

        let stream = ctx.default_stream();
        let up_u64 = |v: &[u64]| -> Result<CudaSlice<u64>, SearchError> {
            let v = if v.is_empty() { vec![0u64] } else { v.to_vec() };
            stream
                .clone_htod(&v)
                .map_err(|e| SearchError::Driver(format!("{e:?}")))
        };
        let up_u32 = |v: &[u32]| -> Result<CudaSlice<u32>, SearchError> {
            let v = if v.is_empty() { vec![0u32] } else { v.to_vec() };
            stream
                .clone_htod(&v)
                .map_err(|e| SearchError::Driver(format!("{e:?}")))
        };

        let d_group_mask = up_u64(&tables.group_mask)?;
        let d_group_off = up_u32(&tables.group_off)?;
        let d_target = up_u64(&tables.target)?;
        let d_target_pat = up_u32(&tables.target_pat)?;
        let d_res_off = up_u32(&tables.res_off)?;
        let d_res = up_u64(&tables.res)?;

        let alloc_u32 = |n: usize| -> Result<CudaSlice<u32>, SearchError> {
            stream
                .alloc_zeros(n)
                .map_err(|e| SearchError::Driver(format!("{e:?}")))
        };
        // Offset table: HALF entries of (x, y, x*y), FE_LIMBS words each.
        let mut d_off_table: CudaSlice<u32> = stream
            .alloc_zeros(half as usize * 3 * FE_LIMBS)
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
        let mut d_giant: CudaSlice<u32> = stream
            .alloc_zeros(3 * FE_LIMBS)
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;

        let d_points = stream
            .alloc_zeros(num_threads as usize * 32)
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
        // Four u32 per hit, matching the device struct.
        let d_hits = alloc_u32(max_hits as usize * 4)?;
        let d_hit_count = alloc_u32(1)?;
        let mut d_status = alloc_u32(1)?;

        // Build the offset table once. A single thread does HALF+1 modular
        // inversions, which takes a few milliseconds against a search that runs
        // for seconds at minimum.
        {
            let mut b = stream.launch_builder(&build);
            b.arg(&mut d_off_table).arg(&mut d_giant).arg(&mut d_status);
            // Safety: the argument list matches `honion_build_offsets`, and both
            // buffers were allocated at the sizes it writes.
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
            stream
                .synchronize()
                .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
            let st = stream
                .clone_dtoh(&d_status)
                .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
            if st.first().copied().unwrap_or(0) != 0 {
                return Err(SearchError::Device(
                    "building the offset table hit a singular point; \
                     this indicates a field-arithmetic fault".into(),
                ));
            }
        }

        Ok(Self {
            ctx,
            func,
            half,
            num_threads,
            d_off_table,
            d_giant,
            d_group_mask,
            d_group_off,
            d_target,
            d_target_pat,
            d_res_off,
            d_res,
            num_groups: tables.num_groups(),
            d_points,
            d_hits,
            d_hit_count,
            d_status,
            max_hits,
        })
    }

    /// Number of concurrent walks.
    #[must_use]
    pub const fn num_threads(&self) -> u32 {
        self.num_threads
    }

    /// Positive offsets in the precomputed table.
    #[must_use]
    pub const fn half(&self) -> u32 {
        self.half
    }

    /// Candidates examined per batch, per thread.
    #[must_use]
    pub const fn candidates_per_batch(&self) -> u32 {
        candidates_per_batch(self.half)
    }

    /// The device's compute capability.
    ///
    /// CUDA-only by design: the backend-neutral way to describe the device is
    /// [`Self::device_info`]. This survives for the tests and tools that
    /// genuinely mean CUDA.
    ///
    /// # Errors
    ///
    /// If the device cannot be queried.
    pub fn compute_capability(&self) -> Result<(i32, i32), SearchError> {
        self.ctx
            .compute_capability()
            .map_err(|e| SearchError::Driver(format!("{e:?}")))
    }

    /// Describe the device this search runs on.
    ///
    /// # Errors
    ///
    /// If the device cannot be queried.
    pub fn device_info(&self) -> Result<DeviceInfo, SearchError> {
        let name = self
            .ctx
            .name()
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
        let (major, minor) = self.compute_capability()?;
        Ok(DeviceInfo {
            backend: "CUDA",
            name,
            detail: format!("compute capability {major}.{minor}"),
        })
    }

    /// Install the starting points for the next launches.
    ///
    /// `points` must contain exactly [`Self::num_threads`] compressed public
    /// keys. These are public data; the corresponding secrets stay with the
    /// caller.
    ///
    /// # Errors
    ///
    /// [`SearchError::WrongPointCount`] if the slice is the wrong length.
    pub fn set_start_points(&mut self, points: &[[u8; 32]]) -> Result<(), SearchError> {
        if points.len() != self.num_threads as usize {
            return Err(SearchError::WrongPointCount {
                expected: self.num_threads as usize,
                found: points.len(),
            });
        }
        let flat: Vec<u8> = points.iter().flatten().copied().collect();
        let stream = self.ctx.default_stream();
        stream
            .memcpy_htod(&flat, &mut self.d_points)
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
        Ok(())
    }

    /// Examine at least `candidates` keys on every thread.
    ///
    /// Work is done in batches of [`Self::candidates_per_batch`], so the request
    /// is rounded up to a whole number of batches; the returned
    /// [`LaunchOutcome::examined`] reports what was actually covered.
    ///
    /// A thread starting from scalar `a0` covers a contiguous run of scalars
    /// `a0 + 8m` centred on its start: `m` ranges over
    /// `-half ..= batches * (2*half+1) - half - 1`. The ranges of successive
    /// batches tile exactly, so no key is examined twice and none is skipped.
    ///
    /// # Errors
    ///
    /// [`SearchError::Device`] if the kernel raised a status flag, which always
    /// indicates a bug or a hardware fault rather than an ordinary condition.
    pub fn launch(&mut self, candidates: u32) -> Result<LaunchOutcome, SearchError> {
        let batches = self.launch_async(candidates)?;
        self.collect(batches)
    }

    /// Enqueue a search without waiting for it.
    ///
    /// Returns the number of batches queued, which [`Self::collect`] needs to
    /// report how much was examined. Between the two calls the CPU is free —
    /// which matters because preparing the next launch's starting points costs
    /// time proportional to the thread count, and on a large GPU that is
    /// hundreds of milliseconds that would otherwise sit in front of every
    /// launch doing nothing.
    ///
    /// The caller must not touch the starting points until [`Self::collect`]
    /// returns; the kernel is reading them.
    ///
    /// # Errors
    ///
    /// As [`Self::launch`].
    pub fn launch_async(&mut self, candidates: u32) -> Result<u32, SearchError> {
        let per_batch = self.candidates_per_batch();
        let num_batches = candidates.div_ceil(per_batch).max(1);
        let stream = self.ctx.default_stream();
        stream
            .memset_zeros(&mut self.d_hit_count)
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
        stream
            .memset_zeros(&mut self.d_status)
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;

        let cfg = LaunchConfig {
            grid_dim: (self.num_threads.div_ceil(BLOCK_SIZE), 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_threads = self.num_threads;
        let num_groups = self.num_groups;
        let max_hits = self.max_hits;
        let mut builder = stream.launch_builder(&self.func);
        builder
            .arg(&self.d_points)
            .arg(&num_threads)
            .arg(&num_batches)
            .arg(&self.d_off_table)
            .arg(&self.d_giant)
            .arg(&num_groups)
            .arg(&self.d_group_mask)
            .arg(&self.d_group_off)
            .arg(&self.d_target)
            .arg(&self.d_target_pat)
            .arg(&self.d_res_off)
            .arg(&self.d_res)
            .arg(&mut self.d_hits)
            .arg(&mut self.d_hit_count)
            .arg(&max_hits)
            .arg(&mut self.d_status);
        // Safety: the argument list matches `honion_search`'s signature in
        // `cuda/search.cu` exactly, in order and in type. Every buffer was
        // allocated above at the size the kernel indexes, and the kernel bounds
        // its thread index against `num_threads`.
        unsafe { builder.launch(cfg) }.map_err(|e| SearchError::Driver(format!("{e:?}")))?;
        Ok(num_batches)
    }

    /// Wait for the enqueued search and read back what it found.
    ///
    /// # Errors
    ///
    /// As [`Self::launch`].
    pub fn collect(&mut self, num_batches: u32) -> Result<LaunchOutcome, SearchError> {
        let per_batch = self.candidates_per_batch();
        let stream = self.ctx.default_stream();
        stream
            .synchronize()
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;

        let status = stream
            .clone_dtoh(&self.d_status)
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
        let status = status.first().copied().unwrap_or(0);
        if status & STATUS_BAD_START_POINT != 0 {
            return Err(SearchError::Device(
                "a starting point failed to decompress; it was not a valid public key".into(),
            ));
        }
        if status & STATUS_SINGULAR != 0 {
            return Err(SearchError::Device(
                "a denominator vanished during the walk, which means a base point \
                 coincided with one of its own offsets; the arithmetic should make \
                 that impossible, so suspect a field-arithmetic bug or a hardware fault"
                    .into(),
            ));
        }

        let count = stream
            .clone_dtoh(&self.d_hit_count)
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
        let total_found = count.first().copied().unwrap_or(0);
        let kept = total_found.min(self.max_hits);

        let mut hits = Vec::new();
        if kept > 0 {
            let raw = stream
                .clone_dtoh(&self.d_hits)
                .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
            for chunk in raw.chunks_exact(4).take(kept as usize) {
                hits.push(Hit {
                    thread_id: chunk[0],
                    // The device writes a signed offset into this slot.
                    offset: chunk[1] as i32,
                    pattern_id: chunk[2],
                    reserved: chunk[3],
                });
            }
        }
        if status & STATUS_HIT_OVERFLOW != 0 {
            // Not fatal: the run found more than the buffer holds. The caller
            // keeps what fit and is told the rest were dropped, rather than
            // silently losing them.
            return Ok(LaunchOutcome {
                hits,
                total_found,
                examined: u64::from(self.num_threads)
                    * u64::from(num_batches)
                    * u64::from(per_batch),
            });
        }

        Ok(LaunchOutcome {
            hits,
            total_found,
            examined: u64::from(self.num_threads)
                * u64::from(num_batches)
                * u64::from(per_batch),
        })
    }
}
