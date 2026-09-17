//! Driving the search kernel.
//!
//! # What crosses the boundary
//!
//! Into the device: compressed public keys (32 bytes each), and the integer
//! tables built by [`crate::tables`]. Out of the device: `(thread, iteration,
//! pattern)` triples and a status word.
//!
//! No secret goes in and no key comes out. The caller holds the secret scalars
//! that generated the starting points, and reconstructs a hit's secret as
//! `a0[thread] + 8 * iteration` from its own memory. This is why the kernel can
//! be treated as an untrusted accelerator: the worst a broken one can do is
//! waste time or report candidates that fail verification.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use crate::nvrtc;
use crate::tables::DeviceTables;

/// Positive offsets in the kernel's precomputed table.
///
/// Each offset yields two candidates — `base + off` and `base - off` — via the
/// dual addition law, so a batch covers `2 * HALF + 1` candidates: the base
/// point and `HALF` pairs either side of it.
///
/// Compiled into the kernel, so changing it changes the generated code rather
/// than a runtime parameter. Larger values amortise the batch's single modular
/// inversion further, but need proportionally more per-thread local memory and
/// more shared memory for the table, both of which cost occupancy.
pub const DEFAULT_HALF: u32 = 512;

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

/// Minimum compute capability required for the warp-cooperative kernel path.
/// CC 7.0 (Volta) is the floor for `__shfl_sync` and IADD3.
pub const MIN_CC_COOPERATIVE: (u32, u32) = (7, 0);

/// Candidates a thread examines per batch, for a given `half`.
#[must_use]
pub const fn candidates_per_batch(half: u32) -> u32 {
    2 * half + 1
}

/// Threads per block. 256 matches the kernel's `__launch_bounds__`.
pub const BLOCK_SIZE: u32 = 256;

/// Threads per block in the small-blocks variant.
const SMALL_BLOCK_SIZE: u32 = 128;

/// Width of a cooperative group: 8 warp lanes, one per field-element limb.
const COOP_WIDTH: u32 = 8;

/// Bytes of device-local memory each walk needs, for a given `half`.
///
/// One field element per candidate — the numerator — for `2 * half`
/// candidates. In the scalar kernel each thread holds 8 limbs; in the
/// cooperative kernel the 8 limbs are spread across 8 threads, but the
/// total per walk is the same.
#[must_use]
pub const fn local_bytes_per_walk(half: u32) -> u64 {
    (2 * half as u64) * (FE_LIMBS as u64) * 4
}

/// Bytes of device-local memory each CUDA thread needs.
#[must_use]
pub const fn local_bytes_per_thread(half: u32) -> u64 {
    local_bytes_per_walk(half)
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
    let ctx = CudaContext::new(0).map_err(|e| SearchError::Driver(format!("{e:?}")))?;
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

/// A candidate the device believes matches.
///
/// "Believes" is exact: this is a claim to be checked, not a result. The
/// `pattern_id` in particular is advisory — verification re-derives which
/// patterns actually match rather than trusting it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct Hit {
    /// Index of the thread that found it; selects the starting scalar.
    pub thread_id: u32,
    /// Signed step count from that thread's starting scalar: the `m` in
    /// `a = a0 + 8m`.
    ///
    /// Signed because the search covers a symmetric range either side of each
    /// starting point — the dual addition law produces `base + off` and
    /// `base - off` together — so a match may lie below where the thread
    /// started.
    pub offset: i32,
    /// Which pattern the device matched. Advisory.
    pub pattern_id: u32,
    /// Padding, so the layout matches the device struct exactly.
    pub reserved: u32,
}

// Status bits the kernel raises. Kept in sync with `cuda/search.cu` by
// `status_bits_match_the_kernel` in the integration tests.
const STATUS_BAD_START_POINT: u32 = 1;
const STATUS_HIT_OVERFLOW: u32 = 2;
const STATUS_SINGULAR: u32 = 4;

/// What one launch produced.
#[derive(Clone, Debug)]
pub struct LaunchOutcome {
    /// Candidates to verify.
    pub hits: Vec<Hit>,
    /// Candidates the device found, which may exceed `hits.len()` if the
    /// buffer overflowed.
    pub total_found: u32,
    /// Public keys examined.
    pub examined: u64,
}

/// A configured search: compiled kernel, uploaded tables, allocated buffers.
pub struct Searcher {
    ctx: Arc<CudaContext>,
    func: CudaFunction,
    half: u32,
    num_threads: u32,
    cooperative: bool,
    block_size: u32,

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
    /// Equivalent to [`Self::with_options`] with default settings.
    pub fn new(
        tables: &DeviceTables,
        num_threads: u32,
        half: u32,
        max_hits: u32,
    ) -> Result<Self, SearchError> {
        Self::with_options(tables, num_threads, half, max_hits, &SearchOptions::default())
    }

    /// Compile the kernel for the present device with tuning options.
    ///
    /// `num_threads` is how many independent walks run concurrently; it should
    /// be a large multiple of [`BLOCK_SIZE`]. `max_hits` bounds the per-launch
    /// hit buffer.
    ///
    /// # Errors
    ///
    /// If no CUDA device is present, the kernel fails to compile, or a device
    /// allocation fails.
    pub fn with_options(
        tables: &DeviceTables,
        num_threads: u32,
        half: u32,
        max_hits: u32,
        opts: &SearchOptions,
    ) -> Result<Self, SearchError> {
        if half == 0 {
            return Err(SearchError::BadParameter(
                "half must be positive; it is the number of precomputed offsets".into(),
            ));
        }
        let ctx = CudaContext::new(0).map_err(|e| SearchError::Driver(format!("{e:?}")))?;
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
        let cc = (major.max(0) as u32, minor.max(0) as u32);

        if let Some(min) = opts.min_cc {
            if cc.0 < min.0 || (cc.0 == min.0 && cc.1 < min.1) {
                return Err(SearchError::BadParameter(format!(
                    "device compute capability {}.{} is below the required minimum {}.{}",
                    cc.0, cc.1, min.0, min.1,
                )));
            }
        }

        let cooperative = opts.cooperative
            && (cc.0 > MIN_CC_COOPERATIVE.0
                || (cc.0 == MIN_CC_COOPERATIVE.0 && cc.1 >= MIN_CC_COOPERATIVE.1));

        let mut defines = vec![
            ("HALF", half.to_string()),
            ("FE_RADIX32", FE_RADIX32.to_owned()),
        ];
        if opts.predicate_carry {
            defines.push(("FE_PREDICATE_CARRY", "1".to_owned()));
        }
        if opts.small_blocks {
            defines.push(("SMALL_BLOCKS", "1".to_owned()));
        }
        if opts.store_dens {
            defines.push(("STORE_DENS", "1".to_owned()));
        }

        let source = if cooperative {
            nvrtc::sources::SEARCH_COOP
        } else {
            nvrtc::sources::SEARCH
        };

        let mut extra_opts = Vec::new();
        if let Some(cap) = opts.max_registers {
            extra_opts.push(format!("--maxrregcount={cap}"));
        }

        let ptx = if extra_opts.is_empty() {
            nvrtc::compile_cached(source, cc, &defines)?
        } else {
            nvrtc::compile_with_options(source, cc, &defines, &extra_opts)?
        };
        let module = ctx
            .load_module(ptx.into())
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
        let search_name = if cooperative { "honion_search_coop" } else { "honion_search" };
        let func = module
            .load_function(search_name)
            .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
        eprintln!(
            "[kernel] {} regs={} local={}B",
            search_name,
            func.num_regs().unwrap_or(-1),
            func.local_size_bytes().unwrap_or(-1),
        );
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

        let block_size = if opts.small_blocks && !cooperative {
            SMALL_BLOCK_SIZE
        } else {
            BLOCK_SIZE
        };

        Ok(Self {
            ctx,
            func,
            half,
            num_threads,
            cooperative,
            block_size,
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

    /// Whether the warp-cooperative kernel is active.
    #[must_use]
    pub const fn cooperative(&self) -> bool {
        self.cooperative
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
    /// # Errors
    ///
    /// If the device cannot be queried.
    pub fn compute_capability(&self) -> Result<(i32, i32), SearchError> {
        self.ctx
            .compute_capability()
            .map_err(|e| SearchError::Driver(format!("{e:?}")))
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

        let num_walks = self.num_threads;
        let cuda_threads = if self.cooperative {
            num_walks * COOP_WIDTH
        } else {
            num_walks
        };
        let bs = self.block_size;
        let cfg = LaunchConfig {
            grid_dim: (cuda_threads.div_ceil(bs), 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_groups = self.num_groups;
        let max_hits = self.max_hits;
        let mut builder = stream.launch_builder(&self.func);
        builder
            .arg(&self.d_points)
            .arg(&num_walks)
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
        // Safety: the argument list matches the search kernel's signature
        // exactly, in order and in type. Every buffer was allocated at the
        // size the kernel indexes. The scalar kernel reads `num_walks` as
        // `num_threads`; the cooperative kernel as `num_walks` — same
        // parameter position and meaning.
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

/// Why a search could not be set up or run.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SearchError {
    /// The CUDA driver reported a failure.
    #[error("CUDA driver error: {0}")]
    Driver(String),
    /// Device code failed to compile.
    #[error(transparent)]
    Nvrtc(#[from] nvrtc::NvrtcError),
    /// The kernel raised a status flag.
    #[error("device reported a fault: {0}")]
    Device(String),
    /// A construction parameter was out of range.
    #[error("{0}")]
    BadParameter(String),
    /// The wrong number of starting points was supplied.
    #[error("expected {expected} starting points, got {found}")]
    WrongPointCount {
        /// Points the searcher was configured for.
        expected: usize,
        /// Points supplied.
        found: usize,
    },
}

/// Tuning options for kernel compilation experiments.
///
/// All fields default to `None` / `false`, which reproduces the baseline
/// kernel. Each option is designed to be toggled independently so that its
/// effect can be measured in isolation.
#[derive(Clone, Debug, Default)]
pub struct SearchOptions {
    /// Enforce a minimum compute capability, rejecting older devices.
    pub min_cc: Option<(u32, u32)>,

    /// Cap registers per thread via NVRTC's `--maxrregcount`. Forces the
    /// compiler to spill to local memory when it would otherwise exceed this
    /// count. Useful for testing whether the occupancy gain from more
    /// blocks/SM outweighs the spill cost.
    ///
    /// The baseline kernel uses ~128 registers (2 blocks/SM on Blackwell).
    /// 85 would allow 3 blocks/SM; 64 would allow 4.
    pub max_registers: Option<u32>,

    /// Compile with the predicate-carry field multiply variant. Uses
    /// predicate registers and `setp` instead of hardware carry chains,
    /// trading more instructions for more scheduling freedom.
    pub predicate_carry: bool,

    /// Use the warp-cooperative kernel (8 threads per walk). Requires
    /// CC >= 7.0; silently falls back to the scalar kernel on older GPUs.
    pub cooperative: bool,

    /// Cache the dual-pair denominators in local memory instead of
    /// recomputing them in the backward pass. Saves 1024 fe_mul per batch
    /// but doubles local-memory traffic (~32 KB extra per thread).
    pub store_dens: bool,

    /// Use 128-thread blocks instead of 256, with `__launch_bounds__(128, 5)`
    /// to target 5 blocks/SM = 20 warps. Trades shared memory for L1 cache
    /// (the offset table is read from global/L2 instead of shared memory) so
    /// more blocks fit.
    pub small_blocks: bool,
}

/// Device information for diagnostics and experiment selection.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    /// Major and minor compute capability, e.g. `(12, 0)`.
    pub compute_capability: (u32, u32),
    /// Number of streaming multiprocessors.
    pub sm_count: u32,
    /// Total device memory in bytes.
    pub total_memory_bytes: u64,
    /// Free device memory in bytes at the time of the query.
    pub free_memory_bytes: u64,
}

/// Query the present device without compiling or allocating anything.
///
/// # Errors
///
/// If no CUDA device is present.
pub fn device_info() -> Result<DeviceInfo, SearchError> {
    let ctx = CudaContext::new(0).map_err(|e| SearchError::Driver(format!("{e:?}")))?;
    let (major, minor) = ctx
        .compute_capability()
        .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
    let (free, total) = ctx
        .mem_get_info()
        .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
    let sm_count = ctx
        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(|e| SearchError::Driver(format!("{e:?}")))?;
    Ok(DeviceInfo {
        compute_capability: (major.max(0) as u32, minor.max(0) as u32),
        sm_count: sm_count.max(0) as u32,
        total_memory_bytes: total as u64,
        free_memory_bytes: free as u64,
    })
}
