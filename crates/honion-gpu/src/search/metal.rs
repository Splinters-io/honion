//! The Apple Metal backend: the real search driver.
//!
//! The Metal counterpart of [`super::cuda`]. It compiles `metal/search.metal`
//! at run time with `HALF` baked in, builds the offset table on the device once,
//! and then dispatches the search kernel per launch. Buffers use unified
//! (`storageModeShared`) memory, so there is no host/device copy to schedule
//! (DEC-METAL-007): the CPU writes start points straight into the buffer the GPU
//! reads, and reads hits straight out of the buffer the GPU wrote.
//!
//! The trust boundary is identical to the CUDA backend: only public points and
//! host-built integer tables cross to the device, and the device returns
//! `(thread, offset)` claims the host re-verifies. No secret is uploaded.

// @decision DEC-METAL-007
// @title The Metal Searcher keeps the CUDA API shape over a synchronous dispatch
// @status accepted
// @rationale `honion-cli` drives `new` → `set_start_points` → `launch_async` →
//   `collect`, and the backend facade preserves those names, so the Metal Searcher
//   implements them unchanged. On unified memory there is no H2D/D2H transfer to
//   overlap, so `launch_async` runs the dispatch and stashes the outcome and
//   `collect` returns it — the host/GPU overlap the CUDA path needs for PCIe
//   transfers buys nothing here. The offset table is read from device memory
//   rather than threadgroup memory (see metal/search.metal); at HALF=512 the
//   table exceeds an Apple GPU's threadgroup allocation, and staging is a
//   bandwidth optimisation deferred as tuning work, not a correctness
//   requirement.

// @decision DEC-METAL-009
// @title A launch is split into watchdog-sized dispatches, measured at run time
// @status accepted
// @rationale macOS kills a command buffer that occupies the GPU too long —
//   `kIOGPUCommandBufferCallbackErrorHang`, a few seconds, and sooner when
//   something else is contending for the device. The kernel previously ran a
//   whole launch in one command buffer, so the limit was reachable by ordinary
//   use: `honion-cli` sizes launches to `--launch-seconds`, which *defaults to
//   4*, and CI hit exactly this on a shared runner GPU. Sizing a dispatch in
//   batches cannot fix it, because how long a batch takes is a property of the
//   device, not of the work; the only quantity that has to stay bounded is
//   wall-clock. So the searcher measures the rate as it goes and chunks the
//   launch to `DISPATCH_TARGET_SECS` per command buffer, starting from a small
//   probe on the very first dispatch, when nothing has been measured yet.
//   Resuming is exact rather than approximate: the kernel writes back the point
//   its thread reached, and `batch_base` keeps reported offsets absolute, so a
//   launch split twenty ways returns what the single-dispatch launch returned.
//   This is Metal-only on purpose. The CUDA path has no equivalent watchdog on
//   the compute-only Linux devices it targets, and DEC-BACKEND-001 keeps that
//   file a verbatim move; adding a mechanism it does not need would cost the
//   auditability that decision buys.

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

use super::{DeviceInfo, Hit, LaunchOutcome, SearchError, candidates_per_batch};
use crate::msl::{MslKernel, MslLibrary, SharedBuffer};
use crate::tables::DeviceTables;

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// Limbs per field element in the Metal field representation (radix 2^25.5).
pub const FE_LIMBS: usize = 10;

/// Words per offset-table entry: (x, y, x*y), each `FE_LIMBS` limbs.
const OFF_STRIDE: usize = 3 * FE_LIMBS;

/// Threads per threadgroup. Matches the kernel's
/// `[[max_total_threads_per_threadgroup(256)]]`.
const THREADGROUP: usize = 256;

/// Wall-clock a single command buffer is aimed at (DEC-METAL-009).
///
/// The watchdog's real limit is a few seconds and is not published, so this is
/// set far enough below it to absorb the things that make a dispatch overrun
/// its estimate — another process taking the GPU, thermal throttling, a display
/// refresh — while still being long enough that per-dispatch overhead stays
/// negligible against the work itself.
const DISPATCH_TARGET_SECS: f64 = 0.25;

/// Batches in the first dispatch of the process, before any measurement.
///
/// Deliberately tiny: on an unknown and possibly slow device this is the one
/// dispatch whose duration cannot be predicted, so it is sized to be quick
/// everywhere rather than efficient anywhere. Every later dispatch is sized
/// from measurement.
const PROBE_BATCHES: u32 = 1;

// Status bits the kernel raises, kept in sync with metal/search.metal.
const STATUS_BAD_START_POINT: u32 = 1;
// STATUS_HIT_OVERFLOW (bit 2) is set by the kernel when more hits are found than
// the buffer holds; the host detects that case directly from `total_found >
// max_hits` in `read_outcome`, so the flag itself is not inspected here.
const STATUS_SINGULAR: u32 = 4;

/// Bytes of thread-private memory each thread's numerator array needs.
///
/// One field element per candidate (only the numerator; denominators are
/// recomputed), for `2 * half` candidates, ten 32-bit limbs each.
#[must_use]
pub const fn local_bytes_per_thread(half: u32) -> u64 {
    (2 * half as u64) * (FE_LIMBS as u64) * 4
}

/// Choose a thread count for the device.
///
/// Sized from the device's recommended working-set budget, mirroring the CUDA
/// backend's "fits in a quarter of free memory" heuristic. Metal end-to-end
/// tuning is future work; this is a safe default that keeps the GPU busy.
///
/// # Errors
///
/// [`SearchError::Driver`] if no Metal device is present.
pub fn auto_threads(half: u32) -> Result<u32, SearchError> {
    let device = MTLCreateSystemDefaultDevice()
        .ok_or_else(|| SearchError::Driver("no Metal device available".into()))?;
    let budget = device.recommendedMaxWorkingSetSize() / 4;
    let per_thread = local_bytes_per_thread(half).max(1);
    let fit = budget / per_thread;
    let groups = (fit / THREADGROUP as u64).clamp(64, 2048);
    Ok((groups as u32) * THREADGROUP as u32)
}

/// A configured Metal search: compiled kernels, uploaded tables, shared buffers.
pub struct Searcher {
    build_kernel: MslKernel,
    search_kernel: MslKernel,
    half: u32,
    num_threads: u32,
    max_hits: u32,

    // Offset table, built on the device at construction.
    off_table: SharedBuffer,
    giant: SharedBuffer,

    // Pattern tables, uploaded once.
    group_mask: SharedBuffer,
    group_off: SharedBuffer,
    target: SharedBuffer,
    target_pat: SharedBuffer,
    res_off: SharedBuffer,
    res: SharedBuffer,

    // Reused across launches.
    points: SharedBuffer,
    /// The kernel's working copy of `points`, advanced in place as a launch's
    /// dispatches walk (DEC-METAL-009). Kept separate so `points` stays the
    /// caller's input and re-launching without re-uploading repeats a launch
    /// rather than silently continuing from wherever the last one stopped.
    walk: SharedBuffer,
    hits: SharedBuffer,
    hit_count: SharedBuffer,
    status: SharedBuffer,

    // Small single-element scalar buffers, bound as `constant uint&`.
    b_num_threads: SharedBuffer,
    b_num_batches: SharedBuffer,
    b_num_groups: SharedBuffer,
    b_max_hits: SharedBuffer,
    b_batch_base: SharedBuffer,

    /// Seconds per batch, measured from the dispatches already run and used to
    /// size the next one. `None` until the first dispatch has been timed.
    secs_per_batch: Option<f64>,

    /// Hard cap on the batches in one dispatch, over and above the measured
    /// sizing. See [`Searcher::set_max_batches_per_dispatch`].
    max_batches: Option<u32>,

    // Outcome stashed by launch_async for collect (DEC-METAL-007).
    pending: Option<LaunchOutcome>,
}

/// Upload a `u64` table into a fresh shared buffer, padding an empty table to
/// one zero so the buffer is never zero-length.
fn upload_u64(k: &MslKernel, v: &[u64]) -> Result<SharedBuffer, SearchError> {
    let n = v.len().max(1);
    let mut buf = k.new_shared_buffer(n * 8).map_err(msl_err)?;
    let dst = buf.as_mut_slice::<u64>();
    if v.is_empty() {
        dst[0] = 0;
    } else {
        dst.copy_from_slice(v);
    }
    Ok(buf)
}

/// Upload a `u32` table into a fresh shared buffer, padding empty to one zero.
fn upload_u32(k: &MslKernel, v: &[u32]) -> Result<SharedBuffer, SearchError> {
    let n = v.len().max(1);
    let mut buf = k.new_shared_buffer(n * 4).map_err(msl_err)?;
    let dst = buf.as_mut_slice::<u32>();
    if v.is_empty() {
        dst[0] = 0;
    } else {
        dst.copy_from_slice(v);
    }
    Ok(buf)
}

/// A single-element `u32` buffer holding `value`.
fn scalar_u32(k: &MslKernel, value: u32) -> Result<SharedBuffer, SearchError> {
    let mut buf = k.new_shared_buffer(4).map_err(msl_err)?;
    buf.as_mut_slice::<u32>()[0] = value;
    Ok(buf)
}

fn msl_err(e: crate::msl::MslError) -> SearchError {
    SearchError::Driver(format!("{e}"))
}

impl Searcher {
    /// Compile the search kernel for the present device and upload `tables`.
    ///
    /// # Errors
    ///
    /// [`SearchError::BadParameter`] if `half` is zero; [`SearchError::Driver`]
    /// if no Metal device is present or the kernel fails to compile;
    /// [`SearchError::Device`] if building the offset table hits a singular
    /// point.
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

        let lib = MslLibrary::compile(
            crate::msl::sources::SEARCH,
            &[("HALF", half.to_string())],
        )
        .map_err(msl_err)?;
        let build_kernel = lib.kernel("honion_build_offsets").map_err(msl_err)?;
        let search_kernel = lib.kernel("honion_search").map_err(msl_err)?;

        // Pattern tables.
        let group_mask = upload_u64(&search_kernel, &tables.group_mask)?;
        let group_off = upload_u32(&search_kernel, &tables.group_off)?;
        let target = upload_u64(&search_kernel, &tables.target)?;
        let target_pat = upload_u32(&search_kernel, &tables.target_pat)?;
        let res_off = upload_u32(&search_kernel, &tables.res_off)?;
        let res = upload_u64(&search_kernel, &tables.res)?;

        // Offset table and giant step, filled by the build kernel below.
        let off_table = search_kernel
            .new_shared_buffer(half as usize * OFF_STRIDE * 4)
            .map_err(msl_err)?;
        let giant = search_kernel
            .new_shared_buffer(OFF_STRIDE * 4)
            .map_err(msl_err)?;

        let points = search_kernel
            .new_shared_buffer((num_threads as usize * 32).max(1))
            .map_err(msl_err)?;
        let walk = search_kernel
            .new_shared_buffer((num_threads as usize * 32).max(1))
            .map_err(msl_err)?;
        let hits = search_kernel
            .new_shared_buffer((max_hits as usize * std::mem::size_of::<Hit>()).max(1))
            .map_err(msl_err)?;
        let hit_count = scalar_u32(&search_kernel, 0)?;
        let status = scalar_u32(&search_kernel, 0)?;

        let b_num_threads = scalar_u32(&search_kernel, num_threads)?;
        let b_num_batches = scalar_u32(&search_kernel, 0)?;
        let b_num_groups = scalar_u32(&search_kernel, tables.num_groups())?;
        let b_max_hits = scalar_u32(&search_kernel, max_hits)?;
        let b_batch_base = scalar_u32(&search_kernel, 0)?;

        let mut this = Self {
            build_kernel,
            search_kernel,
            half,
            num_threads,
            max_hits,
            off_table,
            giant,
            group_mask,
            group_off,
            target,
            target_pat,
            res_off,
            res,
            points,
            walk,
            hits,
            hit_count,
            status,
            b_num_threads,
            b_num_batches,
            b_num_groups,
            b_max_hits,
            b_batch_base,
            secs_per_batch: None,
            max_batches: None,
            pending: None,
        };

        this.build_offsets()?;
        Ok(this)
    }

    /// Build the offset table on the device. One thread, once.
    fn build_offsets(&mut self) -> Result<(), SearchError> {
        self.status.as_mut_slice::<u32>()[0] = 0;
        self.build_kernel
            .dispatch(1, 1, &[&self.off_table, &self.giant, &self.status])
            .map_err(msl_err)?;
        if self.status.as_slice::<u32>()[0] & STATUS_SINGULAR != 0 {
            return Err(SearchError::Device(
                "building the offset table hit a singular point; \
                 this indicates a field-arithmetic fault"
                    .into(),
            ));
        }
        Ok(())
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

    /// Describe the device this search runs on.
    ///
    /// # Errors
    ///
    /// [`SearchError::Driver`] if no Metal device is present.
    pub fn device_info(&self) -> Result<DeviceInfo, SearchError> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| SearchError::Driver("no Metal device available".into()))?;
        Ok(DeviceInfo {
            backend: "Metal",
            name: device.name().to_string(),
            detail: String::new(),
        })
    }

    /// Install the starting points for the next launches.
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
        let dst = self.points.as_mut_slice::<u8>();
        for (i, p) in points.iter().enumerate() {
            dst[i * 32..i * 32 + 32].copy_from_slice(p);
        }
        Ok(())
    }

    /// Examine at least `candidates` keys on every thread.
    ///
    /// # Errors
    ///
    /// [`SearchError::Device`] if the kernel raised a status flag.
    pub fn launch(&mut self, candidates: u32) -> Result<LaunchOutcome, SearchError> {
        let batches = self.launch_async(candidates)?;
        self.collect(batches)
    }

    /// Run the search and stash the outcome for [`Self::collect`].
    ///
    /// On unified memory there is no transfer to overlap, so this runs the
    /// dispatches synchronously (DEC-METAL-007) and returns the batch count.
    ///
    /// The launch is split across as many dispatches as it takes to keep each
    /// command buffer under [`DISPATCH_TARGET_SECS`] (DEC-METAL-009); see
    /// [`Self::chunk_batches`].
    ///
    /// # Errors
    ///
    /// As [`Self::launch`].
    pub fn launch_async(&mut self, candidates: u32) -> Result<u32, SearchError> {
        let per_batch = self.candidates_per_batch();
        let num_batches = candidates.div_ceil(per_batch).max(1);

        // Reset counters. Both are accumulated across this launch's dispatches:
        // hit_count keeps counting up and status keeps OR-ing in, so the
        // outcome read at the end covers the whole launch, not its last chunk.
        self.hit_count.as_mut_slice::<u32>()[0] = 0;
        self.status.as_mut_slice::<u32>()[0] = 0;

        // The kernel walks `walk` in place, so start it from the caller's
        // points. On unified memory this is a plain memcpy of 32 bytes per
        // thread, and it is what keeps `points` meaning "where this launch
        // starts" no matter how many dispatches the launch takes.
        // Disjoint fields, so both borrows coexist and no staging copy is
        // needed: this is one memcpy, not an allocation per launch.
        let src: &[u8] = self.points.as_slice::<u8>();
        let dst: &mut [u8] = self.walk.as_mut_slice::<u8>();
        dst.copy_from_slice(src);

        let tg = self.search_kernel.max_threads_per_threadgroup().min(THREADGROUP);
        let mut done = 0u32;
        while done < num_batches {
            let chunk = self.chunk_batches(num_batches - done);
            self.b_num_batches.as_mut_slice::<u32>()[0] = chunk;
            self.b_batch_base.as_mut_slice::<u32>()[0] = done;

            let started = std::time::Instant::now();
            // Buffer order matches the [[buffer(N)]] indices in honion_search.
            self.search_kernel
                .dispatch(
                    self.num_threads as usize,
                    tg,
                    &[
                        &self.walk,
                        &self.b_num_threads,
                        &self.b_num_batches,
                        &self.off_table,
                        &self.giant,
                        &self.b_num_groups,
                        &self.group_mask,
                        &self.group_off,
                        &self.target,
                        &self.target_pat,
                        &self.res_off,
                        &self.res,
                        &self.hits,
                        &self.hit_count,
                        &self.b_max_hits,
                        &self.status,
                        &self.b_batch_base,
                    ],
                )
                .map_err(msl_err)?;
            self.record_dispatch(chunk, started.elapsed().as_secs_f64());
            done += chunk;
        }

        let outcome = self.read_outcome(num_batches, per_batch)?;
        self.pending = Some(outcome);
        Ok(num_batches)
    }

    /// How many of the `remaining` batches to put in the next command buffer.
    ///
    /// Before anything has been timed this is [`PROBE_BATCHES`] — small enough
    /// to finish quickly even on a slow device, which is the whole point: the
    /// first dispatch of the process is the one with no measurement behind it.
    /// After that it is whatever the measured rate says fits in
    /// [`DISPATCH_TARGET_SECS`].
    fn chunk_batches(&self, remaining: u32) -> u32 {
        let want = match self.secs_per_batch {
            None => PROBE_BATCHES,
            Some(secs) if secs > 0.0 => {
                let fits = DISPATCH_TARGET_SECS / secs;
                // `fits` is finite and positive here, so the clamp lands in
                // range before the cast.
                fits.clamp(1.0, f64::from(u32::MAX)) as u32
            }
            // Too fast to measure: no reason to split at all.
            Some(_) => u32::MAX,
        };
        want.min(self.max_batches.unwrap_or(u32::MAX)).min(remaining).max(1)
    }

    /// Cap how many batches any single dispatch may run, below whatever the
    /// measured rate would have chosen.
    ///
    /// The measured sizing already keeps a command buffer short enough for the
    /// watchdog on the devices this has been run on (DEC-METAL-009), so this is
    /// not needed in normal use. It exists for two cases: a device that is
    /// killed sooner than the measurement predicts — one also driving a display,
    /// or shared with another process — and the tests, which use it to force a
    /// launch to split many ways and check that doing so changes nothing.
    ///
    /// `None`, the default, leaves sizing entirely to the measurement.
    pub fn set_max_batches_per_dispatch(&mut self, cap: Option<u32>) {
        self.max_batches = cap.map(|c| c.max(1));
    }

    /// Fold a finished dispatch into the per-batch rate estimate.
    ///
    /// The estimate tracks the most recent dispatch rather than averaging: a
    /// long launch can span a clock change or a thermal shift, and the next
    /// chunk should be sized by how the device is behaving now. Dispatches too
    /// short to time are ignored, since dividing by their duration would put
    /// the estimate somewhere meaningless.
    fn record_dispatch(&mut self, batches: u32, secs: f64) {
        if secs > 0.0 && batches > 0 {
            self.secs_per_batch = Some(secs / f64::from(batches));
        }
    }

    /// Return the outcome stashed by [`Self::launch_async`].
    ///
    /// # Errors
    ///
    /// As [`Self::launch`].
    pub fn collect(&mut self, _num_batches: u32) -> Result<LaunchOutcome, SearchError> {
        self.pending
            .take()
            .ok_or_else(|| SearchError::Driver("collect called before launch_async".into()))
    }

    /// Read hits and status out of the shared buffers after a dispatch.
    fn read_outcome(&self, num_batches: u32, per_batch: u32) -> Result<LaunchOutcome, SearchError> {
        let status = self.status.as_slice::<u32>()[0];
        if status & STATUS_BAD_START_POINT != 0 {
            return Err(SearchError::Device(
                "a starting point failed to decompress; it was not a valid public key".into(),
            ));
        }
        if status & STATUS_SINGULAR != 0 {
            return Err(SearchError::Device(
                "a denominator vanished during the walk, which means a base point \
                 coincided with one of its own offsets; suspect a field-arithmetic \
                 bug or a hardware fault"
                    .into(),
            ));
        }

        let total_found = self.hit_count.as_slice::<u32>()[0];
        let kept = total_found.min(self.max_hits);

        let mut hits = Vec::with_capacity(kept as usize);
        if kept > 0 {
            let raw = self.hits.as_slice::<Hit>();
            hits.extend_from_slice(&raw[..kept as usize]);
        }

        let examined =
            u64::from(self.num_threads) * u64::from(num_batches) * u64::from(per_batch);
        Ok(LaunchOutcome {
            hits,
            total_found,
            examined,
        })
    }
}
