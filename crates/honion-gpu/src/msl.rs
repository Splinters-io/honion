//! Compiling and launching Metal device code at run time.
//!
//! # Why run-time compilation
//!
//! This mirrors the design of [`crate::nvrtc`] on the CUDA side, for the same
//! reasons: the build needs no offline shader toolchain, the kernel can be
//! specialised per run by baking constants in as `-D` defines, and the source
//! the device compiler sees is fixed when the binary is built rather than read
//! from disk. Metal's `newLibraryWithSource:options:error:` is the structural
//! analog of NVRTC's `compile_ptx` — it compiles a Metal Shading Language
//! string in process and hands back a loadable library.
//!
//! # The source tree is embedded, not read from disk
//!
//! Device sources are pulled in with `include_str!` at build time (see
//! [`sources`]). A binary therefore carries the exact kernels it was built
//! with and cannot be made to compile something else by changing files next to
//! it — the device source is not an input the program parses at run time
//! (langsec rule 4).
//!
//! # The include list is closed
//!
//! [`expand_includes`] resolves `#include "..."` against [`known_headers`] and
//! nothing else. An unknown include is an error, not a filesystem lookup. Metal
//! itself would not resolve a user `#include` from a source string anyway, but
//! doing the expansion here means the closed-set property is enforced by *our*
//! code — auditable and identical to the CUDA path — rather than relying on a
//! platform behaviour. `<metal_stdlib>` and other angle-bracket system headers
//! are left untouched for the Metal compiler to resolve; only quoted includes
//! are ours to close over.

// @decision DEC-METAL-005
// @title Runtime MSL compilation with host-side closed-include concatenation
// @status accepted
// @rationale The CUDA backend earns three things from compiling at run time —
//   no toolkit dependency, per-run specialisation, and a fixed source set — and
//   the Metal backend must not regress any of them. `newLibraryWithSource` is
//   the in-process MSL compiler; `expand_includes` reproduces `nvrtc.rs`'s
//   `known_headers`/`expand_into` verbatim in spirit (recursive, emit-once,
//   unknown-include is an error) so the langsec boundary is the same on both
//   platforms. Angle-bracket system includes are passed through because
//   `<metal_stdlib>` is the standard library, not project source; only quoted
//   includes name our headers and only those are closed over.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

/// The device sources, embedded at build time.
///
/// Waves 3-5 add `fe25519`, `ge25519` and `search` here alongside the probe
/// kernel; the probe exists so the driver is testable before they do.
pub mod sources {
    /// A trivial compute kernel used only to exercise the driver.
    pub const PROBE: &str = include_str!("../metal/probe.metal");
    /// Shared definitions the probe kernel includes by name.
    pub const PROBE_COMMON: &str = include_str!("../metal/probe_common.metal");
    /// Field arithmetic in GF(2^255 - 19), radix 2^25.5 (DEC-METAL-004).
    pub const FE25519: &str = include_str!("../metal/fe25519.metal");
    /// Group arithmetic on the Ed25519 curve, including the dual addition law.
    pub const GE25519: &str = include_str!("../metal/ge25519.metal");
    /// The vanity search kernel and its offset-table builder.
    pub const SEARCH: &str = include_str!("../metal/search.metal");
    /// Test-only kernels exposing the field and group primitives one at a time.
    pub const TESTKERNELS: &str = include_str!("../metal/testkernels.metal");
}

/// The quoted headers that `#include "..."` may name.
///
/// A closed list, deliberately: an include that is not here is an error, so the
/// set of text that can reach the Metal compiler is fixed at build time and
/// cannot be widened by anything the program reads at run time.
fn known_headers() -> BTreeMap<&'static str, &'static str> {
    [
        ("probe_common.metal", sources::PROBE_COMMON),
        ("fe25519.metal", sources::FE25519),
        ("ge25519.metal", sources::GE25519),
    ]
    .into_iter()
    .collect()
}

/// Expand quoted includes recursively, emitting each header at most once.
///
/// Angle-bracket includes (`#include <metal_stdlib>`) are passed through
/// untouched — they name the Metal standard library, which is the compiler's to
/// resolve, not ours. Only quoted includes are resolved against
/// [`known_headers`]; an unrecognised one is [`MslError::UnknownInclude`].
///
/// `seen` is threaded through the recursion rather than recreated per call, so
/// a header reached by two paths is still emitted once — the effect `#pragma
/// once` has in a real preprocessor, which we get here by construction.
fn expand_into(
    source: &str,
    seen: &mut Vec<&'static str>,
    out: &mut String,
) -> Result<(), MslError> {
    let known = known_headers();
    for line in source.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("#include \"") else {
            out.push_str(line);
            out.push('\n');
            continue;
        };
        let Some(name) = rest.split('"').next() else {
            return Err(MslError::MalformedInclude {
                line: line.to_owned(),
            });
        };
        let Some((canonical, body)) = known.get_key_value(name) else {
            return Err(MslError::UnknownInclude {
                name: name.to_owned(),
            });
        };
        if seen.contains(canonical) {
            continue;
        }
        seen.push(canonical);
        expand_into(body, seen, out)?;
    }
    Ok(())
}

/// Expand every quoted include in `source`, emitting each header at most once.
///
/// # Errors
///
/// [`MslError::UnknownInclude`] if a quoted include names a header outside the
/// closed set; [`MslError::MalformedInclude`] if an include line cannot be
/// parsed.
pub fn expand_includes(source: &str) -> Result<String, MslError> {
    let mut seen: Vec<&'static str> = Vec::new();
    let mut out = String::with_capacity(source.len() * 3);
    expand_into(source, &mut seen, &mut out)?;
    Ok(out)
}

/// Prepend `-D name=value`-style defines as `#define` lines.
///
/// Metal's `MTLCompileOptions` has no define list the way NVRTC's options do,
/// so specialisation constants are injected as text ahead of the source. This
/// runs *after* include expansion, so a define is visible to every header.
fn inject_defines(defines: &[(&str, String)], source: &str) -> String {
    let mut out = String::with_capacity(source.len() + defines.len() * 32);
    for (name, value) in defines {
        out.push_str("#define ");
        out.push_str(name);
        out.push(' ');
        out.push_str(value);
        out.push('\n');
    }
    out.push_str(source);
    out
}

/// A compiled compute pipeline: the device, a queue, and one ready-to-dispatch
/// kernel.
///
/// Holds everything needed to run one kernel repeatedly. Built once (compiling
/// MSL and creating the pipeline state are the expensive steps), then launched
/// many times. Not `Send`: Metal objects are tied to the thread that made them,
/// and the search loop that uses this is single-threaded, exactly as the CUDA
/// `Searcher` is driven.
pub struct MslKernel {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

/// A compiled MSL library: the device, a queue, and a library that may hold
/// several kernel functions.
///
/// Compiling the source (include expansion, define injection, the Metal
/// compiler) is the expensive step and is done once here; [`Self::kernel`] then
/// builds a ready-to-dispatch [`MslKernel`] per named function, sharing this
/// library's device and queue. This mirrors the CUDA path, where one NVRTC
/// module yields both `honion_search` and `honion_build_offsets`. Not `Send`,
/// for the same reason as [`MslKernel`].
pub struct MslLibrary {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl MslLibrary {
    /// Compile `source` with `defines` injected as compile constants.
    ///
    /// # Errors
    ///
    /// [`MslError::NoDevice`] if the system has no Metal device;
    /// [`MslError::UnknownInclude`]/[`MslError::MalformedInclude`] from include
    /// expansion; [`MslError::Compile`] if the Metal compiler rejects the
    /// source; [`MslError::NoQueue`] if a command queue cannot be created.
    pub fn compile(source: &str, defines: &[(&str, String)]) -> Result<Self, MslError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MslError::NoDevice)?;

        let expanded = expand_includes(source)?;
        let full = inject_defines(defines, &expanded);
        let src = NSString::from_str(&full);

        // IEEE-safe math. Fast/relaxed modes reassociate and approximate float
        // ops, which is irrelevant to integer field arithmetic but a needless
        // source of surprise if float ever appears. `Safe` is Metal's default
        // for source compiles; setting it explicitly keeps that guarantee even
        // if the default shifts.
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);

        let library = device
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|e| MslError::Compile {
                message: nserror_string(&e),
            })?;

        let queue = device.newCommandQueue().ok_or(MslError::NoQueue)?;

        Ok(Self {
            device,
            queue,
            library,
        })
    }

    /// Build a compute pipeline for the function named `function`.
    ///
    /// # Errors
    ///
    /// [`MslError::FunctionNotFound`] if the named function is absent;
    /// [`MslError::Pipeline`] if the pipeline cannot be built.
    pub fn kernel(&self, function: &str) -> Result<MslKernel, MslError> {
        let name = NSString::from_str(function);
        let func = self
            .library
            .newFunctionWithName(&name)
            .ok_or_else(|| MslError::FunctionNotFound {
                name: function.to_owned(),
            })?;

        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| MslError::Pipeline {
                message: nserror_string(&e),
            })?;

        // Retained clones bump the refcount; the kernel keeps the device and
        // queue alive independently of this library.
        Ok(MslKernel {
            device: self.device.clone(),
            queue: self.queue.clone(),
            pipeline,
        })
    }

    /// The device this library was compiled for.
    #[must_use]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }
}

impl MslKernel {
    /// Compile `source` and build a pipeline for the function named
    /// `function`, with `defines` injected as compile constants.
    ///
    /// A convenience for the common single-kernel case; equivalent to
    /// [`MslLibrary::compile`] followed by [`MslLibrary::kernel`].
    ///
    /// # Errors
    ///
    /// As [`MslLibrary::compile`] and [`MslLibrary::kernel`].
    pub fn compile(
        source: &str,
        function: &str,
        defines: &[(&str, String)],
    ) -> Result<Self, MslError> {
        MslLibrary::compile(source, defines)?.kernel(function)
    }

    /// The device this kernel runs on.
    #[must_use]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// The maximum threads-per-threadgroup this pipeline supports.
    ///
    /// The analog of choosing a CUDA block size: a dispatch's threadgroup width
    /// must not exceed this.
    #[must_use]
    pub fn max_threads_per_threadgroup(&self) -> usize {
        self.pipeline.maxTotalThreadsPerThreadgroup()
    }

    /// Allocate a `storageModeShared` buffer of `len` bytes, zero-initialised.
    ///
    /// Shared storage means the CPU and GPU see the same memory: on Apple
    /// Silicon there is no separate device address space, so there is no
    /// host-to-device copy to schedule or overlap — the discrete-GPU transfer
    /// machinery the CUDA path carries (see `docs/06-performance.md`) simply
    /// does not apply here.
    ///
    /// # Errors
    ///
    /// [`MslError::Allocation`] if the device cannot allocate the buffer.
    pub fn new_shared_buffer(&self, len: usize) -> Result<SharedBuffer, MslError> {
        let buf = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or(MslError::Allocation { bytes: len })?;
        // Shared buffers are not guaranteed zeroed; do it so callers can rely
        // on a clean slate the way `alloc_zeros` gives on CUDA.
        // Safety: `contents()` points to `len` bytes of shared, CPU-visible
        // memory owned by `buf`, which outlives this write.
        unsafe {
            std::ptr::write_bytes(buf.contents().as_ptr().cast::<u8>(), 0, len);
        }
        Ok(SharedBuffer { buffer: buf, len })
    }

    /// Dispatch the kernel over `grid` threads, with `buffers` bound to indices
    /// `0, 1, …`, and block until it completes.
    ///
    /// The grid is rounded up to a whole number of threadgroups of width
    /// `threadgroup` (capped at [`Self::max_threads_per_threadgroup`]); the
    /// kernel is responsible for bounds-checking its thread index against the
    /// real work count, exactly as the CUDA kernel does.
    ///
    /// # Errors
    ///
    /// [`MslError::BadParameter`] if `threadgroup` is zero or exceeds the
    /// pipeline maximum, or if no buffers are bound; [`MslError::Encoder`] if a
    /// command encoder cannot be created; [`MslError::Dispatch`] if the command
    /// buffer faulted on the device instead of running to completion.
    pub fn dispatch(
        &self,
        grid: usize,
        threadgroup: usize,
        buffers: &[&SharedBuffer],
    ) -> Result<(), MslError> {
        if threadgroup == 0 || threadgroup > self.max_threads_per_threadgroup() {
            return Err(MslError::BadParameter(format!(
                "threadgroup width {threadgroup} must be in 1..={}",
                self.max_threads_per_threadgroup()
            )));
        }
        if buffers.is_empty() {
            return Err(MslError::BadParameter(
                "a dispatch must bind at least one buffer".into(),
            ));
        }

        let command_buffer = self.queue.commandBuffer().ok_or(MslError::Encoder {
            what: "command buffer",
        })?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MslError::Encoder {
                what: "compute command encoder",
            })?;
        encoder.setComputePipelineState(&self.pipeline);
        for (i, b) in buffers.iter().enumerate() {
            // Safety: each buffer outlives the encoding, and index `i` matches
            // the kernel's `[[buffer(i)]]` binding by construction of the call.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&b.buffer), 0, i);
            }
        }

        let groups = grid.div_ceil(threadgroup).max(1);
        let grid_size = MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: threadgroup,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // A command buffer that faults on the GPU — timeout, memory fault,
        // device removal — still returns from `waitUntilCompleted` normally.
        // Its status is the *only* signal that the kernel did not run; the
        // output buffer is simply left untouched. Returning `Ok(())` here
        // would hand the caller a zero-filled buffer indistinguishable from a
        // legitimate all-zero result, so the failure has to be checked and
        // surfaced rather than inferred from the data.
        let status = command_buffer.status();
        if status != MTLCommandBufferStatus::Completed {
            return Err(MslError::Dispatch {
                status: status.0,
                message: command_buffer
                    .error()
                    .map_or_else(|| "no further detail".to_owned(), |e| nserror_string(&e)),
            });
        }
        Ok(())
    }
}

/// A `storageModeShared` buffer, visible to both CPU and GPU with no copy.
pub struct SharedBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
}

// The MTLBuffer trait is only needed for the `contents()`/`length()` calls in
// this impl block.
use objc2_metal::MTLBuffer;

impl SharedBuffer {
    /// Length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// View the buffer's contents as a slice of `T`.
    ///
    /// # Panics
    ///
    /// If the byte length is not a whole number of `T`.
    #[must_use]
    pub fn as_slice<T: Copy>(&self) -> &[T] {
        let elem = std::mem::size_of::<T>();
        assert!(elem > 0 && self.len % elem == 0, "buffer length not a multiple of element size");
        let ptr: NonNull<c_void> = self.buffer.contents();
        // Safety: shared storage keeps this pointer valid for the buffer's
        // lifetime, which outlives the borrow; the length divides evenly into
        // `T` by the assertion above; and the buffer was allocated by Metal
        // with suitable alignment for scalar element types.
        unsafe { std::slice::from_raw_parts(ptr.as_ptr().cast::<T>(), self.len / elem) }
    }

    /// View the buffer's contents as a mutable slice of `T`.
    ///
    /// # Panics
    ///
    /// If the byte length is not a whole number of `T`.
    pub fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        let elem = std::mem::size_of::<T>();
        assert!(elem > 0 && self.len % elem == 0, "buffer length not a multiple of element size");
        let ptr: NonNull<c_void> = self.buffer.contents();
        // Safety: as `as_slice`, and `&mut self` guarantees no aliasing view of
        // the same buffer exists for the borrow.
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr().cast::<T>(), self.len / elem) }
    }
}

/// Render an `NSError` as a human-readable string for a compile/pipeline
/// failure message.
fn nserror_string(err: &objc2_foundation::NSError) -> String {
    err.localizedDescription().to_string()
}

/// Why Metal device code could not be compiled or run.
#[derive(Debug, Clone, thiserror::Error)]
pub enum MslError {
    /// No Metal device is present on this system.
    #[error("no Metal device is available on this system")]
    NoDevice,
    /// The device could not create a command queue.
    #[error("could not create a Metal command queue")]
    NoQueue,
    /// A quoted `#include` named a header outside the embedded set.
    #[error("device source includes unknown header {name:?}; only the embedded headers resolve")]
    UnknownInclude {
        /// The unresolvable header name.
        name: String,
    },
    /// A quoted `#include` line could not be parsed.
    #[error("malformed include directive: {line}")]
    MalformedInclude {
        /// The offending line.
        line: String,
    },
    /// The Metal compiler rejected the source.
    #[error("Metal compilation failed:\n{message}")]
    Compile {
        /// The compiler diagnostic.
        message: String,
    },
    /// The compiled library had no function of the requested name.
    #[error("compiled library has no function named {name:?}")]
    FunctionNotFound {
        /// The missing function name.
        name: String,
    },
    /// The compute pipeline could not be built from the function.
    #[error("could not build a compute pipeline:\n{message}")]
    Pipeline {
        /// The pipeline-creation diagnostic.
        message: String,
    },
    /// A shared buffer allocation failed.
    #[error("could not allocate a {bytes}-byte shared buffer")]
    Allocation {
        /// The requested size.
        bytes: usize,
    },
    /// A command buffer or encoder could not be created.
    #[error("could not create a Metal {what}")]
    Encoder {
        /// Which object failed to allocate.
        what: &'static str,
    },
    /// A dispatch parameter was out of range.
    #[error("{0}")]
    BadParameter(String),
    /// The command buffer did not complete: the kernel faulted on the device.
    #[error("Metal dispatch did not complete (command buffer status {status}): {message}")]
    Dispatch {
        /// The raw `MTLCommandBufferStatus` the buffer ended in.
        status: usize,
        /// The command buffer's error description, when it carried one.
        message: String,
    },
}
