//! Compile every device source for the installed GPU and report PTX size.
//!
//! A fast check that the kernels build for the machine's actual architecture,
//! without running anything or needing test fixtures.
use cudarc::driver::CudaContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `DriverError` does not implement `std::error::Error`, so it is rendered
    // rather than propagated.
    let arch = match CudaContext::new(0) {
        Ok(ctx) => {
            let (major, minor) = ctx
                .compute_capability()
                .map_err(|e| format!("querying compute capability: {e:?}"))?;
            println!("device compute capability {major}.{minor}");
            (major as u32, minor as u32)
        }
        Err(e) => {
            println!("no CUDA device ({e:?}); compiling for compute_120 instead");
            (12, 0)
        }
    };
    for (name, src) in [
        ("search.cu", honion_gpu::nvrtc::sources::SEARCH),
        ("testkernels.cu", honion_gpu::nvrtc::sources::TESTKERNELS),
    ] {
        let t = std::time::Instant::now();
        let ptx = honion_gpu::nvrtc::compile(src, arch, &[])?;
        println!("{name:<16} {:>8.2?}  {:>6} KB PTX", t.elapsed(), ptx.len() / 1024);
    }
    Ok(())
}
