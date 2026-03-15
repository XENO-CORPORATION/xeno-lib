//! Hardware capability detection for GPU-accelerated encoding and decoding.
//!
//! This module detects available hardware encoders/decoders on the user's system
//! by dynamically loading vendor-specific libraries at runtime. No compile-time
//! dependency on CUDA, Intel Media SDK, or AMD AMF is required.
//!
//! # Supported Hardware
//!
//! - **NVIDIA**: Detects GPU name, driver version, NVENC/NVDEC availability,
//!   CUDA version, and VRAM via the CUDA Driver API (`nvcuda.dll` / `libcuda.so`)
//! - **Intel**: Detects Quick Sync Video (QSV) support via Intel Media SDK
//!   library presence (`libmfx` / `mfx.dll`)
//! - **AMD**: Detects Advanced Media Framework (AMF) support via AMD runtime
//!   library presence (`amfrt64.dll` / `libamfrt64.so`)
//!
//! # Example
//!
//! ```
//! use xeno_lib::hardware::{detect_hardware, get_supported_codecs};
//!
//! let caps = detect_hardware();
//! if let Some(ref nvidia) = caps.nvidia {
//!     println!("NVIDIA GPU: {} (VRAM: {} MB)", nvidia.gpu_name, nvidia.vram_mb);
//!     println!("  NVENC: {}, NVDEC: {}", nvidia.nvenc_available, nvidia.nvdec_available);
//! }
//!
//! let codecs = get_supported_codecs(&caps);
//! println!("H.264 HW encode: {}", codecs.h264.hardware_encode);
//! println!("H.265 HW encode: {}", codecs.h265.hardware_encode);
//! ```

mod detect;

pub use detect::{
    detect_hardware, get_supported_codecs, AmdInfo, CodecCapability, CodecSupport,
    HardwareCapabilities, IntelInfo, NvidiaInfo,
};
