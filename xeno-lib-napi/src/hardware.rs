//! N-API bindings for hardware capability detection.
//!
//! Exposes GPU/encoder detection to Node.js / Electron so the app can
//! display available hardware and choose optimal encode/decode paths.

use napi_derive::napi;

/// NVIDIA GPU information exposed to JavaScript.
#[napi(object)]
pub struct NvidiaInfoJs {
    /// GPU device name (e.g., "NVIDIA GeForce RTX 4090").
    pub gpu_name: String,
    /// Driver version string (e.g., "560.35").
    pub driver_version: String,
    /// Whether NVENC (hardware video encoding) is available.
    pub nvenc_available: bool,
    /// Whether NVDEC (hardware video decoding) is available.
    pub nvdec_available: bool,
    /// Video RAM in megabytes.
    pub vram_mb: f64,
}

/// Intel GPU information exposed to JavaScript.
#[napi(object)]
pub struct IntelInfoJs {
    /// GPU device name.
    pub gpu_name: String,
    /// Whether Intel Quick Sync Video is available.
    pub qsv_available: bool,
}

/// AMD GPU information exposed to JavaScript.
#[napi(object)]
pub struct AmdInfoJs {
    /// GPU device name.
    pub gpu_name: String,
    /// Whether AMD Advanced Media Framework is available.
    pub amf_available: bool,
}

/// Hardware acceleration capabilities detected on the current system.
#[napi(object)]
pub struct HardwareCapabilitiesJs {
    /// NVIDIA GPU info, or null if no NVIDIA GPU is detected.
    pub nvidia: Option<NvidiaInfoJs>,
    /// Intel GPU info, or null if no Intel GPU is detected.
    pub intel: Option<IntelInfoJs>,
    /// AMD GPU info, or null if no AMD GPU is detected.
    pub amd: Option<AmdInfoJs>,
}

/// Codec capability for a single codec.
#[napi(object)]
pub struct CodecCapabilityJs {
    /// Whether software encoding is available.
    pub encode: bool,
    /// Whether software decoding is available.
    pub decode: bool,
    /// Whether hardware-accelerated encoding is available.
    pub hardware_encode: bool,
    /// Whether hardware-accelerated decoding is available.
    pub hardware_decode: bool,
}

/// Codec support for all known codecs.
#[napi(object)]
pub struct CodecSupportJs {
    /// H.264/AVC support.
    pub h264: CodecCapabilityJs,
    /// H.265/HEVC support.
    pub h265: CodecCapabilityJs,
    /// AV1 support.
    pub av1: CodecCapabilityJs,
    /// VP9 support.
    pub vp9: CodecCapabilityJs,
}

/// Detect available hardware acceleration on the current system.
///
/// Dynamically loads vendor-specific libraries (CUDA, Intel Media SDK, AMD AMF)
/// to detect available GPUs and their encoding/decoding capabilities.
///
/// Returns `null` for each vendor if no hardware is detected.
/// This function never throws — missing hardware is reported as `null` fields.
///
/// # Example (JavaScript)
///
/// ```js
/// const { detectHardware } = require('@xeno/lib');
/// const hw = detectHardware();
/// if (hw.nvidia) {
///   console.log(`NVIDIA: ${hw.nvidia.gpuName}, VRAM: ${hw.nvidia.vramMb} MB`);
/// }
/// ```
#[napi]
pub fn detect_hardware() -> HardwareCapabilitiesJs {
    let caps = xeno_lib::hardware::detect_hardware();

    HardwareCapabilitiesJs {
        nvidia: caps.nvidia.map(|n| NvidiaInfoJs {
            gpu_name: n.gpu_name,
            driver_version: n.driver_version,
            nvenc_available: n.nvenc_available,
            nvdec_available: n.nvdec_available,
            vram_mb: n.vram_mb as f64,
        }),
        intel: caps.intel.map(|i| IntelInfoJs {
            gpu_name: i.gpu_name,
            qsv_available: i.qsv_available,
        }),
        amd: caps.amd.map(|a| AmdInfoJs {
            gpu_name: a.gpu_name,
            amf_available: a.amf_available,
        }),
    }
}

/// Get supported codecs based on compiled features and detected hardware.
///
/// Returns information about which codecs can encode/decode and whether
/// hardware acceleration is available for each.
///
/// This function calls `detectHardware()` internally, so there's no need
/// to call both — use this function if you only need codec information.
///
/// # Example (JavaScript)
///
/// ```js
/// const { getSupportedCodecs } = require('@xeno/lib');
/// const codecs = getSupportedCodecs();
/// console.log(`H.264 HW encode: ${codecs.h264.hardwareEncode}`);
/// console.log(`H.265 HW decode: ${codecs.h265.hardwareDecode}`);
/// ```
#[napi]
pub fn get_supported_codecs() -> CodecSupportJs {
    let hw = xeno_lib::hardware::detect_hardware();
    let codecs = xeno_lib::hardware::get_supported_codecs(&hw);

    CodecSupportJs {
        h264: CodecCapabilityJs {
            encode: codecs.h264.encode,
            decode: codecs.h264.decode,
            hardware_encode: codecs.h264.hardware_encode,
            hardware_decode: codecs.h264.hardware_decode,
        },
        h265: CodecCapabilityJs {
            encode: codecs.h265.encode,
            decode: codecs.h265.decode,
            hardware_encode: codecs.h265.hardware_encode,
            hardware_decode: codecs.h265.hardware_decode,
        },
        av1: CodecCapabilityJs {
            encode: codecs.av1.encode,
            decode: codecs.av1.decode,
            hardware_encode: codecs.av1.hardware_encode,
            hardware_decode: codecs.av1.hardware_decode,
        },
        vp9: CodecCapabilityJs {
            encode: codecs.vp9.encode,
            decode: codecs.vp9.decode,
            hardware_encode: codecs.vp9.hardware_encode,
            hardware_decode: codecs.vp9.hardware_decode,
        },
    }
}
