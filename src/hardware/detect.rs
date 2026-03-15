//! Hardware encoder/decoder detection via dynamic library loading.
//!
//! Detects NVIDIA (NVENC/NVDEC), Intel (QSV), and AMD (AMF) hardware
//! acceleration by loading vendor libraries at runtime. No compile-time
//! dependencies on CUDA SDK, Intel Media SDK, or AMD AMF SDK are required.

use std::ffi::CStr;

/// Information about available hardware acceleration capabilities.
#[derive(Debug, Clone, Default)]
pub struct HardwareCapabilities {
    /// NVIDIA GPU information, if an NVIDIA GPU is detected.
    pub nvidia: Option<NvidiaInfo>,
    /// Intel GPU information, if an Intel GPU with QSV is detected.
    pub intel: Option<IntelInfo>,
    /// AMD GPU information, if an AMD GPU with AMF is detected.
    pub amd: Option<AmdInfo>,
}

/// NVIDIA GPU information.
#[derive(Debug, Clone)]
pub struct NvidiaInfo {
    /// GPU device name (e.g., "NVIDIA GeForce RTX 4090").
    pub gpu_name: String,
    /// Driver version string (e.g., "560.35.03").
    pub driver_version: String,
    /// Whether NVENC (hardware video encoding) is available.
    pub nvenc_available: bool,
    /// Whether NVDEC (hardware video decoding) is available.
    pub nvdec_available: bool,
    /// CUDA version if available (e.g., "12.4").
    pub cuda_version: Option<String>,
    /// Video RAM in megabytes.
    pub vram_mb: u64,
}

/// Intel GPU information.
#[derive(Debug, Clone)]
pub struct IntelInfo {
    /// GPU device name (e.g., "Intel UHD Graphics 770").
    pub gpu_name: String,
    /// Whether Intel Quick Sync Video is available.
    pub qsv_available: bool,
}

/// AMD GPU information.
#[derive(Debug, Clone)]
pub struct AmdInfo {
    /// GPU device name (e.g., "AMD Radeon RX 7900 XTX").
    pub gpu_name: String,
    /// Whether AMD Advanced Media Framework is available.
    pub amf_available: bool,
}

/// Codec capability information for a single codec.
#[derive(Debug, Clone, Default)]
pub struct CodecCapability {
    /// Whether software encoding is available.
    pub encode: bool,
    /// Whether software decoding is available.
    pub decode: bool,
    /// Whether hardware-accelerated encoding is available.
    pub hardware_encode: bool,
    /// Whether hardware-accelerated decoding is available.
    pub hardware_decode: bool,
}

/// Supported codec information for all codecs.
#[derive(Debug, Clone, Default)]
pub struct CodecSupport {
    /// H.264/AVC codec support.
    pub h264: CodecCapability,
    /// H.265/HEVC codec support.
    pub h265: CodecCapability,
    /// AV1 codec support.
    pub av1: CodecCapability,
    /// VP9 codec support.
    pub vp9: CodecCapability,
}

// CUDA Driver API constants for device attribute queries.
// These match the CUdevice_attribute enum in cuda.h.
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

// CUresult success code.
const CUDA_SUCCESS: i32 = 0;

/// Detect all available hardware acceleration capabilities.
///
/// This function dynamically loads vendor-specific libraries to detect
/// GPU hardware. It will not fail if no GPU is present — it simply returns
/// `None` for each vendor.
///
/// # Platform Behavior
///
/// - **Windows**: Loads `nvcuda.dll`, `nvEncodeAPI64.dll`, `mfx.dll`, `amfrt64.dll`
/// - **Linux**: Loads `libcuda.so`, `libnvidia-encode.so`, `libmfx.so`, `libamfrt64.so`
/// - **macOS**: Only checks for Intel QSV (no NVIDIA/AMD GPU support on macOS)
///
/// # Example
///
/// ```
/// use xeno_lib::hardware::detect_hardware;
///
/// let caps = detect_hardware();
/// if let Some(nvidia) = &caps.nvidia {
///     println!("Found NVIDIA GPU: {}", nvidia.gpu_name);
/// }
/// ```
pub fn detect_hardware() -> HardwareCapabilities {
    HardwareCapabilities {
        nvidia: detect_nvidia(),
        intel: detect_intel(),
        amd: detect_amd(),
    }
}

/// Get codec support based on detected hardware and compiled features.
///
/// Combines information about compiled software codecs and detected hardware
/// to produce a complete picture of what codecs are available.
///
/// # Arguments
///
/// * `hw` - Hardware capabilities from `detect_hardware()`
///
/// # Example
///
/// ```
/// use xeno_lib::hardware::{detect_hardware, get_supported_codecs};
///
/// let hw = detect_hardware();
/// let codecs = get_supported_codecs(&hw);
/// println!("Can encode H.264: {}", codecs.h264.encode);
/// ```
pub fn get_supported_codecs(hw: &HardwareCapabilities) -> CodecSupport {
    let has_nvenc = hw
        .nvidia
        .as_ref()
        .map(|n| n.nvenc_available)
        .unwrap_or(false);
    let has_nvdec = hw
        .nvidia
        .as_ref()
        .map(|n| n.nvdec_available)
        .unwrap_or(false);
    let has_qsv = hw.intel.as_ref().map(|i| i.qsv_available).unwrap_or(false);
    let has_amf = hw.amd.as_ref().map(|a| a.amf_available).unwrap_or(false);

    let hw_encode = has_nvenc || has_qsv || has_amf;
    let hw_decode = has_nvdec || has_qsv;

    CodecSupport {
        h264: CodecCapability {
            // OpenH264 is available if compiled with video-encode-h264 / video-decode-sw
            encode: cfg!(feature = "video-encode-h264"),
            decode: cfg!(feature = "video-decode-sw") || has_nvdec,
            hardware_encode: hw_encode,
            hardware_decode: hw_decode,
        },
        h265: CodecCapability {
            encode: false, // No software HEVC encoder yet
            decode: cfg!(feature = "video-decode-hevc") || has_nvdec,
            hardware_encode: hw_encode,
            hardware_decode: hw_decode,
        },
        av1: CodecCapability {
            encode: cfg!(feature = "video-encode"),
            decode: cfg!(feature = "video-decode-sw") || has_nvdec,
            // NVENC AV1 requires RTX 40 series; QSV AV1 requires Arc/12th+ gen
            hardware_encode: has_nvenc || has_qsv || has_amf,
            hardware_decode: has_nvdec || has_qsv,
        },
        vp9: CodecCapability {
            encode: false, // No VP9 encoder
            decode: has_nvdec,
            hardware_encode: false, // No hardware VP9 encoding
            hardware_decode: has_nvdec,
        },
    }
}

/// Detect NVIDIA GPU capabilities via the CUDA Driver API.
///
/// Dynamically loads nvcuda.dll (Windows) or libcuda.so (Linux) and queries
/// device properties. Also checks for NVENC by attempting to load the
/// NVENC library.
fn detect_nvidia() -> Option<NvidiaInfo> {
    // Determine library names based on platform
    #[cfg(target_os = "windows")]
    let cuda_lib_name = "nvcuda.dll";
    #[cfg(target_os = "linux")]
    let cuda_lib_name = "libcuda.so.1";
    #[cfg(target_os = "macos")]
    return None; // No NVIDIA GPU support on macOS

    // SAFETY: We are loading a well-known system library and calling documented
    // CUDA Driver API functions with correct signatures. All function pointers
    // are validated before use. Error codes are checked.
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    {
        let cuda_lib = match unsafe { libloading::Library::new(cuda_lib_name) } {
            Ok(lib) => lib,
            Err(_) => return None, // CUDA not installed
        };

        // Load CUDA Driver API functions
        type CuInit = unsafe extern "C" fn(flags: u32) -> i32;
        type CuDeviceGetCount = unsafe extern "C" fn(count: *mut i32) -> i32;
        type CuDeviceGet = unsafe extern "C" fn(device: *mut i32, ordinal: i32) -> i32;
        type CuDeviceGetName = unsafe extern "C" fn(name: *mut u8, len: i32, dev: i32) -> i32;
        type CuDeviceGetAttribute =
            unsafe extern "C" fn(pi: *mut i32, attrib: i32, dev: i32) -> i32;
        type CuDeviceTotalMem = unsafe extern "C" fn(bytes: *mut usize, dev: i32) -> i32;
        type CuDriverGetVersion = unsafe extern "C" fn(version: *mut i32) -> i32;

        let cu_init: CuInit = match unsafe { cuda_lib.get::<CuInit>(b"cuInit\0") } {
            Ok(f) => *f,
            Err(_) => return None,
        };

        // Initialize CUDA
        let result = unsafe { cu_init(0) };
        if result != CUDA_SUCCESS {
            return None;
        }

        let cu_device_get_count: CuDeviceGetCount =
            match unsafe { cuda_lib.get::<CuDeviceGetCount>(b"cuDeviceGetCount\0") } {
                Ok(f) => *f,
                Err(_) => return None,
            };

        let mut device_count: i32 = 0;
        let result = unsafe { cu_device_get_count(&mut device_count) };
        if result != CUDA_SUCCESS || device_count == 0 {
            return None;
        }

        let cu_device_get: CuDeviceGet =
            match unsafe { cuda_lib.get::<CuDeviceGet>(b"cuDeviceGet\0") } {
                Ok(f) => *f,
                Err(_) => return None,
            };

        let cu_device_get_name: CuDeviceGetName =
            match unsafe { cuda_lib.get::<CuDeviceGetName>(b"cuDeviceGetName\0") } {
                Ok(f) => *f,
                Err(_) => return None,
            };

        let cu_device_get_attribute: CuDeviceGetAttribute =
            match unsafe { cuda_lib.get::<CuDeviceGetAttribute>(b"cuDeviceGetAttribute\0") } {
                Ok(f) => *f,
                Err(_) => return None,
            };

        let cu_device_total_mem: CuDeviceTotalMem =
            match unsafe { cuda_lib.get::<CuDeviceTotalMem>(b"cuDeviceTotalMem_v2\0") } {
                Ok(f) => *f,
                Err(_) => return None,
            };

        let cu_driver_get_version: CuDriverGetVersion =
            match unsafe { cuda_lib.get::<CuDriverGetVersion>(b"cuDriverGetVersion\0") } {
                Ok(f) => *f,
                Err(_) => return None,
            };

        // Get device 0 (primary GPU)
        let mut device: i32 = 0;
        let result = unsafe { cu_device_get(&mut device, 0) };
        if result != CUDA_SUCCESS {
            return None;
        }

        // Get GPU name
        let mut name_buf = [0u8; 256];
        let result = unsafe { cu_device_get_name(name_buf.as_mut_ptr(), 256, device) };
        let gpu_name = if result == CUDA_SUCCESS {
            // SAFETY: CUDA writes a null-terminated C string into name_buf
            unsafe { CStr::from_ptr(name_buf.as_ptr() as *const i8) }
                .to_string_lossy()
                .to_string()
        } else {
            "Unknown NVIDIA GPU".to_string()
        };

        // Get driver version
        let mut driver_version: i32 = 0;
        let result = unsafe { cu_driver_get_version(&mut driver_version) };
        let driver_version_str = if result == CUDA_SUCCESS {
            format!("{}.{}", driver_version / 1000, (driver_version % 1000) / 10)
        } else {
            "Unknown".to_string()
        };

        // Get CUDA version from compute capability
        let mut compute_major: i32 = 0;
        let mut compute_minor: i32 = 0;
        unsafe {
            cu_device_get_attribute(
                &mut compute_major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device,
            );
            cu_device_get_attribute(
                &mut compute_minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device,
            );
        }
        let cuda_version = if compute_major > 0 {
            Some(format!("{}.{}", compute_major, compute_minor))
        } else {
            None
        };

        // Get total VRAM
        let mut total_mem: usize = 0;
        let result = unsafe { cu_device_total_mem(&mut total_mem, device) };
        let vram_mb = if result == CUDA_SUCCESS {
            (total_mem / (1024 * 1024)) as u64
        } else {
            0
        };

        // Check NVENC availability by trying to load the NVENC library
        let nvenc_available = check_nvenc_available();

        // NVDEC is available on all CUDA-capable GPUs with compute capability >= 3.0
        let nvdec_available = compute_major >= 3;

        Some(NvidiaInfo {
            gpu_name,
            driver_version: driver_version_str,
            nvenc_available,
            nvdec_available,
            cuda_version,
            vram_mb,
        })
    }
}

/// Check if NVENC is available by loading the NVENC library.
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn check_nvenc_available() -> bool {
    #[cfg(target_os = "windows")]
    let nvenc_lib_name = "nvEncodeAPI64.dll";
    #[cfg(target_os = "linux")]
    let nvenc_lib_name = "libnvidia-encode.so.1";

    // SAFETY: We are only loading the library to check its presence.
    // No functions are called.
    unsafe { libloading::Library::new(nvenc_lib_name) }.is_ok()
}

/// Detect Intel GPU with Quick Sync Video support.
///
/// Checks for the Intel Media SDK / oneVPL runtime library.
fn detect_intel() -> Option<IntelInfo> {
    #[cfg(target_os = "windows")]
    {
        // Check for Intel Media SDK / oneVPL on Windows
        // Try multiple library names (oneVPL supersedes Media SDK)
        let lib_names = ["libmfx.dll", "mfx.dll", "libvpl.dll"];
        let qsv_available = lib_names.iter().any(|name| {
            // SAFETY: We are only loading the library to check its presence.
            unsafe { libloading::Library::new(name) }.is_ok()
        });

        if !qsv_available {
            // Also check for Intel GPU via DirectX (igdumdim64.dll is the Intel UMD driver)
            let intel_driver =
                unsafe { libloading::Library::new("igdumdim64.dll") }.is_ok();
            if !intel_driver {
                return None;
            }
            // Intel driver found but no Media SDK — limited QSV support
            return Some(IntelInfo {
                gpu_name: "Intel GPU (driver detected, QSV SDK not found)".to_string(),
                qsv_available: false,
            });
        }

        Some(IntelInfo {
            gpu_name: "Intel GPU with Quick Sync Video".to_string(),
            qsv_available: true,
        })
    }

    #[cfg(target_os = "linux")]
    {
        // Check for Intel Media SDK / oneVPL on Linux
        let lib_names = ["libmfx.so.1", "libmfx.so", "libvpl.so.2", "libvpl.so"];
        let qsv_available = lib_names.iter().any(|name| {
            // SAFETY: We are only loading the library to check its presence.
            unsafe { libloading::Library::new(name) }.is_ok()
        });

        if !qsv_available {
            return None;
        }

        Some(IntelInfo {
            gpu_name: "Intel GPU with Quick Sync Video".to_string(),
            qsv_available: true,
        })
    }

    #[cfg(target_os = "macos")]
    {
        // macOS uses VideoToolbox for hardware encoding, not QSV directly
        // Intel Macs have QSV through VideoToolbox, Apple Silicon does not
        #[cfg(target_arch = "x86_64")]
        {
            Some(IntelInfo {
                gpu_name: "Intel GPU (VideoToolbox)".to_string(),
                qsv_available: true,
            })
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            None
        }
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// Detect AMD GPU with Advanced Media Framework support.
///
/// Checks for the AMD AMF runtime library.
fn detect_amd() -> Option<AmdInfo> {
    #[cfg(target_os = "windows")]
    let amf_lib_name = "amfrt64.dll";
    #[cfg(target_os = "linux")]
    let amf_lib_name = "libamfrt64.so.1";
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    return None; // AMD AMF not available on macOS

    #[cfg(any(target_os = "windows", target_os = "linux"))]
    {
        // SAFETY: We are only loading the library to check its presence.
        let amf_available = unsafe { libloading::Library::new(amf_lib_name) }.is_ok();

        if !amf_available {
            return None;
        }

        Some(AmdInfo {
            gpu_name: "AMD GPU with AMF".to_string(),
            amf_available: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_hardware_does_not_panic() {
        // This test verifies that hardware detection doesn't crash,
        // regardless of what hardware is available on the test machine.
        let caps = detect_hardware();
        // Just verify the struct is valid — actual GPU presence varies
        let _ = format!("{:?}", caps);
    }

    #[test]
    fn get_supported_codecs_with_no_hardware() {
        let caps = HardwareCapabilities::default();
        let codecs = get_supported_codecs(&caps);

        // Without hardware, only software codecs should be available
        assert!(!codecs.h264.hardware_encode);
        assert!(!codecs.h264.hardware_decode);
        assert!(!codecs.h265.hardware_encode);
        assert!(!codecs.h265.hardware_decode);
        assert!(!codecs.vp9.hardware_encode);
    }

    #[test]
    fn get_supported_codecs_with_nvidia() {
        let caps = HardwareCapabilities {
            nvidia: Some(NvidiaInfo {
                gpu_name: "Test GPU".to_string(),
                driver_version: "999.99".to_string(),
                nvenc_available: true,
                nvdec_available: true,
                cuda_version: Some("12.0".to_string()),
                vram_mb: 8192,
            }),
            intel: None,
            amd: None,
        };

        let codecs = get_supported_codecs(&caps);
        assert!(codecs.h264.hardware_encode);
        assert!(codecs.h264.hardware_decode);
        assert!(codecs.h265.hardware_encode);
        assert!(codecs.h265.hardware_decode);
        assert!(codecs.h265.decode); // NVDEC provides H.265 decode
        assert!(codecs.av1.hardware_encode);
        assert!(codecs.av1.hardware_decode);
        assert!(codecs.vp9.hardware_decode);
        assert!(!codecs.vp9.hardware_encode); // No HW VP9 encode
    }

    #[test]
    fn get_supported_codecs_with_intel_qsv() {
        let caps = HardwareCapabilities {
            nvidia: None,
            intel: Some(IntelInfo {
                gpu_name: "Intel UHD 770".to_string(),
                qsv_available: true,
            }),
            amd: None,
        };

        let codecs = get_supported_codecs(&caps);
        assert!(codecs.h264.hardware_encode);
        assert!(codecs.h264.hardware_decode);
        assert!(codecs.h265.hardware_encode);
    }

    #[test]
    fn get_supported_codecs_with_amd_amf() {
        let caps = HardwareCapabilities {
            nvidia: None,
            intel: None,
            amd: Some(AmdInfo {
                gpu_name: "AMD RX 7900".to_string(),
                amf_available: true,
            }),
        };

        let codecs = get_supported_codecs(&caps);
        assert!(codecs.h264.hardware_encode);
        assert!(codecs.h265.hardware_encode);
        // AMF doesn't provide decode
        assert!(!codecs.h264.hardware_decode);
    }

    #[test]
    fn hardware_capabilities_default_is_empty() {
        let caps = HardwareCapabilities::default();
        assert!(caps.nvidia.is_none());
        assert!(caps.intel.is_none());
        assert!(caps.amd.is_none());
    }

    #[test]
    fn codec_support_default_all_false() {
        let support = CodecSupport::default();
        assert!(!support.h264.encode);
        assert!(!support.h264.decode);
        assert!(!support.h264.hardware_encode);
        assert!(!support.h264.hardware_decode);
    }

    #[test]
    fn detect_hardware_returns_valid_nvidia_info_if_present() {
        // Integration test: actually tries to detect hardware
        let caps = detect_hardware();
        if let Some(nvidia) = &caps.nvidia {
            assert!(!nvidia.gpu_name.is_empty());
            assert!(!nvidia.driver_version.is_empty());
            // VRAM should be reasonable (at least 256 MB for any GPU)
            assert!(nvidia.vram_mb >= 256, "VRAM too low: {} MB", nvidia.vram_mb);
        }
        // If no NVIDIA GPU, that's fine — test passes
    }
}
