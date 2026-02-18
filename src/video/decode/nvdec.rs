//! NVIDIA NVDEC hardware video decoder implementation.
//!
//! This module provides a safe wrapper around NVIDIA's NVDEC API for
//! GPU-accelerated video decoding using dynamic library loading.
//!
//! # Features
//!
//! - **No compile-time CUDA version constraints** - uses runtime loading
//! - **Graceful fallback** - returns error if CUDA/NVDEC not available
//! - **Works with any CUDA version** - 11.x, 12.x, 13.x, etc.
//!
//! # Requirements
//!
//! - NVIDIA GPU with NVDEC support
//! - NVIDIA driver installed (provides nvcuvid.dll/libnvcuvid.so)
//!
//! # Architecture
//!
//! NVDEC uses a callback-based architecture:
//! 1. VideoParser parses the bitstream and extracts NAL/OBU units
//! 2. Parser callbacks trigger decode operations
//! 3. NVDEC hardware decodes frames to GPU memory
//! 4. Frames are mapped and copied to CPU for processing

use std::collections::VecDeque;
use std::ffi::{c_int, c_uint, c_ulong, c_ulonglong, c_void, c_uchar};
use std::path::Path;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};

use libloading::{Library, Symbol};

use crate::video::VideoError;

use super::{DecodeCodec, DecodedFrame, DecoderCapabilities, DecoderConfig, OutputFormat, VideoDecoder};

// ============================================================================
// CUDA/NVDEC Types (ABI-stable definitions)
// ============================================================================

type CUcontext = *mut c_void;
type CUdevice = c_int;
type CUdeviceptr = c_ulonglong;
type CUvideodecoder = *mut c_void;
type CUvideoparser = *mut c_void;
type CUstream = *mut c_void;
type CUresult = c_int;

const CUDA_SUCCESS: CUresult = 0;

// Codec type enum (matches NVIDIA's cudaVideoCodec)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaVideoCodec {
    Mpeg1 = 0,
    Mpeg2 = 1,
    Mpeg4 = 2,
    Vc1 = 3,
    H264 = 4,
    Jpeg = 5,
    H264Svc = 6,
    H264Mvc = 7,
    Hevc = 8,
    Vp8 = 9,
    Vp9 = 10,
    Av1 = 11,
}

impl From<DecodeCodec> for CudaVideoCodec {
    fn from(codec: DecodeCodec) -> Self {
        match codec {
            DecodeCodec::Av1 => CudaVideoCodec::Av1,
            DecodeCodec::H264 => CudaVideoCodec::H264,
            DecodeCodec::H265 => CudaVideoCodec::Hevc,
            DecodeCodec::Vp8 => CudaVideoCodec::Vp8,
            DecodeCodec::Vp9 => CudaVideoCodec::Vp9,
        }
    }
}

// Chroma format enum
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaVideoChromaFormat {
    Monochrome = 0,
    Yuv420 = 1,
    Yuv422 = 2,
    Yuv444 = 3,
}

// Surface format enum
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaVideoSurfaceFormat {
    Nv12 = 0,
    P016 = 1,
    Yuv444 = 2,
    Yuv444_16Bit = 3,
}

impl From<OutputFormat> for CudaVideoSurfaceFormat {
    fn from(fmt: OutputFormat) -> Self {
        match fmt {
            OutputFormat::Nv12 | OutputFormat::Yuv420 | OutputFormat::Bgra | OutputFormat::Rgba => {
                CudaVideoSurfaceFormat::Nv12
            }
        }
    }
}

// Deinterlace mode
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaVideoDeinterlaceMode {
    Weave = 0,
    Bob = 1,
    Adaptive = 2,
}

// Decoder capabilities struct
#[repr(C)]
#[derive(Debug, Clone, Default)]
struct CuvidDecodeCaps {
    codec_type: c_int,
    chroma_format: c_int,
    bit_depth_minus8: c_uint,
    reserved1: [c_uint; 3],
    is_supported: c_uint,
    num_nv_decs: c_uint,
    output_format_mask: c_uint,
    max_width: c_uint,
    max_height: c_uint,
    min_width: c_uint,
    min_height: c_uint,
    is_hist_supported: c_uint,
    reserved2: [c_uint; 4],
}

// Decoder create info struct
#[repr(C)]
#[derive(Debug, Clone)]
struct CuvidDecodeCreateInfo {
    width: c_ulong,
    height: c_ulong,
    num_decode_surfaces: c_ulong,
    codec_type: c_int,
    chroma_format: c_int,
    creation_flags: c_ulong,
    bit_depth_minus8: c_ulong,
    intra_decode_only: c_ulong,
    max_width: c_ulong,
    max_height: c_ulong,
    display_left: c_int,
    display_top: c_int,
    display_right: c_int,
    display_bottom: c_int,
    output_format: c_int,
    deinterlace_mode: c_int,
    target_width: c_ulong,
    target_height: c_ulong,
    num_output_surfaces: c_ulong,
    video_lock: *mut c_void,
    target_left: c_int,
    target_top: c_int,
    target_right: c_int,
    target_bottom: c_int,
    enable_histogram: c_ulong,
}

// Video format info (returned by sequence callback)
#[repr(C)]
#[derive(Debug, Clone, Default)]
struct CuvidVideoFormat {
    codec: c_int,
    frame_rate_numerator: c_uint,
    frame_rate_denominator: c_uint,
    progressive_sequence: c_uchar,
    bit_depth_luma_minus8: c_uchar,
    bit_depth_chroma_minus8: c_uchar,
    min_num_decode_surfaces: c_uchar,
    coded_width: c_uint,
    coded_height: c_uint,
    display_left: c_int,
    display_top: c_int,
    display_right: c_int,
    display_bottom: c_int,
    chroma_format: c_int,
    bitrate: c_uint,
    display_aspect_ratio_x: c_uint,
    display_aspect_ratio_y: c_uint,
    video_signal_description_present_flag: c_uchar,
    video_full_range_flag: c_uchar,
    color_primaries: c_uchar,
    transfer_characteristics: c_uchar,
    matrix_coefficients: c_uchar,
    reserved1: [c_uchar; 3],
    seqhdr_data_length: c_uint,
}

// Source data packet for parser
#[repr(C)]
#[derive(Debug, Clone)]
struct CuvidSourceDataPacket {
    flags: c_ulong,
    payload_size: c_ulong,
    payload: *const c_uchar,
    timestamp: c_ulonglong,
}

// Picture parameters for decode (simplified - actual struct is codec-specific)
#[repr(C)]
#[derive(Debug, Clone)]
struct CuvidPicParams {
    pic_width_in_mbs: c_int,
    pic_height_in_mbs: c_int,
    curr_pic_idx: c_int,
    field_pic_flag: c_int,
    bottom_field_flag: c_int,
    second_field: c_int,
    num_bitstream_packets: c_uint,
    bitstream_data_len: c_uint,
    bitstream_data: *const c_uchar,
    num_slices: c_uint,
    slice_data_offsets: *const c_uint,
    ref_pic_idx: [c_int; 16],
    // Codec-specific parameters follow (union in C)
    codec_specific: [c_uchar; 1024], // Reserved space for codec params
}

// Parser callback function signatures
type PfnSequenceCallback = unsafe extern "C" fn(user_data: *mut c_void, format: *mut CuvidVideoFormat) -> c_int;
type PfnDecodePicture = unsafe extern "C" fn(user_data: *mut c_void, pic_params: *mut CuvidPicParams) -> c_int;
type PfnDisplayPicture = unsafe extern "C" fn(user_data: *mut c_void, disp_info: *mut CuvidParsedDispInfo) -> c_int;
type PfnGetOperatingPoint = unsafe extern "C" fn(user_data: *mut c_void, op_info: *mut CuvidOperatingPointInfo) -> c_int;
type PfnGetSeiMsg = unsafe extern "C" fn(user_data: *mut c_void, sei_info: *mut CuvidSeiMessageInfo) -> c_int;

// Display info from parser
#[repr(C)]
#[derive(Debug, Clone, Default)]
struct CuvidParsedDispInfo {
    picture_index: c_int,
    progressive_frame: c_int,
    top_field_first: c_int,
    repeat_first_field: c_int,
    timestamp: c_ulonglong,
}

// Operating point info for AV1
#[repr(C)]
#[derive(Debug, Clone, Default)]
struct CuvidOperatingPointInfo {
    codec: c_int,
    av1_operating_points_cnt_minus1: c_uchar,
    av1_seq_level_idx: [c_uchar; 32],
    av1_op_idx: c_uchar,
}

// SEI message info
#[repr(C)]
#[derive(Debug, Clone)]
struct CuvidSeiMessageInfo {
    sei_data: *mut c_void,
    sei_message: *mut c_void,
    sei_message_count: c_uint,
    pic_idx: c_uint,
}

// Parser parameters
#[repr(C)]
struct CuvidParserParams {
    codec_type: c_int,
    max_num_decode_surfaces: c_uint,
    clock_rate: c_uint,
    error_threshold: c_uint,
    max_display_delay: c_uint,
    annex_b: c_uint,
    user_data: *mut c_void,
    pfn_sequence_callback: Option<PfnSequenceCallback>,
    pfn_decode_picture: Option<PfnDecodePicture>,
    pfn_display_picture: Option<PfnDisplayPicture>,
    pfn_get_operating_point: Option<PfnGetOperatingPoint>,
    pfn_get_sei_msg: Option<PfnGetSeiMsg>,
    reserved: [*mut c_void; 5],
}

// Decode status
#[repr(C)]
#[derive(Debug, Clone, Default)]
struct CuvidDecodeStatus {
    decode_status: c_uint,
    reserved: [c_uint; 31],
    p_reserved: [*mut c_void; 8],
}

// Video processor parameters for mapping
#[repr(C)]
#[derive(Debug, Clone)]
struct CuvidProcParams {
    progressive_frame: c_int,
    second_field: c_int,
    top_field_first: c_int,
    unpaired_field: c_int,
    reserved_flags: c_uint,
    reserved_zero: c_uint,
    input_stream: CUstream,
    output_stream: CUstream,
    reserved: [*mut c_void; 46],
}

// Packet flags
const CUVID_PKT_ENDOFSTREAM: c_ulong = 0x01;
const CUVID_PKT_TIMESTAMP: c_ulong = 0x02;
#[allow(dead_code)]
const CUVID_PKT_DISCONTINUITY: c_ulong = 0x04;

// ============================================================================
// Dynamic Library Loading
// ============================================================================

/// Loaded CUDA library and function pointers
#[allow(dead_code)]
struct CudaLibrary {
    _lib: Library,
    cu_init: Symbol<'static, unsafe extern "C" fn(c_uint) -> CUresult>,
    cu_device_get: Symbol<'static, unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult>,
    cu_device_get_count: Symbol<'static, unsafe extern "C" fn(*mut c_int) -> CUresult>,
    cu_ctx_create: Symbol<'static, unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult>,
    cu_ctx_destroy: Symbol<'static, unsafe extern "C" fn(CUcontext) -> CUresult>,
    cu_ctx_push_current: Symbol<'static, unsafe extern "C" fn(CUcontext) -> CUresult>,
    cu_ctx_pop_current: Symbol<'static, unsafe extern "C" fn(*mut CUcontext) -> CUresult>,
    // Memory operations
    cu_memcpy_dtoh: Symbol<'static, unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult>,
    cu_memcpy_2d: Symbol<'static, unsafe extern "C" fn(*const CudaMemcpy2D) -> CUresult>,
}

// CUDA 2D memory copy parameters
#[repr(C)]
#[derive(Debug, Clone)]
struct CudaMemcpy2D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: c_uint,
    src_host: *const c_void,
    src_device: CUdeviceptr,
    src_array: *mut c_void,
    src_pitch: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: c_uint,
    dst_host: *mut c_void,
    dst_device: CUdeviceptr,
    dst_array: *mut c_void,
    dst_pitch: usize,
    width_in_bytes: usize,
    height: usize,
}

// Memory types
const CU_MEMORYTYPE_HOST: c_uint = 0x01;
const CU_MEMORYTYPE_DEVICE: c_uint = 0x02;

/// Loaded NVDEC library and function pointers
#[allow(dead_code)]
struct NvdecLibrary {
    _lib: Library,
    // Decoder management
    cuvid_get_decoder_caps: Symbol<'static, unsafe extern "C" fn(*mut CuvidDecodeCaps) -> CUresult>,
    cuvid_create_decoder: Symbol<'static, unsafe extern "C" fn(*mut CUvideodecoder, *const CuvidDecodeCreateInfo) -> CUresult>,
    cuvid_destroy_decoder: Symbol<'static, unsafe extern "C" fn(CUvideodecoder) -> CUresult>,
    // Parser management
    cuvid_create_video_parser: Symbol<'static, unsafe extern "C" fn(*mut CUvideoparser, *const CuvidParserParams) -> CUresult>,
    cuvid_parse_video_data: Symbol<'static, unsafe extern "C" fn(CUvideoparser, *const CuvidSourceDataPacket) -> CUresult>,
    cuvid_destroy_video_parser: Symbol<'static, unsafe extern "C" fn(CUvideoparser) -> CUresult>,
    // Decode and map operations
    cuvid_decode_picture: Symbol<'static, unsafe extern "C" fn(CUvideodecoder, *const CuvidPicParams) -> CUresult>,
    cuvid_get_decode_status: Symbol<'static, unsafe extern "C" fn(CUvideodecoder, c_int, *mut CuvidDecodeStatus) -> CUresult>,
    cuvid_map_video_frame: Symbol<'static, unsafe extern "C" fn(CUvideodecoder, c_int, *mut CUdeviceptr, *mut c_uint, *const CuvidProcParams) -> CUresult>,
    cuvid_unmap_video_frame: Symbol<'static, unsafe extern "C" fn(CUvideodecoder, CUdeviceptr) -> CUresult>,
}

/// Global library state using OnceLock for thread-safe initialization
static LIBRARIES: OnceLock<Result<(CudaLibrary, NvdecLibrary), String>> = OnceLock::new();

/// Load CUDA and NVDEC libraries dynamically.
fn load_libraries() -> Result<(), VideoError> {
    let result = LIBRARIES.get_or_init(|| load_libraries_internal());

    match result {
        Ok(_) => Ok(()),
        Err(e) => Err(VideoError::Gpu {
            message: e.clone(),
        }),
    }
}

fn load_libraries_internal() -> Result<(CudaLibrary, NvdecLibrary), String> {
    // Platform-specific library names
    #[cfg(windows)]
    let cuda_names = ["nvcuda.dll"];
    #[cfg(not(windows))]
    let cuda_names = ["libcuda.so.1", "libcuda.so"];

    #[cfg(windows)]
    let nvdec_names = ["nvcuvid.dll"];
    #[cfg(not(windows))]
    let nvdec_names = ["libnvcuvid.so.1", "libnvcuvid.so"];

    // Load CUDA library
    let cuda_lib = cuda_names
        .iter()
        .find_map(|name| unsafe { Library::new(name).ok() })
        .ok_or_else(|| "CUDA library not found. Is NVIDIA driver installed?".to_string())?;

    // Load NVDEC library
    let nvdec_lib = nvdec_names
        .iter()
        .find_map(|name| unsafe { Library::new(name).ok() })
        .ok_or_else(|| "NVDEC library not found. Is NVIDIA driver installed?".to_string())?;

    // Get CUDA function pointers (using unsafe to create 'static lifetime)
    let cuda_lib = Box::leak(Box::new(cuda_lib));
    let nvdec_lib = Box::leak(Box::new(nvdec_lib));

    unsafe {
        let cuda = CudaLibrary {
            _lib: ptr::read(cuda_lib),
            cu_init: cuda_lib.get(b"cuInit\0").map_err(|e| format!("cuInit: {}", e))?,
            cu_device_get: cuda_lib.get(b"cuDeviceGet\0").map_err(|e| format!("cuDeviceGet: {}", e))?,
            cu_device_get_count: cuda_lib.get(b"cuDeviceGetCount\0").map_err(|e| format!("cuDeviceGetCount: {}", e))?,
            cu_ctx_create: cuda_lib.get(b"cuCtxCreate_v2\0").map_err(|e| format!("cuCtxCreate: {}", e))?,
            cu_ctx_destroy: cuda_lib.get(b"cuCtxDestroy_v2\0").map_err(|e| format!("cuCtxDestroy: {}", e))?,
            cu_ctx_push_current: cuda_lib.get(b"cuCtxPushCurrent_v2\0").map_err(|e| format!("cuCtxPushCurrent: {}", e))?,
            cu_ctx_pop_current: cuda_lib.get(b"cuCtxPopCurrent_v2\0").map_err(|e| format!("cuCtxPopCurrent: {}", e))?,
            cu_memcpy_dtoh: cuda_lib.get(b"cuMemcpyDtoH_v2\0").map_err(|e| format!("cuMemcpyDtoH: {}", e))?,
            cu_memcpy_2d: cuda_lib.get(b"cuMemcpy2D_v2\0").map_err(|e| format!("cuMemcpy2D: {}", e))?,
        };

        let nvdec = NvdecLibrary {
            _lib: ptr::read(nvdec_lib),
            // Decoder management
            cuvid_get_decoder_caps: nvdec_lib.get(b"cuvidGetDecoderCaps\0").map_err(|e| format!("cuvidGetDecoderCaps: {}", e))?,
            cuvid_create_decoder: nvdec_lib.get(b"cuvidCreateDecoder\0").map_err(|e| format!("cuvidCreateDecoder: {}", e))?,
            cuvid_destroy_decoder: nvdec_lib.get(b"cuvidDestroyDecoder\0").map_err(|e| format!("cuvidDestroyDecoder: {}", e))?,
            // Parser management
            cuvid_create_video_parser: nvdec_lib.get(b"cuvidCreateVideoParser\0").map_err(|e| format!("cuvidCreateVideoParser: {}", e))?,
            cuvid_parse_video_data: nvdec_lib.get(b"cuvidParseVideoData\0").map_err(|e| format!("cuvidParseVideoData: {}", e))?,
            cuvid_destroy_video_parser: nvdec_lib.get(b"cuvidDestroyVideoParser\0").map_err(|e| format!("cuvidDestroyVideoParser: {}", e))?,
            // Decode and map operations
            cuvid_decode_picture: nvdec_lib.get(b"cuvidDecodePicture\0").map_err(|e| format!("cuvidDecodePicture: {}", e))?,
            cuvid_get_decode_status: nvdec_lib.get(b"cuvidGetDecodeStatus\0").map_err(|e| format!("cuvidGetDecodeStatus: {}", e))?,
            cuvid_map_video_frame: nvdec_lib.get(b"cuvidMapVideoFrame64\0").map_err(|e| format!("cuvidMapVideoFrame64: {}", e))?,
            cuvid_unmap_video_frame: nvdec_lib.get(b"cuvidUnmapVideoFrame64\0").map_err(|e| format!("cuvidUnmapVideoFrame64: {}", e))?,
        };

        Ok((cuda, nvdec))
    }
}

fn cuda() -> &'static CudaLibrary {
    let libs = LIBRARIES.get().expect("Libraries not initialized");
    &libs.as_ref().expect("CUDA not loaded").0
}

fn nvdec() -> &'static NvdecLibrary {
    let libs = LIBRARIES.get().expect("Libraries not initialized");
    &libs.as_ref().expect("NVDEC not loaded").1
}

// ============================================================================
// Parser Callback State (shared between callbacks and decoder)
// ============================================================================

/// State shared between parser callbacks and the decoder.
/// Uses Arc<Mutex> for thread-safe access from callback context.
struct ParserCallbackState {
    /// Decoder handle (set after sequence callback determines resolution)
    decoder: CUvideodecoder,
    /// Video format from sequence callback
    video_format: Option<CuvidVideoFormat>,
    /// Pending decoded frames ready for output
    decoded_frames: Vec<DecodedFrame>,
    /// Current decode index
    decode_index: u64,
    /// Output format requested
    output_format: OutputFormat,
    /// Decoder config
    config: DecoderConfig,
    /// Error message if any callback fails
    error: Option<String>,
}

impl Default for ParserCallbackState {
    fn default() -> Self {
        Self {
            decoder: ptr::null_mut(),
            video_format: None,
            decoded_frames: Vec::new(),
            decode_index: 0,
            output_format: OutputFormat::Nv12,
            config: DecoderConfig::new(DecodeCodec::Av1),
            error: None,
        }
    }
}

// Static callback functions for NVDEC parser
// These are called from native code and must use extern "C" ABI

/// Sequence callback - called when video format is detected/changed
unsafe extern "C" fn sequence_callback(user_data: *mut c_void, format: *mut CuvidVideoFormat) -> c_int {
    // SAFETY: user_data points to valid ParserCallbackState, format is valid per NVDEC contract
    let state = unsafe { &mut *(user_data as *mut ParserCallbackState) };
    let fmt = unsafe { &*format };

    // Store video format
    state.video_format = Some(fmt.clone());

    let width = fmt.coded_width;
    let height = fmt.coded_height;

    // Create or recreate decoder if resolution changed
    if !state.decoder.is_null() {
        let nvdec_lib = nvdec();
        // SAFETY: decoder handle is valid
        unsafe { (nvdec_lib.cuvid_destroy_decoder)(state.decoder) };
        state.decoder = ptr::null_mut();
    }

    let nvdec_lib = nvdec();

    let create_info = CuvidDecodeCreateInfo {
        width: width as c_ulong,
        height: height as c_ulong,
        num_decode_surfaces: state.config.num_surfaces as c_ulong,
        codec_type: fmt.codec,
        chroma_format: fmt.chroma_format,
        creation_flags: 4, // cudaVideoCreate_PreferCUVID
        bit_depth_minus8: fmt.bit_depth_luma_minus8 as c_ulong,
        intra_decode_only: 0,
        max_width: state.config.max_width as c_ulong,
        max_height: state.config.max_height as c_ulong,
        display_left: fmt.display_left,
        display_top: fmt.display_top,
        display_right: fmt.display_right,
        display_bottom: fmt.display_bottom,
        output_format: CudaVideoSurfaceFormat::from(state.output_format) as c_int,
        deinterlace_mode: CudaVideoDeinterlaceMode::Adaptive as c_int,
        target_width: width as c_ulong,
        target_height: height as c_ulong,
        num_output_surfaces: 2,
        video_lock: ptr::null_mut(),
        target_left: 0,
        target_top: 0,
        target_right: width as c_int,
        target_bottom: height as c_int,
        enable_histogram: 0,
    };

    // SAFETY: create_info is properly initialized, decoder pointer is valid
    let result = unsafe { (nvdec_lib.cuvid_create_decoder)(&mut state.decoder, &create_info) };
    if result != CUDA_SUCCESS {
        state.error = Some(format!("cuvidCreateDecoder failed in callback: {}", result));
        return 0; // Return 0 to indicate failure
    }

    // Return number of decode surfaces
    fmt.min_num_decode_surfaces as c_int
}

/// Decode picture callback - called when a picture should be decoded
unsafe extern "C" fn decode_picture_callback(user_data: *mut c_void, pic_params: *mut CuvidPicParams) -> c_int {
    // SAFETY: user_data points to valid ParserCallbackState per NVDEC contract
    let state = unsafe { &mut *(user_data as *mut ParserCallbackState) };

    if state.decoder.is_null() {
        state.error = Some("Decoder not initialized in decode callback".to_string());
        return 0;
    }

    let nvdec_lib = nvdec();
    // SAFETY: decoder and pic_params are valid per NVDEC contract
    let result = unsafe { (nvdec_lib.cuvid_decode_picture)(state.decoder, pic_params) };
    if result != CUDA_SUCCESS {
        state.error = Some(format!("cuvidDecodePicture failed: {}", result));
        return 0;
    }

    1 // Success
}

/// Display picture callback - called when a decoded picture is ready for display
unsafe extern "C" fn display_picture_callback(user_data: *mut c_void, disp_info: *mut CuvidParsedDispInfo) -> c_int {
    // SAFETY: user_data points to valid ParserCallbackState per NVDEC contract
    let state = unsafe { &mut *(user_data as *mut ParserCallbackState) };

    if disp_info.is_null() {
        // End of stream
        return 1;
    }

    // SAFETY: disp_info is not null (checked above) and valid per NVDEC contract
    let info = unsafe { &*disp_info };

    if state.decoder.is_null() {
        state.error = Some("Decoder not initialized in display callback".to_string());
        return 0;
    }

    let video_format = match &state.video_format {
        Some(fmt) => fmt,
        None => {
            state.error = Some("Video format not set in display callback".to_string());
            return 0;
        }
    };

    let width = video_format.coded_width;
    let height = video_format.coded_height;

    // Map the decoded frame from GPU
    let nvdec_lib = nvdec();
    let cuda_lib = cuda();

    let mut device_ptr: CUdeviceptr = 0;
    let mut pitch: c_uint = 0;

    let proc_params = CuvidProcParams {
        progressive_frame: info.progressive_frame,
        second_field: 0,
        top_field_first: info.top_field_first,
        unpaired_field: 0,
        reserved_flags: 0,
        reserved_zero: 0,
        input_stream: ptr::null_mut(),
        output_stream: ptr::null_mut(),
        reserved: [ptr::null_mut(); 46],
    };

    // SAFETY: decoder, info.picture_index, and proc_params are valid per NVDEC contract
    let result = unsafe {
        (nvdec_lib.cuvid_map_video_frame)(
            state.decoder,
            info.picture_index,
            &mut device_ptr,
            &mut pitch,
            &proc_params,
        )
    };

    if result != CUDA_SUCCESS {
        state.error = Some(format!("cuvidMapVideoFrame failed: {}", result));
        return 0;
    }

    // Copy NV12 frame from GPU to CPU
    // NV12 format: Y plane (width * height) + UV plane (width * height / 2)
    let y_size = (pitch as usize) * (height as usize);
    let uv_size = (pitch as usize) * (height as usize / 2);
    let total_size = y_size + uv_size;

    let mut cpu_data = vec![0u8; total_size];

    // Copy Y plane
    let copy_y = CudaMemcpy2D {
        src_x_in_bytes: 0,
        src_y: 0,
        src_memory_type: CU_MEMORYTYPE_DEVICE,
        src_host: ptr::null(),
        src_device: device_ptr,
        src_array: ptr::null_mut(),
        src_pitch: pitch as usize,
        dst_x_in_bytes: 0,
        dst_y: 0,
        dst_memory_type: CU_MEMORYTYPE_HOST,
        dst_host: cpu_data.as_mut_ptr() as *mut c_void,
        dst_device: 0,
        dst_array: ptr::null_mut(),
        dst_pitch: width as usize,
        width_in_bytes: width as usize,
        height: height as usize,
    };

    // SAFETY: copy_y is properly initialized with valid device and host pointers
    let result = unsafe { (cuda_lib.cu_memcpy_2d)(&copy_y) };
    if result != CUDA_SUCCESS {
        // SAFETY: decoder and device_ptr are valid
        unsafe { (nvdec_lib.cuvid_unmap_video_frame)(state.decoder, device_ptr) };
        state.error = Some(format!("cuMemcpy2D (Y) failed: {}", result));
        return 0;
    }

    // Copy UV plane (interleaved, height/2)
    // SAFETY: cpu_data is large enough and pointer arithmetic is valid
    let uv_dst_ptr = unsafe { cpu_data.as_mut_ptr().add(width as usize * height as usize) as *mut c_void };
    let copy_uv = CudaMemcpy2D {
        src_x_in_bytes: 0,
        src_y: 0,
        src_memory_type: CU_MEMORYTYPE_DEVICE,
        src_host: ptr::null(),
        src_device: device_ptr + (pitch as u64 * height as u64),
        src_array: ptr::null_mut(),
        src_pitch: pitch as usize,
        dst_x_in_bytes: 0,
        dst_y: 0,
        dst_memory_type: CU_MEMORYTYPE_HOST,
        dst_host: uv_dst_ptr,
        dst_device: 0,
        dst_array: ptr::null_mut(),
        dst_pitch: width as usize,
        width_in_bytes: width as usize,
        height: height as usize / 2,
    };

    // SAFETY: copy_uv is properly initialized with valid device and host pointers
    let result = unsafe { (cuda_lib.cu_memcpy_2d)(&copy_uv) };
    if result != CUDA_SUCCESS {
        // SAFETY: decoder and device_ptr are valid
        unsafe { (nvdec_lib.cuvid_unmap_video_frame)(state.decoder, device_ptr) };
        state.error = Some(format!("cuMemcpy2D (UV) failed: {}", result));
        return 0;
    }

    // Unmap the frame
    // SAFETY: decoder and device_ptr are valid
    unsafe { (nvdec_lib.cuvid_unmap_video_frame)(state.decoder, device_ptr) };

    // Remove padding from the data (pitch may be larger than width)
    let mut final_data = Vec::with_capacity((width * height * 3 / 2) as usize);

    // Copy Y plane without padding
    for row in 0..height as usize {
        let start = row * width as usize;
        let end = start + width as usize;
        final_data.extend_from_slice(&cpu_data[start..end]);
    }

    // Copy UV plane without padding
    let uv_start = width as usize * height as usize;
    for row in 0..(height as usize / 2) {
        let start = uv_start + row * width as usize;
        let end = start + width as usize;
        final_data.extend_from_slice(&cpu_data[start..end]);
    }

    // Create decoded frame
    let frame = DecodedFrame {
        width,
        height,
        pts: info.timestamp as i64,
        decode_index: state.decode_index,
        format: OutputFormat::Nv12,
        data: final_data,
        strides: vec![width as usize, width as usize],
    };

    state.decoded_frames.push(frame);
    state.decode_index += 1;

    1 // Success
}

// ============================================================================
// NvDecoder Implementation
// ============================================================================

/// NVDEC hardware video decoder.
///
/// Provides GPU-accelerated decoding for AV1, H.264, H.265, VP8, and VP9.
pub struct NvDecoder {
    config: DecoderConfig,
    ctx: CUcontext,
    decoder: CUvideodecoder,
    parser: CUvideoparser,
    callback_state: Box<ParserCallbackState>,
    decoded_frames: Arc<Mutex<VecDeque<DecodedFrame>>>,
    width: u32,
    height: u32,
    frame_count: u64,
    initialized: bool,
}

impl NvDecoder {
    /// Create a new NVDEC decoder.
    pub fn new(config: DecoderConfig) -> Result<Self, VideoError> {
        // Load libraries
        load_libraries()?;

        let cuda_lib = cuda();

        // Initialize CUDA
        let result = unsafe { (cuda_lib.cu_init)(0) };
        if result != CUDA_SUCCESS {
            return Err(VideoError::Gpu {
                message: format!("cuInit failed: {}", result),
            });
        }

        // Get device
        let mut device: CUdevice = 0;
        let result = unsafe { (cuda_lib.cu_device_get)(&mut device, config.device_index) };
        if result != CUDA_SUCCESS {
            return Err(VideoError::Gpu {
                message: format!("cuDeviceGet failed: {}", result),
            });
        }

        // Create context
        let mut ctx: CUcontext = ptr::null_mut();
        let result = unsafe { (cuda_lib.cu_ctx_create)(&mut ctx, 0, device) };
        if result != CUDA_SUCCESS {
            return Err(VideoError::Gpu {
                message: format!("cuCtxCreate failed: {}", result),
            });
        }

        // Check capabilities
        let caps = Self::query_capabilities_internal(config.codec)?;
        if !caps.supported {
            unsafe { (cuda_lib.cu_ctx_destroy)(ctx) };
            return Err(VideoError::Decoding {
                message: format!("{:?} not supported on this GPU", config.codec),
            });
        }

        // Initialize callback state
        let mut callback_state = Box::new(ParserCallbackState {
            decoder: ptr::null_mut(),
            video_format: None,
            decoded_frames: Vec::new(),
            decode_index: 0,
            output_format: config.output_format,
            config: config.clone(),
            error: None,
        });

        // Create video parser with callbacks
        let nvdec_lib = nvdec();
        let mut parser: CUvideoparser = ptr::null_mut();

        let parser_params = CuvidParserParams {
            codec_type: CudaVideoCodec::from(config.codec) as c_int,
            max_num_decode_surfaces: config.num_surfaces,
            clock_rate: 0, // Use timestamp from bitstream
            error_threshold: 0,
            max_display_delay: if config.low_latency { 0 } else { 4 },
            annex_b: 0, // Not using Annex B format for IVF
            user_data: callback_state.as_mut() as *mut ParserCallbackState as *mut c_void,
            pfn_sequence_callback: Some(sequence_callback),
            pfn_decode_picture: Some(decode_picture_callback),
            pfn_display_picture: Some(display_picture_callback),
            pfn_get_operating_point: None, // Optional for AV1
            pfn_get_sei_msg: None, // Optional
            reserved: [ptr::null_mut(); 5],
        };

        let result = unsafe { (nvdec_lib.cuvid_create_video_parser)(&mut parser, &parser_params) };
        if result != CUDA_SUCCESS {
            unsafe { (cuda_lib.cu_ctx_destroy)(ctx) };
            return Err(VideoError::Decoding {
                message: format!("cuvidCreateVideoParser failed: {}", result),
            });
        }

        Ok(Self {
            config,
            ctx,
            decoder: ptr::null_mut(),
            parser,
            callback_state,
            decoded_frames: Arc::new(Mutex::new(VecDeque::new())),
            width: 0,
            height: 0,
            frame_count: 0,
            initialized: false,
        })
    }

    /// Check if NVDEC is available on this system.
    pub fn is_available() -> bool {
        if load_libraries().is_err() {
            return false;
        }

        let cuda = cuda();
        let result = unsafe { (cuda.cu_init)(0) };
        if result != CUDA_SUCCESS {
            return false;
        }

        let mut count: c_int = 0;
        let result = unsafe { (cuda.cu_device_get_count)(&mut count) };
        result == CUDA_SUCCESS && count > 0
    }

    /// Get NVDEC capabilities for a specific codec.
    pub fn query_capabilities(codec: DecodeCodec, device_index: i32) -> Result<DecoderCapabilities, VideoError> {
        load_libraries()?;

        let cuda = cuda();

        // Initialize CUDA
        let result = unsafe { (cuda.cu_init)(0) };
        if result != CUDA_SUCCESS {
            return Err(VideoError::Gpu {
                message: format!("cuInit failed: {}", result),
            });
        }

        // Get device
        let mut device: CUdevice = 0;
        let result = unsafe { (cuda.cu_device_get)(&mut device, device_index) };
        if result != CUDA_SUCCESS {
            return Err(VideoError::Gpu {
                message: format!("Device {} not available", device_index),
            });
        }

        // Create temporary context
        let mut ctx: CUcontext = ptr::null_mut();
        let result = unsafe { (cuda.cu_ctx_create)(&mut ctx, 0, device) };
        if result != CUDA_SUCCESS {
            return Err(VideoError::Gpu {
                message: format!("cuCtxCreate failed: {}", result),
            });
        }

        let caps = Self::query_capabilities_internal(codec);

        // Destroy context
        unsafe { (cuda.cu_ctx_destroy)(ctx) };

        caps
    }

    fn query_capabilities_internal(codec: DecodeCodec) -> Result<DecoderCapabilities, VideoError> {
        let nvdec = nvdec();

        let mut caps = CuvidDecodeCaps {
            codec_type: CudaVideoCodec::from(codec) as c_int,
            chroma_format: CudaVideoChromaFormat::Yuv420 as c_int,
            bit_depth_minus8: 0,
            ..Default::default()
        };

        let result = unsafe { (nvdec.cuvid_get_decoder_caps)(&mut caps) };
        if result != CUDA_SUCCESS {
            return Err(VideoError::Decoding {
                message: format!("cuvidGetDecoderCaps failed: {}", result),
            });
        }

        Ok(DecoderCapabilities {
            supported: caps.is_supported != 0,
            max_width: caps.max_width,
            max_height: caps.max_height,
            max_bit_depth: caps.bit_depth_minus8 + 8,
            num_engines: caps.num_nv_decs,
        })
    }

    /// Initialize the decoder for a specific resolution.
    #[allow(dead_code)]
    fn init_decoder(&mut self, width: u32, height: u32) -> Result<(), VideoError> {
        if self.initialized && self.width == width && self.height == height {
            return Ok(());
        }

        // Destroy existing decoder if any
        if !self.decoder.is_null() {
            let nvdec = nvdec();
            unsafe { (nvdec.cuvid_destroy_decoder)(self.decoder) };
            self.decoder = ptr::null_mut();
        }

        let nvdec = nvdec();

        let create_info = CuvidDecodeCreateInfo {
            width: width as c_ulong,
            height: height as c_ulong,
            num_decode_surfaces: self.config.num_surfaces as c_ulong,
            codec_type: CudaVideoCodec::from(self.config.codec) as c_int,
            chroma_format: CudaVideoChromaFormat::Yuv420 as c_int,
            creation_flags: 4, // cudaVideoCreate_PreferCUVID
            bit_depth_minus8: 0,
            intra_decode_only: 0,
            max_width: self.config.max_width as c_ulong,
            max_height: self.config.max_height as c_ulong,
            display_left: 0,
            display_top: 0,
            display_right: width as c_int,
            display_bottom: height as c_int,
            output_format: CudaVideoSurfaceFormat::from(self.config.output_format) as c_int,
            deinterlace_mode: CudaVideoDeinterlaceMode::Adaptive as c_int,
            target_width: width as c_ulong,
            target_height: height as c_ulong,
            num_output_surfaces: 2,
            video_lock: ptr::null_mut(),
            target_left: 0,
            target_top: 0,
            target_right: width as c_int,
            target_bottom: height as c_int,
            enable_histogram: 0,
        };

        let result = unsafe { (nvdec.cuvid_create_decoder)(&mut self.decoder, &create_info) };
        if result != CUDA_SUCCESS {
            return Err(VideoError::Decoding {
                message: format!("cuvidCreateDecoder failed: {}", result),
            });
        }

        self.width = width;
        self.height = height;
        self.initialized = true;

        Ok(())
    }

    /// Decode an IVF file containing AV1/VP9/VP8 video.
    pub fn decode_ivf_file<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<DecodedFrame>, VideoError> {
        use std::fs::File;
        use std::io::{BufReader, Read};

        let file = File::open(path.as_ref()).map_err(|e| VideoError::Io {
            message: format!("Failed to open file: {}", e),
        })?;
        let mut reader = BufReader::new(file);

        // Read IVF header (32 bytes)
        let mut header = [0u8; 32];
        reader.read_exact(&mut header).map_err(|e| VideoError::Io {
            message: format!("Failed to read IVF header: {}", e),
        })?;

        // Validate signature
        if &header[0..4] != b"DKIF" {
            return Err(VideoError::Decoding {
                message: "Invalid IVF file: bad signature".to_string(),
            });
        }

        // Parse header
        let width = u16::from_le_bytes([header[12], header[13]]) as u32;
        let height = u16::from_le_bytes([header[14], header[15]]) as u32;
        let frame_count = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);

        // Detect codec from FourCC
        let fourcc = std::str::from_utf8(&header[8..12]).unwrap_or("????");
        let detected_codec = DecodeCodec::from_fourcc(fourcc).ok_or_else(|| VideoError::Decoding {
            message: format!("Unknown codec: {}", fourcc),
        })?;

        if detected_codec != self.config.codec {
            return Err(VideoError::Decoding {
                message: format!(
                    "Codec mismatch: configured {:?}, file has {:?}",
                    self.config.codec, detected_codec
                ),
            });
        }

        // Store width/height for reference
        self.width = width;
        self.height = height;

        // Clear any previous decoded frames from callback state
        self.callback_state.decoded_frames.clear();
        self.callback_state.error = None;

        let nvdec_lib = nvdec();

        // Read and decode frames using the parser
        for _frame_idx in 0..frame_count {
            // Read frame header (12 bytes)
            let mut frame_header = [0u8; 12];
            if reader.read_exact(&mut frame_header).is_err() {
                break; // End of file
            }

            let frame_size = u32::from_le_bytes([
                frame_header[0],
                frame_header[1],
                frame_header[2],
                frame_header[3],
            ]) as usize;

            let frame_pts = u64::from_le_bytes([
                frame_header[4],
                frame_header[5],
                frame_header[6],
                frame_header[7],
                frame_header[8],
                frame_header[9],
                frame_header[10],
                frame_header[11],
            ]);

            // Read frame data
            let mut frame_data = vec![0u8; frame_size];
            reader.read_exact(&mut frame_data).map_err(|e| VideoError::Io {
                message: format!("Failed to read frame data: {}", e),
            })?;

            // Create packet and feed to parser
            let packet = CuvidSourceDataPacket {
                flags: CUVID_PKT_TIMESTAMP,
                payload_size: frame_data.len() as c_ulong,
                payload: frame_data.as_ptr(),
                timestamp: frame_pts,
            };

            let result = unsafe { (nvdec_lib.cuvid_parse_video_data)(self.parser, &packet) };
            if result != CUDA_SUCCESS {
                return Err(VideoError::Decoding {
                    message: format!("cuvidParseVideoData failed: {}", result),
                });
            }

            // Check for errors from callbacks
            if let Some(ref error) = self.callback_state.error {
                return Err(VideoError::Decoding {
                    message: error.clone(),
                });
            }
        }

        // Send end-of-stream packet to flush remaining frames
        let eos_packet = CuvidSourceDataPacket {
            flags: CUVID_PKT_ENDOFSTREAM,
            payload_size: 0,
            payload: ptr::null(),
            timestamp: 0,
        };

        let result = unsafe { (nvdec_lib.cuvid_parse_video_data)(self.parser, &eos_packet) };
        if result != CUDA_SUCCESS {
            return Err(VideoError::Decoding {
                message: format!("cuvidParseVideoData (EOS) failed: {}", result),
            });
        }

        // Check for errors from callbacks
        if let Some(ref error) = self.callback_state.error {
            return Err(VideoError::Decoding {
                message: error.clone(),
            });
        }

        // Move decoded frames from callback state
        let frames = std::mem::take(&mut self.callback_state.decoded_frames);
        self.frame_count += frames.len() as u64;

        // Update decoder handle from callback state
        self.decoder = self.callback_state.decoder;
        self.initialized = !self.decoder.is_null();

        Ok(frames)
    }

    /// Move frames produced by parser callbacks into the public dequeue.
    fn drain_callback_frames_to_queue(&mut self) -> Result<usize, VideoError> {
        let frames = std::mem::take(&mut self.callback_state.decoded_frames);
        if frames.is_empty() {
            return Ok(0);
        }

        let count = frames.len();
        let mut queue = self.decoded_frames.lock().map_err(|_| VideoError::Decoding {
            message: "Failed to lock frame queue".to_string(),
        })?;
        queue.extend(frames);
        Ok(count)
    }
}

impl VideoDecoder for NvDecoder {
    fn decode_file(&mut self, path: &str) -> Result<(), VideoError> {
        let ext = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "ivf" => {
                let frames = self.decode_ivf_file(path)?;
                let mut queue = self.decoded_frames.lock().map_err(|_| VideoError::Decoding {
                    message: "Failed to lock frame queue".to_string(),
                })?;
                queue.extend(frames);
                Ok(())
            }
            _ => Err(VideoError::Decoding {
                message: format!("Unsupported container: {}", ext),
            }),
        }
    }

    fn decode_packet(&mut self, data: &[u8], pts: i64) -> Result<(), VideoError> {
        if data.is_empty() {
            return Ok(());
        }

        let nvdec_lib = nvdec();
        let packet = CuvidSourceDataPacket {
            flags: CUVID_PKT_TIMESTAMP,
            payload_size: data.len() as c_ulong,
            payload: data.as_ptr(),
            timestamp: pts.max(0) as c_ulonglong,
        };

        let result = unsafe { (nvdec_lib.cuvid_parse_video_data)(self.parser, &packet) };
        if result != CUDA_SUCCESS {
            return Err(VideoError::Decoding {
                message: format!("cuvidParseVideoData failed: {}", result),
            });
        }

        if let Some(error) = self.callback_state.error.take() {
            return Err(VideoError::Decoding { message: error });
        }

        self.decoder = self.callback_state.decoder;
        self.initialized = !self.decoder.is_null();

        let produced = self.drain_callback_frames_to_queue()?;
        self.frame_count += produced as u64;
        Ok(())
    }

    fn flush(&mut self) -> Result<(), VideoError> {
        let nvdec_lib = nvdec();
        let eos_packet = CuvidSourceDataPacket {
            flags: CUVID_PKT_ENDOFSTREAM,
            payload_size: 0,
            payload: ptr::null(),
            timestamp: 0,
        };

        let result = unsafe { (nvdec_lib.cuvid_parse_video_data)(self.parser, &eos_packet) };
        if result != CUDA_SUCCESS {
            return Err(VideoError::Decoding {
                message: format!("cuvidParseVideoData (EOS) failed: {}", result),
            });
        }

        if let Some(error) = self.callback_state.error.take() {
            return Err(VideoError::Decoding { message: error });
        }

        self.decoder = self.callback_state.decoder;
        self.initialized = !self.decoder.is_null();

        let produced = self.drain_callback_frames_to_queue()?;
        self.frame_count += produced as u64;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<DecodedFrame>, VideoError> {
        let mut frames = self.decoded_frames.lock().map_err(|_| VideoError::Decoding {
            message: "Failed to lock frame queue".to_string(),
        })?;
        Ok(frames.pop_front())
    }

    fn get_capabilities(&self, codec: DecodeCodec) -> Result<DecoderCapabilities, VideoError> {
        Self::query_capabilities(codec, self.config.device_index)
    }
}

impl Drop for NvDecoder {
    fn drop(&mut self) {
        if let Ok(()) = load_libraries() {
            let nvdec_lib = nvdec();
            let cuda_lib = cuda();

            // Destroy parser first
            if !self.parser.is_null() {
                unsafe { (nvdec_lib.cuvid_destroy_video_parser)(self.parser) };
            }

            // Destroy decoder (may be in self.decoder or callback_state)
            if !self.decoder.is_null() {
                unsafe { (nvdec_lib.cuvid_destroy_decoder)(self.decoder) };
            } else if !self.callback_state.decoder.is_null() {
                unsafe { (nvdec_lib.cuvid_destroy_decoder)(self.callback_state.decoder) };
            }

            // Destroy CUDA context
            if !self.ctx.is_null() {
                unsafe { (cuda_lib.cu_ctx_destroy)(self.ctx) };
            }
        }
    }
}
