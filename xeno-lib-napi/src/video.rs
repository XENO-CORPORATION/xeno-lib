//! N-API bindings for video decode/encode operations.
//!
//! Exposes HEVC decoding and NVENC availability detection to Node.js / Electron.

use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Decode a single HEVC/H.265 frame from raw NAL unit data to RGBA pixels.
///
/// Takes Annex B format HEVC data and returns an RGBA buffer.
///
/// # Current Status
///
/// Returns an error until libde265 is linked. The NAL unit parsing and
/// YUV→RGBA conversion pipeline are fully implemented and ready.
///
/// # Example (JavaScript)
///
/// ```js
/// const { decodeHevcFrame } = require('@xeno/lib');
/// const hevcData = fs.readFileSync('frame.h265');
/// const rgba = await decodeHevcFrame(hevcData);
/// ```
#[napi]
pub async fn decode_hevc_frame(data: Buffer) -> Result<Buffer> {
    let data_slice = data.as_ref();

    if data_slice.is_empty() {
        return Err(napi::Error::from_reason("HEVC data is empty"));
    }

    // Run decoding on a blocking thread to avoid blocking the Node.js event loop
    let data_vec = data_slice.to_vec();
    let result = tokio::task::spawn_blocking(move || {
        xeno_lib::video::decode::hevc::decode_hevc_frame(&data_vec)
    })
    .await
    .map_err(|e| napi::Error::from_reason(format!("Task join error: {}", e)))?;

    match result {
        Ok(frame) => Ok(Buffer::from(frame.data)),
        Err(e) => Err(napi::Error::from_reason(format!("HEVC decode error: {}", e))),
    }
}

/// Check if NVENC hardware encoding is available on the current system.
///
/// Returns `true` if the NVIDIA NVENC library can be loaded, indicating
/// that hardware encoding is possible. Does not require creating an
/// encoder session.
///
/// # Example (JavaScript)
///
/// ```js
/// const { isNvencAvailable } = require('@xeno/lib');
/// if (isNvencAvailable()) {
///   console.log('NVENC hardware encoding is available');
/// }
/// ```
#[napi]
pub fn is_nvenc_available() -> Result<bool> {
    Ok(xeno_lib::video::encode::nvenc::NvencSession::is_available())
}
