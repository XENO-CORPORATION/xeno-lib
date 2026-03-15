//! N-API bindings for xeno-lib.
//!
//! Exposes xeno-lib's image processing, audio decode/encode, and AI model
//! inference to Node.js / Electron via napi-rs.
//!
//! All image buffers are RGBA u8 (4 bytes per pixel).
//! All audio samples are f32 PCM (-1.0 to 1.0).

mod image_processing;
mod image_encoding;
mod audio_processing;
mod ai_models;
mod helpers;
