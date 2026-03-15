//! N-API bindings for xeno-lib.
//!
//! Exposes xeno-lib's image processing, audio decode/encode, and AI model
//! inference to Node.js / Electron via napi-rs.
//!
//! All image buffers are RGBA u8 (4 bytes per pixel, row-major).
//! All audio samples are f32 PCM (-1.0 to 1.0).
//!
//! # Error Handling
//!
//! Every public function validates its inputs and returns descriptive
//! `napi::Error` messages. No function will panic at the N-API boundary.
//! Invalid inputs (zero dimensions, wrong buffer sizes, NaN values, missing
//! files) are caught early and reported as JavaScript errors.

mod image_processing;
mod image_encoding;
mod audio_processing;
mod ai_models;
mod helpers;
mod validation;
