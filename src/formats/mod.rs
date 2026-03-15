//! Modern image format support.
//!
//! This module provides encode/decode functions for modern image formats
//! that go beyond the standard PNG/JPEG/WebP pipeline.
//!
//! # Supported Formats
//!
//! - **AVIF** (AV1 Image Format) — modern, efficient image format based on AV1.
//!   Supported via the `image` crate's built-in AVIF support (uses `ravif` for
//!   encoding and `dav1d` for decoding). Feature flag: `format-avif`.
//!
//! - **HEIF/HEIC** (High Efficiency Image Format) — Apple's container for
//!   HEVC-encoded images. Stub implementation pending `libheif-rs` integration.
//!   Feature flag: `format-heif`.
//!
//! - **WebP** — improved encode/decode with quality and lossless mode support.
//!   Always available via the `image` crate's default features.
//!
//! # Feature Flags
//!
//! - `format-avif` — AVIF encode/decode (uses `image` crate AVIF features)
//! - `format-heif` — HEIF/HEIC support (stub, pending libheif-rs)

#[cfg(feature = "format-avif")]
pub mod avif;

#[cfg(feature = "format-heif")]
pub mod heif;

pub mod webp;

// Re-exports for convenience
#[cfg(feature = "format-avif")]
pub use avif::{decode_avif, encode_avif, decode_avif_from_file, AvifEncodeConfig};

#[cfg(feature = "format-heif")]
pub use heif::{decode_heif, encode_heif, HeifEncodeConfig, is_heif_available};

pub use webp::{encode_webp_advanced, decode_webp, WebpEncodeConfig, WebpMode};
