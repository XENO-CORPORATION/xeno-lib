// Staged for migration to xeno-rt
//
// This module contains extracted ONNX inference code from xeno-lib AI modules.
// Each submodule mirrors the inference portions of its corresponding source module
// (model loading, session management, preprocessing to tensors, inference execution,
// postprocessing from tensors). Pure image/audio processing utilities remain in the
// original modules.
//
// These files are NOT wired into the build -- they exist solely as a staging area
// for the xeno-rt migration. See docs/AI_MIGRATION.md for the full plan.

#[cfg(feature = "background-removal")]
pub mod background_removal;

#[cfg(feature = "upscale")]
pub mod upscale;

#[cfg(feature = "face-restore")]
pub mod face_restore;

#[cfg(feature = "colorize")]
pub mod colorize;

#[cfg(feature = "inpaint")]
pub mod inpaint;
