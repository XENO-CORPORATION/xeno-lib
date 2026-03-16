//! LaTeX compilation using the embedded Tectonic engine.
//!
//! This module provides LaTeX to PDF compilation without requiring
//! any external TeX distribution to be installed. Tectonic automatically
//! downloads required LaTeX packages on first use (~200MB cache, stored
//! in the user's platform-specific cache directory).
//!
//! # Examples
//!
//! ```no_run
//! use xeno_lib::latex::compile_latex;
//!
//! let tex = r#"\documentclass{article}
//! \begin{document}
//! Hello, world!
//! \end{document}"#;
//!
//! let pdf_bytes = compile_latex(tex).expect("compilation failed");
//! assert!(!pdf_bytes.is_empty());
//! ```

use std::path::Path;

/// Compile LaTeX source to PDF bytes.
///
/// Uses the embedded Tectonic engine to compile the given LaTeX source
/// code directly to PDF in memory. No external TeX distribution is required.
///
/// Note: The first invocation may be slow as Tectonic downloads required
/// LaTeX packages from the network. Subsequent compilations use a local cache.
///
/// # Arguments
/// * `tex_source` - The LaTeX source code as a string
///
/// # Returns
/// The compiled PDF as a byte vector, or an error message
///
/// # Errors
/// Returns an error string if:
/// - The LaTeX source contains syntax errors
/// - Required packages cannot be downloaded
/// - The Tectonic engine encounters an internal error
pub fn compile_latex(tex_source: &str) -> Result<Vec<u8>, String> {
    let pdf = tectonic::latex_to_pdf(tex_source)
        .map_err(|e| format!("LaTeX compilation failed: {}", e))?;
    Ok(pdf)
}

/// Compile a `.tex` file to PDF.
///
/// Reads the file at `tex_path` and compiles it to PDF using the
/// embedded Tectonic engine.
///
/// # Arguments
/// * `tex_path` - Path to a `.tex` file on disk
///
/// # Returns
/// The compiled PDF as a byte vector, or an error message
///
/// # Errors
/// Returns an error string if the file cannot be read or compilation fails.
pub fn compile_file(tex_path: &Path) -> Result<Vec<u8>, String> {
    let source = std::fs::read_to_string(tex_path)
        .map_err(|e| format!("Could not read file '{}': {}", tex_path.display(), e))?;
    compile_latex(&source)
}
