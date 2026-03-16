use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Compile LaTeX source code to PDF.
///
/// Uses the embedded Tectonic engine — no external TeX distribution required.
/// The first invocation may be slower as Tectonic downloads required packages.
///
/// @param texSource - LaTeX source code string
/// @returns The compiled PDF as a Buffer
#[napi]
pub async fn compile_latex(tex_source: String) -> Result<Buffer> {
    let pdf = tokio::task::spawn_blocking(move || {
        xeno_lib::latex::compile_latex(&tex_source)
    })
    .await
    .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    .map_err(|e| Error::from_reason(e))?;

    Ok(Buffer::from(pdf))
}
