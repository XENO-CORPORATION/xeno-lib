//! Configuration for text/image to 3D mesh generation.

use std::path::PathBuf;

/// Available text-to-3D models.
///
/// # Model Comparison (2025-2026 Research)
///
/// - **TripoSR**: Single-image-to-3D, fast (~0.5s), produces textured meshes.
///   Best for quick preview meshes. ONNX available via export.
/// - **InstantMesh**: Higher quality multi-view reconstruction, slower.
///   Better topology for animation. ONNX export experimental.
/// - **Wonder3D**: Multi-view diffusion for consistent 3D generation.
///   Highest quality but slowest. ONNX not yet stable.
///
/// Chosen approach: TripoSR for speed, InstantMesh for quality.
/// Both support ONNX export. Wonder3D kept as stub for future integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mesh3DModel {
    /// TripoSR — fast single-image-to-3D mesh generation.
    /// Produces textured mesh in ~0.5s on GPU.
    #[default]
    TripoSR,

    /// InstantMesh — high-quality multi-view 3D reconstruction.
    /// Better topology, suitable for animation workflows.
    InstantMesh,

    /// Wonder3D — multi-view diffusion for consistent 3D.
    /// Highest quality, experimental ONNX support.
    Wonder3D,
}

impl Mesh3DModel {
    /// Returns the default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            Self::TripoSR => "triposr.onnx",
            Self::InstantMesh => "instantmesh.onnx",
            Self::Wonder3D => "wonder3d.onnx",
        }
    }

    /// Returns expected input image size (width, height).
    pub fn input_size(&self) -> (u32, u32) {
        match self {
            Self::TripoSR => (256, 256),
            Self::InstantMesh => (256, 256),
            Self::Wonder3D => (256, 256),
        }
    }

    /// Returns a human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::TripoSR => "TripoSR (Fast)",
            Self::InstantMesh => "InstantMesh (Quality)",
            Self::Wonder3D => "Wonder3D (Multi-view)",
        }
    }
}

/// Output format for 3D mesh export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshFormat {
    /// OBJ format — widely supported, includes UVs and normals.
    #[default]
    Obj,
    /// GLB/glTF binary — modern, includes textures and materials.
    Glb,
    /// STL — simple, for 3D printing.
    Stl,
    /// PLY — point cloud with vertex colors.
    Ply,
}

impl MeshFormat {
    /// Returns file extension.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Obj => "obj",
            Self::Glb => "glb",
            Self::Stl => "stl",
            Self::Ply => "ply",
        }
    }
}

/// Configuration for 3D mesh generation.
#[derive(Debug, Clone)]
pub struct Mesh3DConfig {
    /// Model to use.
    pub model: Mesh3DModel,
    /// Custom model path.
    pub model_path: Option<PathBuf>,
    /// Use GPU acceleration.
    pub use_gpu: bool,
    /// GPU device ID.
    pub gpu_device_id: i32,
    /// Output mesh format.
    pub output_format: MeshFormat,
    /// Mesh resolution (number of vertices target). Default: 50000.
    pub mesh_resolution: u32,
    /// Whether to generate texture coordinates.
    pub generate_uvs: bool,
    /// Whether to generate vertex normals.
    pub generate_normals: bool,
    /// Whether to remove background before processing.
    pub remove_background: bool,
    /// Foreground ratio for cropping (0.0-1.0). Default: 0.85.
    pub foreground_ratio: f32,
}

impl Default for Mesh3DConfig {
    fn default() -> Self {
        Self {
            model: Mesh3DModel::default(),
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            output_format: MeshFormat::default(),
            mesh_resolution: 50_000,
            generate_uvs: true,
            generate_normals: true,
            remove_background: true,
            foreground_ratio: 0.85,
        }
    }
}

impl Mesh3DConfig {
    /// Create config with specified model.
    pub fn new(model: Mesh3DModel) -> Self {
        Self {
            model,
            ..Default::default()
        }
    }

    /// Set model path.
    pub fn with_model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Enable/disable GPU.
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set output format.
    pub fn with_format(mut self, format: MeshFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Set mesh resolution.
    pub fn with_resolution(mut self, vertices: u32) -> Self {
        self.mesh_resolution = vertices.max(1000);
        self
    }

    /// Get effective model path.
    pub fn effective_model_path(&self) -> PathBuf {
        if let Some(ref path) = self.model_path {
            path.clone()
        } else {
            crate::model_utils::default_model_path(self.model.default_filename())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Mesh3DConfig::default();
        assert_eq!(config.model, Mesh3DModel::TripoSR);
        assert!(config.use_gpu);
        assert_eq!(config.mesh_resolution, 50_000);
        assert!(config.generate_uvs);
    }

    #[test]
    fn test_model_filenames() {
        assert_eq!(Mesh3DModel::TripoSR.default_filename(), "triposr.onnx");
        assert_eq!(Mesh3DModel::InstantMesh.default_filename(), "instantmesh.onnx");
        assert_eq!(Mesh3DModel::Wonder3D.default_filename(), "wonder3d.onnx");
    }

    #[test]
    fn test_mesh_format_extension() {
        assert_eq!(MeshFormat::Obj.extension(), "obj");
        assert_eq!(MeshFormat::Glb.extension(), "glb");
        assert_eq!(MeshFormat::Stl.extension(), "stl");
    }

    #[test]
    fn test_config_builder() {
        let config = Mesh3DConfig::new(Mesh3DModel::InstantMesh)
            .with_gpu(false)
            .with_format(MeshFormat::Glb)
            .with_resolution(100_000);

        assert_eq!(config.model, Mesh3DModel::InstantMesh);
        assert!(!config.use_gpu);
        assert_eq!(config.output_format, MeshFormat::Glb);
        assert_eq!(config.mesh_resolution, 100_000);
    }

    #[test]
    fn test_resolution_minimum() {
        let config = Mesh3DConfig::default().with_resolution(500);
        assert_eq!(config.mesh_resolution, 1000);
    }
}
