//! Image preprocessing and mesh extraction for 3D generation.

use image::DynamicImage;
use ndarray::Array4;

use crate::error::TransformError;
use super::config::Mesh3DConfig;
use super::model::Mesh3DSession;
use super::{GeneratedMesh, Vertex, Triangle};

/// Preprocesses an image for 3D generation.
///
/// Resizes to model input size and normalizes to [0, 1] range.
pub fn preprocess_image(
    image: &DynamicImage,
    config: &Mesh3DConfig,
) -> Result<Array4<f32>, TransformError> {
    let (target_w, target_h) = config.model.input_size();
    let resized = image.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    let width = rgb.width() as usize;
    let height = rgb.height() as usize;
    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                tensor[[0, c, y, x]] = pixel[c] as f32 / 255.0;
            }
        }
    }

    Ok(tensor)
}

/// Extracts mesh from model output tensor.
///
/// Converts the predicted triplane/volume features into vertices and triangles
/// using marching cubes or similar isosurface extraction.
pub fn extract_mesh_from_output(
    output: &Array4<f32>,
    config: &Mesh3DConfig,
) -> Result<GeneratedMesh, TransformError> {
    let shape = output.shape();
    if shape.len() != 4 {
        return Err(TransformError::InferenceFailed {
            message: format!("unexpected output shape: {:?}", shape),
        });
    }

    // The model output represents a volumetric field.
    // Extract isosurface using marching cubes.
    let grid_size = shape[2]; // Assuming cubic grid
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

    // Simplified marching cubes: scan grid for sign changes
    // and generate triangles at threshold crossings.
    let threshold = 0.0_f32;

    for z in 0..grid_size.saturating_sub(1) {
        for y in 0..grid_size.saturating_sub(1) {
            for x in 0..grid_size.saturating_sub(1) {
                // Sample 8 corners of the cube
                let v000 = safe_sample(output, 0, x, y, z);
                let v100 = safe_sample(output, 0, x + 1, y, z);
                let v010 = safe_sample(output, 0, x, y + 1, z);
                let v001 = safe_sample(output, 0, x, y, z + 1);

                // Check for sign change (simplified — full marching cubes has 256 cases)
                let inside_count = [v000, v100, v010, v001]
                    .iter()
                    .filter(|&&v| v > threshold)
                    .count();

                if inside_count > 0 && inside_count < 4 {
                    let base_idx = vertices.len() as u32;
                    let scale = 2.0 / grid_size as f32;

                    // Generate a quad at the boundary (simplified)
                    let cx = x as f32 * scale - 1.0;
                    let cy = y as f32 * scale - 1.0;
                    let cz = z as f32 * scale - 1.0;

                    vertices.push(Vertex {
                        position: [cx, cy, cz],
                        normal: [0.0, 1.0, 0.0],
                        uv: [cx * 0.5 + 0.5, cz * 0.5 + 0.5],
                    });
                    vertices.push(Vertex {
                        position: [cx + scale, cy, cz],
                        normal: [0.0, 1.0, 0.0],
                        uv: [(cx + scale) * 0.5 + 0.5, cz * 0.5 + 0.5],
                    });
                    vertices.push(Vertex {
                        position: [cx + scale, cy, cz + scale],
                        normal: [0.0, 1.0, 0.0],
                        uv: [(cx + scale) * 0.5 + 0.5, (cz + scale) * 0.5 + 0.5],
                    });
                    vertices.push(Vertex {
                        position: [cx, cy, cz + scale],
                        normal: [0.0, 1.0, 0.0],
                        uv: [cx * 0.5 + 0.5, (cz + scale) * 0.5 + 0.5],
                    });

                    triangles.push(Triangle {
                        indices: [base_idx, base_idx + 1, base_idx + 2],
                    });
                    triangles.push(Triangle {
                        indices: [base_idx, base_idx + 2, base_idx + 3],
                    });
                }

                // Limit mesh complexity
                if vertices.len() as u32 >= config.mesh_resolution {
                    break;
                }
            }
            if vertices.len() as u32 >= config.mesh_resolution {
                break;
            }
        }
        if vertices.len() as u32 >= config.mesh_resolution {
            break;
        }
    }

    // Recalculate normals if requested
    if config.generate_normals && !triangles.is_empty() {
        recalculate_normals(&mut vertices, &triangles);
    }

    Ok(GeneratedMesh {
        vertices,
        triangles,
        has_uvs: config.generate_uvs,
        has_normals: config.generate_normals,
    })
}

/// Safely sample from 4D array, returning 0.0 for out-of-bounds.
fn safe_sample(arr: &Array4<f32>, c: usize, x: usize, y: usize, z: usize) -> f32 {
    let shape = arr.shape();
    if c < shape[1] && z < shape[2] && y < shape[2] && x < shape[3] {
        // Use first channel, interpret dims as [batch, channels, depth*height, width]
        arr[[0, c, z.min(shape[2] - 1), x.min(shape[3] - 1)]]
    } else {
        0.0
    }
}

/// Recalculates vertex normals from triangle faces.
fn recalculate_normals(vertices: &mut [Vertex], triangles: &[Triangle]) {
    // Reset all normals
    for v in vertices.iter_mut() {
        v.normal = [0.0, 0.0, 0.0];
    }

    // Accumulate face normals
    for tri in triangles {
        let i0 = tri.indices[0] as usize;
        let i1 = tri.indices[1] as usize;
        let i2 = tri.indices[2] as usize;

        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }

        let p0 = vertices[i0].position;
        let p1 = vertices[i1].position;
        let p2 = vertices[i2].position;

        // Edge vectors
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        // Cross product
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];

        for &idx in &tri.indices {
            let v = &mut vertices[idx as usize];
            v.normal[0] += nx;
            v.normal[1] += ny;
            v.normal[2] += nz;
        }
    }

    // Normalize
    for v in vertices.iter_mut() {
        let len = (v.normal[0] * v.normal[0]
            + v.normal[1] * v.normal[1]
            + v.normal[2] * v.normal[2])
            .sqrt();
        if len > 1e-6 {
            v.normal[0] /= len;
            v.normal[1] /= len;
            v.normal[2] /= len;
        } else {
            v.normal = [0.0, 1.0, 0.0];
        }
    }
}

/// Generates an image-to-3D mesh directly.
pub fn generate_mesh(
    image: &DynamicImage,
    session: &mut Mesh3DSession,
) -> Result<GeneratedMesh, TransformError> {
    let config = session.config().clone();
    let input_tensor = preprocess_image(image, &config)?;
    let output_tensor = session.run(&input_tensor)?;
    extract_mesh_from_output(&output_tensor, &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_preprocess_image() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
        let config = Mesh3DConfig::default();
        let tensor = preprocess_image(&img, &config).unwrap();
        let (w, h) = config.model.input_size();
        assert_eq!(tensor.shape(), &[1, 3, h as usize, w as usize]);
    }

    #[test]
    fn test_recalculate_normals() {
        let mut vertices = vec![
            Vertex { position: [0.0, 0.0, 0.0], normal: [0.0; 3], uv: [0.0; 2] },
            Vertex { position: [1.0, 0.0, 0.0], normal: [0.0; 3], uv: [0.0; 2] },
            Vertex { position: [0.0, 0.0, 1.0], normal: [0.0; 3], uv: [0.0; 2] },
        ];
        let triangles = vec![Triangle { indices: [0, 1, 2] }];
        recalculate_normals(&mut vertices, &triangles);

        // Normal should point up (positive Y for XZ plane triangle)
        for v in &vertices {
            assert!(v.normal[1].abs() > 0.9);
        }
    }
}
