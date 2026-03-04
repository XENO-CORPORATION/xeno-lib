//! Pose estimation processing logic.

use image::{imageops::FilterType, DynamicImage, Rgba, RgbaImage};
use ndarray::Array4;

use super::model::{DetectedPose, Keypoint, PoseSession};
use super::{BodyKeypoint, SKELETON_CONNECTIONS};
use crate::error::TransformError;

/// Detect poses in an image.
pub fn detect_pose(
    image: &DynamicImage,
    session: &mut PoseSession,
) -> Result<Vec<DetectedPose>, TransformError> {
    let (input_w, input_h) = session.input_size();

    // Resize to model input size
    let resized = image.resize_exact(input_w, input_h, FilterType::Lanczos3);

    // Convert to tensor
    let input_tensor = image_to_tensor(&resized)?;

    // Run inference
    let mut poses = session.run(&input_tensor)?;

    // Keypoints are already normalized 0-1, so they work for any image size
    // Just need to ensure they're properly clipped

    for pose in &mut poses {
        for kp in &mut pose.keypoints {
            kp.x = kp.x.clamp(0.0, 1.0);
            kp.y = kp.y.clamp(0.0, 1.0);
        }
    }

    Ok(poses)
}

/// Visualize detected poses on an image.
pub fn visualize_pose(image: &DynamicImage, poses: &[DetectedPose]) -> DynamicImage {
    let mut rgba = image.to_rgba8();
    let (width, height) = (rgba.width(), rgba.height());
    let threshold = 0.3;

    // Colors for different people
    let colors = [
        Rgba([255, 0, 0, 255]),   // Red
        Rgba([0, 255, 0, 255]),   // Green
        Rgba([0, 0, 255, 255]),   // Blue
        Rgba([255, 255, 0, 255]), // Yellow
        Rgba([255, 0, 255, 255]), // Magenta
        Rgba([0, 255, 255, 255]), // Cyan
    ];

    for (i, pose) in poses.iter().enumerate() {
        let color = colors[i % colors.len()];

        // Draw skeleton connections
        for (start, end) in SKELETON_CONNECTIONS {
            if let (Some(kp1), Some(kp2)) = (pose.get(*start), pose.get(*end)) {
                if kp1.confidence >= threshold && kp2.confidence >= threshold {
                    let (x1, y1) = kp1.to_pixel(width, height);
                    let (x2, y2) = kp2.to_pixel(width, height);
                    draw_line(&mut rgba, x1 as i32, y1 as i32, x2 as i32, y2 as i32, color);
                }
            }
        }

        // Draw keypoints
        for kp in &pose.keypoints {
            if kp.confidence >= threshold {
                let (x, y) = kp.to_pixel(width, height);
                draw_circle(&mut rgba, x, y, 4, color);
            }
        }
    }

    DynamicImage::ImageRgba8(rgba)
}

/// Visualize pose with custom styling.
pub fn visualize_pose_styled(
    image: &DynamicImage,
    poses: &[DetectedPose],
    keypoint_radius: u32,
    line_thickness: u32,
    threshold: f32,
) -> DynamicImage {
    let mut rgba = image.to_rgba8();
    let (width, height) = (rgba.width(), rgba.height());

    let colors = [
        Rgba([255, 0, 0, 255]),
        Rgba([0, 255, 0, 255]),
        Rgba([0, 0, 255, 255]),
        Rgba([255, 255, 0, 255]),
        Rgba([255, 0, 255, 255]),
        Rgba([0, 255, 255, 255]),
    ];

    for (i, pose) in poses.iter().enumerate() {
        let color = colors[i % colors.len()];

        // Draw connections with thickness
        for (start, end) in SKELETON_CONNECTIONS {
            if let (Some(kp1), Some(kp2)) = (pose.get(*start), pose.get(*end)) {
                if kp1.confidence >= threshold && kp2.confidence >= threshold {
                    let (x1, y1) = kp1.to_pixel(width, height);
                    let (x2, y2) = kp2.to_pixel(width, height);

                    // Draw thick line
                    for t in 0..line_thickness {
                        let offset = t as i32 - (line_thickness / 2) as i32;
                        draw_line(
                            &mut rgba,
                            x1 as i32 + offset,
                            y1 as i32,
                            x2 as i32 + offset,
                            y2 as i32,
                            color,
                        );
                        draw_line(
                            &mut rgba,
                            x1 as i32,
                            y1 as i32 + offset,
                            x2 as i32,
                            y2 as i32 + offset,
                            color,
                        );
                    }
                }
            }
        }

        // Draw keypoints
        for kp in &pose.keypoints {
            if kp.confidence >= threshold {
                let (x, y) = kp.to_pixel(width, height);
                draw_circle(&mut rgba, x, y, keypoint_radius, color);
            }
        }
    }

    DynamicImage::ImageRgba8(rgba)
}

/// Get pose analysis metrics.
pub fn analyze_pose(pose: &DetectedPose) -> PoseAnalysis {
    let threshold = 0.3;

    // Calculate joint angles
    let left_elbow_angle = calculate_angle(
        pose.get(BodyKeypoint::LeftShoulder),
        pose.get(BodyKeypoint::LeftElbow),
        pose.get(BodyKeypoint::LeftWrist),
        threshold,
    );

    let right_elbow_angle = calculate_angle(
        pose.get(BodyKeypoint::RightShoulder),
        pose.get(BodyKeypoint::RightElbow),
        pose.get(BodyKeypoint::RightWrist),
        threshold,
    );

    let left_knee_angle = calculate_angle(
        pose.get(BodyKeypoint::LeftHip),
        pose.get(BodyKeypoint::LeftKnee),
        pose.get(BodyKeypoint::LeftAnkle),
        threshold,
    );

    let right_knee_angle = calculate_angle(
        pose.get(BodyKeypoint::RightHip),
        pose.get(BodyKeypoint::RightKnee),
        pose.get(BodyKeypoint::RightAnkle),
        threshold,
    );

    // Calculate body orientation
    let facing = calculate_facing(pose, threshold);

    PoseAnalysis {
        left_elbow_angle,
        right_elbow_angle,
        left_knee_angle,
        right_knee_angle,
        facing,
        confidence: pose.confidence,
    }
}

/// Pose analysis result.
#[derive(Debug, Clone)]
pub struct PoseAnalysis {
    /// Left elbow angle in degrees.
    pub left_elbow_angle: Option<f32>,
    /// Right elbow angle in degrees.
    pub right_elbow_angle: Option<f32>,
    /// Left knee angle in degrees.
    pub left_knee_angle: Option<f32>,
    /// Right knee angle in degrees.
    pub right_knee_angle: Option<f32>,
    /// Body facing direction.
    pub facing: BodyFacing,
    /// Overall confidence.
    pub confidence: f32,
}

/// Body facing direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyFacing {
    Front,
    Back,
    Left,
    Right,
    Unknown,
}

/// Convert image to tensor for pose model.
fn image_to_tensor(image: &DynamicImage) -> Result<Array4<f32>, TransformError> {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    // MoveNet uses int8 input, but we'll normalize to 0-255 float
    let mut tensor = Array4::<f32>::zeros((1, height, width, 3));

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            tensor[[0, y, x, 0]] = pixel[0] as f32;
            tensor[[0, y, x, 1]] = pixel[1] as f32;
            tensor[[0, y, x, 2]] = pixel[2] as f32;
        }
    }

    Ok(tensor)
}

/// Calculate angle between three points.
fn calculate_angle(
    p1: Option<&Keypoint>,
    p2: Option<&Keypoint>,
    p3: Option<&Keypoint>,
    threshold: f32,
) -> Option<f32> {
    let p1 = p1.filter(|k| k.confidence >= threshold)?;
    let p2 = p2.filter(|k| k.confidence >= threshold)?;
    let p3 = p3.filter(|k| k.confidence >= threshold)?;

    let v1 = (p1.x - p2.x, p1.y - p2.y);
    let v2 = (p3.x - p2.x, p3.y - p2.y);

    let dot = v1.0 * v2.0 + v1.1 * v2.1;
    let mag1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
    let mag2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();

    if mag1 > 0.0 && mag2 > 0.0 {
        let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
        Some(cos_angle.acos().to_degrees())
    } else {
        None
    }
}

/// Calculate body facing direction.
fn calculate_facing(pose: &DetectedPose, threshold: f32) -> BodyFacing {
    let left_shoulder = pose.get(BodyKeypoint::LeftShoulder);
    let right_shoulder = pose.get(BodyKeypoint::RightShoulder);
    let nose = pose.get(BodyKeypoint::Nose);

    match (left_shoulder, right_shoulder, nose) {
        (Some(ls), Some(rs), Some(n))
            if ls.confidence >= threshold
                && rs.confidence >= threshold
                && n.confidence >= threshold =>
        {
            let shoulder_width = (rs.x - ls.x).abs();
            let nose_offset = n.x - (ls.x + rs.x) / 2.0;

            if shoulder_width < 0.05 {
                // Shoulders very close = side view
                if nose_offset > 0.0 {
                    BodyFacing::Right
                } else {
                    BodyFacing::Left
                }
            } else if nose_offset.abs() < shoulder_width * 0.3 {
                // Nose centered = front/back
                // Check if eyes are visible for front vs back
                let left_eye = pose.get(BodyKeypoint::LeftEye);
                let right_eye = pose.get(BodyKeypoint::RightEye);

                if left_eye.map(|e| e.confidence >= threshold).unwrap_or(false)
                    && right_eye
                        .map(|e| e.confidence >= threshold)
                        .unwrap_or(false)
                {
                    BodyFacing::Front
                } else {
                    BodyFacing::Back
                }
            } else if nose_offset > 0.0 {
                BodyFacing::Right
            } else {
                BodyFacing::Left
            }
        }
        _ => BodyFacing::Unknown,
    }
}

/// Draw a line using Bresenham's algorithm.
fn draw_line(img: &mut RgbaImage, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: Rgba<u8>) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let (w, h) = (img.width() as i32, img.height() as i32);

    loop {
        if x0 >= 0 && x0 < w && y0 >= 0 && y0 < h {
            img.put_pixel(x0 as u32, y0 as u32, color);
        }

        if x0 == x1 && y0 == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Draw a filled circle.
fn draw_circle(img: &mut RgbaImage, cx: u32, cy: u32, radius: u32, color: Rgba<u8>) {
    let (w, h) = (img.width(), img.height());
    let r = radius as i32;

    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let px = cx as i32 + dx;
                let py = cy as i32 + dy;
                if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                    img.put_pixel(px as u32, py as u32, color);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_angle() {
        // 90 degree angle
        let p1 = Keypoint {
            x: 0.0,
            y: 0.0,
            confidence: 1.0,
        };
        let p2 = Keypoint {
            x: 0.0,
            y: 1.0,
            confidence: 1.0,
        };
        let p3 = Keypoint {
            x: 1.0,
            y: 1.0,
            confidence: 1.0,
        };

        let angle = calculate_angle(Some(&p1), Some(&p2), Some(&p3), 0.5).unwrap();
        assert!((angle - 90.0).abs() < 1.0);
    }

    #[test]
    fn test_body_facing() {
        let pose = DetectedPose {
            keypoints: vec![
                Keypoint {
                    x: 0.5,
                    y: 0.3,
                    confidence: 0.9,
                }, // nose
                Keypoint {
                    x: 0.45,
                    y: 0.35,
                    confidence: 0.9,
                }, // left eye
                Keypoint {
                    x: 0.55,
                    y: 0.35,
                    confidence: 0.9,
                }, // right eye
                Keypoint {
                    x: 0.4,
                    y: 0.4,
                    confidence: 0.9,
                }, // left ear
                Keypoint {
                    x: 0.6,
                    y: 0.4,
                    confidence: 0.9,
                }, // right ear
                Keypoint {
                    x: 0.35,
                    y: 0.5,
                    confidence: 0.9,
                }, // left shoulder
                Keypoint {
                    x: 0.65,
                    y: 0.5,
                    confidence: 0.9,
                }, // right shoulder
            ],
            confidence: 0.9,
            bbox: None,
        };

        let facing = calculate_facing(&pose, 0.5);
        assert_eq!(facing, BodyFacing::Front);
    }

    #[test]
    fn test_visualize_pose() {
        let img = DynamicImage::new_rgba8(640, 480);
        let poses = vec![DetectedPose {
            keypoints: vec![Keypoint {
                x: 0.5,
                y: 0.3,
                confidence: 0.9,
            }],
            confidence: 0.9,
            bbox: None,
        }];

        let result = visualize_pose(&img, &poses);
        assert_eq!(result.width(), 640);
        assert_eq!(result.height(), 480);
    }
}
