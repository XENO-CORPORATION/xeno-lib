//! AI-powered human pose estimation.
//!
//! Detect body keypoints and skeletal structure in images.
//!
//! # Models
//!
//! - **MoveNet**: Fast and accurate pose estimation
//! - **MediaPipe Pose**: Full body pose with 33 landmarks
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::pose::{detect_pose, load_pose_model, PoseConfig, visualize_pose};
//!
//! let config = PoseConfig::default();
//! let mut model = load_pose_model(&config)?;
//!
//! let image = image::open("photo.jpg")?;
//! let poses = detect_pose(&image, &mut model)?;
//!
//! for pose in &poses {
//!     println!("Person detected with {} keypoints", pose.keypoints.len());
//! }
//!
//! let visualized = visualize_pose(&image, &poses);
//! visualized.save("pose_output.png")?;
//! ```

mod config;
mod model;
mod processor;

pub use config::*;
pub use model::*;
pub use processor::*;

/// Standard body keypoint indices (COCO format).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyKeypoint {
    Nose = 0,
    LeftEye = 1,
    RightEye = 2,
    LeftEar = 3,
    RightEar = 4,
    LeftShoulder = 5,
    RightShoulder = 6,
    LeftElbow = 7,
    RightElbow = 8,
    LeftWrist = 9,
    RightWrist = 10,
    LeftHip = 11,
    RightHip = 12,
    LeftKnee = 13,
    RightKnee = 14,
    LeftAnkle = 15,
    RightAnkle = 16,
}

impl BodyKeypoint {
    /// Get all keypoints.
    pub fn all() -> &'static [BodyKeypoint] {
        &[
            Self::Nose,
            Self::LeftEye,
            Self::RightEye,
            Self::LeftEar,
            Self::RightEar,
            Self::LeftShoulder,
            Self::RightShoulder,
            Self::LeftElbow,
            Self::RightElbow,
            Self::LeftWrist,
            Self::RightWrist,
            Self::LeftHip,
            Self::RightHip,
            Self::LeftKnee,
            Self::RightKnee,
            Self::LeftAnkle,
            Self::RightAnkle,
        ]
    }

    /// Get the name of the keypoint.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Nose => "nose",
            Self::LeftEye => "left_eye",
            Self::RightEye => "right_eye",
            Self::LeftEar => "left_ear",
            Self::RightEar => "right_ear",
            Self::LeftShoulder => "left_shoulder",
            Self::RightShoulder => "right_shoulder",
            Self::LeftElbow => "left_elbow",
            Self::RightElbow => "right_elbow",
            Self::LeftWrist => "left_wrist",
            Self::RightWrist => "right_wrist",
            Self::LeftHip => "left_hip",
            Self::RightHip => "right_hip",
            Self::LeftKnee => "left_knee",
            Self::RightKnee => "right_knee",
            Self::LeftAnkle => "left_ankle",
            Self::RightAnkle => "right_ankle",
        }
    }
}

/// Skeleton connections for visualization.
pub const SKELETON_CONNECTIONS: &[(BodyKeypoint, BodyKeypoint)] = &[
    // Face
    (BodyKeypoint::LeftEar, BodyKeypoint::LeftEye),
    (BodyKeypoint::LeftEye, BodyKeypoint::Nose),
    (BodyKeypoint::Nose, BodyKeypoint::RightEye),
    (BodyKeypoint::RightEye, BodyKeypoint::RightEar),
    // Arms
    (BodyKeypoint::LeftShoulder, BodyKeypoint::LeftElbow),
    (BodyKeypoint::LeftElbow, BodyKeypoint::LeftWrist),
    (BodyKeypoint::RightShoulder, BodyKeypoint::RightElbow),
    (BodyKeypoint::RightElbow, BodyKeypoint::RightWrist),
    // Torso
    (BodyKeypoint::LeftShoulder, BodyKeypoint::RightShoulder),
    (BodyKeypoint::LeftShoulder, BodyKeypoint::LeftHip),
    (BodyKeypoint::RightShoulder, BodyKeypoint::RightHip),
    (BodyKeypoint::LeftHip, BodyKeypoint::RightHip),
    // Legs
    (BodyKeypoint::LeftHip, BodyKeypoint::LeftKnee),
    (BodyKeypoint::LeftKnee, BodyKeypoint::LeftAnkle),
    (BodyKeypoint::RightHip, BodyKeypoint::RightKnee),
    (BodyKeypoint::RightKnee, BodyKeypoint::RightAnkle),
];
