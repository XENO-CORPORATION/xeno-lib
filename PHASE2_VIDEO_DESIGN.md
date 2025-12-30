# Phase 2: Native Video Module Architecture

## Vision

xeno-lib aims to provide native video processing capabilities without FFmpeg dependency, enabling:
- Frame-by-frame video processing with xeno-lib image functions
- Video format conversion between modern codecs
- Animated format creation (GIF, WebP, APNG)
- Future: real-time video effects and AI-powered video processing

---

## Architecture Overview

```
xeno-lib/
├── src/
│   ├── video/                    # NEW: Native video module
│   │   ├── mod.rs               # Public API exports
│   │   ├── container/           # Container parsing/muxing
│   │   │   ├── mod.rs
│   │   │   ├── mp4.rs           # MP4/MOV via symphonia
│   │   │   ├── mkv.rs           # MKV/WebM via symphonia
│   │   │   └── traits.rs        # Container abstraction
│   │   ├── codec/               # Video codecs
│   │   │   ├── mod.rs
│   │   │   ├── av1.rs           # AV1 via rav1e (encode) + rav1d (decode)
│   │   │   ├── vp9.rs           # VP9 via vpx-rs or similar
│   │   │   └── traits.rs        # Codec abstraction
│   │   ├── frame.rs             # VideoFrame type
│   │   ├── pipeline.rs          # Processing pipeline
│   │   └── sequence.rs          # Image sequence I/O
│   └── ...
```

---

## Dependencies Strategy

### Container Parsing

| Crate | Purpose | Status |
|-------|---------|--------|
| [symphonia](https://github.com/pdeljanov/Symphonia) | MP4/MKV/WebM demuxing | Pure Rust, production-ready |
| [mp4](https://crates.io/crates/mp4) | MP4 muxing | Pure Rust, mature |

**Why Symphonia?**
- Pure Rust, no C dependencies
- ~85-115% of FFmpeg demuxing performance
- Supports MP4, MKV, WebM, OGG containers
- Active development, Mozilla backing

### Video Codecs

| Codec | Encode | Decode | Notes |
|-------|--------|--------|-------|
| **AV1** | rav1e | rav1d | Pure Rust + ASM, Mozilla/Xiph |
| **VP9** | TBD | TBD | Limited pure Rust options |
| **H.264** | TBD | openh264-rs | Needs investigation |
| **H.265** | TBD | TBD | No good pure Rust option yet |

**Initial Focus: AV1**
- rav1e is the most mature pure Rust video encoder
- AV1 is the future (royalty-free, excellent compression)
- Start here, expand to H.264/VP9 later via bindings if needed

---

## Core Types

### VideoFrame

```rust
/// A single decoded video frame
pub struct VideoFrame {
    /// Pixel data as RGBA
    pub image: DynamicImage,
    /// Presentation timestamp in milliseconds
    pub pts_ms: i64,
    /// Duration in milliseconds
    pub duration_ms: i64,
    /// Frame number (0-indexed)
    pub frame_number: u64,
}
```

### VideoMetadata

```rust
pub struct VideoMetadata {
    pub width: u32,
    pub height: u32,
    pub frame_rate: f64,
    pub duration_ms: i64,
    pub frame_count: u64,
    pub codec: VideoCodec,
    pub container: ContainerFormat,
    pub has_audio: bool,
}

pub enum VideoCodec {
    AV1,
    VP9,
    H264,
    H265,
    Unknown(String),
}

pub enum ContainerFormat {
    Mp4,
    Mkv,
    WebM,
    Mov,
}
```

---

## API Design

### Reading Video Frames

```rust
use xeno_lib::video::{VideoReader, VideoFrame};

// Open video file
let reader = VideoReader::open("input.mp4")?;
println!("Video: {}x{} @ {} fps",
    reader.metadata().width,
    reader.metadata().height,
    reader.metadata().frame_rate);

// Iterate frames
for frame in reader.frames() {
    let frame: VideoFrame = frame?;

    // Process with xeno-lib functions
    let processed = xeno_lib::adjust_brightness(&frame.image, 1.2)?;

    // Do something with processed frame...
}

// Or seek to specific frame
let frame = reader.seek_frame(100)?;
```

### Writing Video

```rust
use xeno_lib::video::{VideoWriter, VideoWriterConfig, VideoCodec};

let config = VideoWriterConfig {
    width: 1920,
    height: 1080,
    frame_rate: 30.0,
    codec: VideoCodec::AV1,
    quality: 80,  // 0-100
    ..Default::default()
};

let mut writer = VideoWriter::create("output.mp4", config)?;

for frame in processed_frames {
    writer.write_frame(&frame)?;
}

writer.finalize()?;
```

### Processing Pipeline

```rust
use xeno_lib::video::{VideoReader, VideoWriter, VideoProcessor};
use xeno_lib::{adjust_brightness, gaussian_blur};

// High-level API for common workflows
let processor = VideoProcessor::new()
    .add_filter(|frame| adjust_brightness(&frame, 1.2))
    .add_filter(|frame| gaussian_blur(&frame, 1.5))
    .with_codec(VideoCodec::AV1)
    .with_quality(85);

processor.process("input.mp4", "output.mp4")?;
```

---

## Feature Flags

```toml
[features]
default = []

# Video support (container parsing only - minimal deps)
video = ["symphonia"]

# AV1 codec support
video-av1 = ["video", "rav1e", "rav1d"]

# Full video support (all available codecs)
video-full = ["video-av1"]
```

---

## Implementation Phases

### Phase 2.1: Container Parsing (Current)
- [ ] Add symphonia dependency
- [ ] Implement VideoReader for MP4/MKV demuxing
- [ ] Implement frame iteration
- [ ] Add seek functionality
- [ ] Add xeno-edit `video-info` command

### Phase 2.2: AV1 Encoding
- [ ] Add rav1e dependency
- [ ] Implement VideoWriter with AV1 encoding
- [ ] Add mp4 muxing for output
- [ ] Add xeno-edit `video-encode` command

### Phase 2.3: Video Processing Pipeline
- [ ] Implement VideoProcessor builder
- [ ] Add parallel frame processing
- [ ] Memory-efficient streaming
- [ ] Progress reporting

### Phase 2.4: CLI Commands
- [ ] `xeno-edit video-info` - metadata extraction
- [ ] `xeno-edit video-frames` - extract frames as images
- [ ] `xeno-edit video-encode` - encode image sequence to video
- [ ] `xeno-edit video-process` - apply filters to video

---

## Performance Considerations

### Memory Management
- Stream frames instead of loading entire video
- Use memory-mapped I/O where possible
- Limit concurrent decoded frames

### Parallelism
- Decode frames in parallel (where supported)
- Process frames in parallel with rayon
- Encode with rav1e's built-in threading

### Target Performance
- Demuxing: Match Symphonia's ~100% of FFmpeg speed
- Decoding: rav1d targets dav1d parity
- Encoding: rav1e is ~60% of libaom speed but much safer
- Processing: Leverage xeno-lib's SIMD acceleration

---

## Comparison with FFmpeg

| Capability | xeno-lib (Goal) | FFmpeg |
|------------|-----------------|--------|
| Container parsing | Pure Rust | C |
| AV1 encode | rav1e (Rust+ASM) | libaom/SVT-AV1 |
| AV1 decode | rav1d (Rust) | dav1d |
| H.264 | TBD | libx264 |
| Memory safety | Guaranteed | Manual |
| Bundle size | ~10-20MB | ~30-50MB |
| Cross-platform | Native Rust | Requires build |

**Trade-offs:**
- xeno-lib: Safer, simpler deployment, modern codecs first
- FFmpeg: Broader codec support, more mature

---

## CLI Integration

### Planned Commands

```bash
# Get video metadata
xeno-edit video-info input.mp4

# Extract frames
xeno-edit video-frames input.mp4 -o ./frames/ --format png

# Encode image sequence to video
xeno-edit video-encode --fps 30 --codec av1 -o output.mp4 frame*.png

# Process video (apply xeno-lib filters)
xeno-edit video-process --brightness 1.2 --blur 1.5 input.mp4 output.mp4
```

---

## References

- [Symphonia](https://github.com/pdeljanov/Symphonia) - Pure Rust media demuxer
- [rav1e](https://github.com/xiph/rav1e) - Rust AV1 encoder
- [rav1d](https://github.com/memorysafety/rav1d) - Rust AV1 decoder (dav1d port)
- [mp4](https://crates.io/crates/mp4) - Rust MP4 reader/writer
- [matroska](https://crates.io/crates/matroska) - Rust MKV parser
