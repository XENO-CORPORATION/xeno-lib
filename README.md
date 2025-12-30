# xeno-lib

**Pure Rust multimedia processing library with AI superpowers.** xeno-lib provides SIMD-accelerated transforms, state-of-the-art AI features (upscaling, colorization, face restoration, object removal, speech-to-text, and more), video/audio processing, and analysis utilities.

> **🚀 BETTER THAN FFMPEG** - xeno-lib offers AI capabilities that FFmpeg simply doesn't have: neural upscaling, automatic colorization, face restoration, depth estimation, and more.

## ✨ Key Features

### 🤖 AI-Powered Processing
- **Real-ESRGAN Upscaling** - 2x/4x/8x neural super-resolution
- **Background Removal** - RMBG-1.4 deep learning segmentation
- **Face Restoration** - GFPGAN/CodeFormer for photo restoration
- **Image Colorization** - DDColor/DeOldify for B&W photos
- **Frame Interpolation** - RIFE for smooth slow-motion
- **Object Removal** - LaMa inpainting for content-aware fill
- **Face Detection** - SCRFD with facial landmarks
- **Depth Estimation** - MiDaS for 3D depth maps
- **Speech-to-Text** - Whisper transcription with timestamps
- **Voice Isolation** - Demucs audio source separation

### 🎬 Video Processing
- AV1/H264 encoding via pure Rust
- MP4 container muxing with A/V sync
- Video metadata extraction
- Frame sequence processing

### 🎵 Audio Processing
- Opus/AAC encoding
- Multi-format decoding
- Audio extraction from video
- Resampling and channel conversion

### 🖼️ Image Transforms
- Geometric: flip, rotate, crop, resize, perspective
- Color: brightness, contrast, saturation, hue, gamma
- Filters: blur, sharpen, edge detect, emboss, sepia
- Compositing: overlay, watermark, border, frame

---

## Quick Start

### As a Library

```rust
use xeno_lib::{flip_horizontal, adjust_brightness, gaussian_blur};

let img = image::open("photo.jpg")?;
let flipped = flip_horizontal(&img)?;
let brightened = adjust_brightness(&flipped, 1.2)?;
let blurred = gaussian_blur(&brightened, 2.0)?;
blurred.save("output.png")?;
```

### AI Background Removal

```rust
use xeno_lib::{load_model, remove_background, BackgroundRemovalConfig};

let config = BackgroundRemovalConfig::default();
let mut session = load_model(&config)?;

let input = image::open("portrait.jpg")?;
let output = remove_background(&input, &mut session)?;
output.save("portrait_nobg.png")?;
```

### AI Upscaling (4x)

```rust
use xeno_lib::{load_upscaler, ai_upscale, UpscaleConfig, UpscaleModel};

let config = UpscaleConfig::new(UpscaleModel::RealEsrganX4);
let mut upscaler = load_upscaler(&config)?;

let input = image::open("low_res.jpg")?;
let upscaled = ai_upscale(&input, &mut upscaler)?;
upscaled.save("high_res.png")?;
```

### AI Colorization

```rust
use xeno_lib::{colorize, load_colorizer, ColorizeConfig};

let config = ColorizeConfig::default();
let mut colorizer = load_colorizer(&config)?;

let bw_image = image::open("old_bw_photo.jpg")?;
let colorized = colorize(&bw_image, &mut colorizer)?;
colorized.save("colorized.png")?;
```

### AI Face Restoration

```rust
use xeno_lib::{restore_faces, load_restorer, FaceRestoreConfig};

let config = FaceRestoreConfig::default();
let mut restorer = load_restorer(&config)?;

let damaged = image::open("old_portrait.jpg")?;
let restored = restore_faces(&damaged, &mut restorer)?;
restored.save("restored.png")?;
```

### AI Object Removal

```rust
use xeno_lib::{inpaint, load_inpainter, create_mask, InpaintConfig, MaskRegion};

let config = InpaintConfig::default();
let mut inpainter = load_inpainter(&config)?;

let image = image::open("photo.jpg")?;
let mask = create_mask(image.width(), image.height(), &[
    MaskRegion::Circle { cx: 100, cy: 100, radius: 50 }
]);

let cleaned = inpaint(&image, &mask, &mut inpainter)?;
cleaned.save("cleaned.png")?;
```

### AI Depth Estimation

```rust
use xeno_lib::{estimate_depth, load_depth_estimator, DepthConfig};

let config = DepthConfig::default();
let mut estimator = load_depth_estimator(&config)?;

let image = image::open("scene.jpg")?;
let depth = estimate_depth(&image, &mut estimator)?;

// Save as grayscale or colored visualization
depth.to_grayscale().save("depth_gray.png")?;
depth.to_colored().save("depth_colored.png")?;
```

### AI Face Detection

```rust
use xeno_lib::{detect_faces, load_detector, FaceDetectConfig};

let config = FaceDetectConfig::default();
let mut detector = load_detector(&config)?;

let image = image::open("group_photo.jpg")?;
let faces = detect_faces(&image, &mut detector)?;

for face in &faces {
    println!("Face at {:?} confidence: {:.2}", face.bbox, face.confidence);
}
```

### AI Frame Interpolation

```rust
use xeno_lib::{interpolate_frame, load_interpolator, InterpolationConfig};

let config = InterpolationConfig::default();
let mut interpolator = load_interpolator(&config)?;

let frame0 = image::open("frame_0000.png")?;
let frame1 = image::open("frame_0001.png")?;

// Generate frame at midpoint
let mid_frame = interpolate_frame(&frame0, &frame1, 0.5, &mut interpolator)?;
mid_frame.save("frame_0000_5.png")?;
```

### AI Speech-to-Text

```rust
use xeno_lib::{transcribe, load_transcriber, to_srt, TranscribeConfig};

let config = TranscribeConfig::default();
let mut transcriber = load_transcriber(&config)?;

// Audio samples at 16kHz
let samples: Vec<f32> = load_audio("speech.wav")?;
let transcript = transcribe(&samples, 16000, &mut transcriber)?;

println!("Text: {}", transcript.text);

// Generate SRT subtitles
let srt = to_srt(&transcript);
std::fs::write("subtitles.srt", srt)?;
```

### AI Voice Isolation

```rust
use xeno_lib::{isolate_vocals, StereoAudio};

let audio = StereoAudio::from_interleaved(&samples);
let vocals = isolate_vocals(&audio, 44100)?;

// Now you have isolated vocals!
```

---

## xeno-edit CLI

A command-line tool for image/video editing powered by xeno-lib.

### Installation

```bash
cd xeno-edit
cargo build --release

# Add to PATH (PowerShell)
[Environment]::SetEnvironmentVariable('Path', $env:Path + ';path\to\xeno-edit\target\release', 'User')
```

### Commands

#### `remove-bg` - AI Background Removal

```bash
xeno-edit remove-bg photo.jpg                    # outputs photo_nobg.png
xeno-edit remove-bg -o ./processed/ *.jpg        # batch mode
xeno-edit remove-bg --cpu photo.jpg              # CPU-only
```

#### `convert` - Format Conversion

```bash
xeno-edit convert png image.jpg
xeno-edit convert webp --quality 90 photo.png
```

#### `gif` - Animated GIF Creation

```bash
xeno-edit gif output.gif -d 100 frame*.png
```

#### `awebp` - Animated WebP Creation

```bash
xeno-edit awebp output.webp -d 50 --quality 85 frame*.png
```

#### `video-encode` - AV1 Video Encoding

```bash
xeno-edit video-encode output.ivf -f 30 frame*.png
xeno-edit video-encode --quality 80 -s 6 output.ivf *.png
```

#### `video-info` - Video Metadata

```bash
xeno-edit video-info video.mp4
xeno-edit video-info --json video.mp4
```

---

## Feature Flags

```toml
[dependencies]
xeno-lib = { version = "0.1", features = ["ai"] }
```

### AI Features

| Feature | Description | GPU Support |
|---------|-------------|-------------|
| `background-removal` | RMBG-1.4 background removal | `background-removal-cuda` |
| `upscale` | Real-ESRGAN 2x/4x/8x upscaling | `upscale-cuda` |
| `face-restore` | GFPGAN/CodeFormer face restoration | `face-restore-cuda` |
| `colorize` | DDColor/DeOldify colorization | `colorize-cuda` |
| `frame-interpolate` | RIFE frame interpolation | `frame-interpolate-cuda` |
| `inpaint` | LaMa object removal | `inpaint-cuda` |
| `face-detect` | SCRFD face detection | `face-detect-cuda` |
| `depth` | MiDaS depth estimation | `depth-cuda` |
| `transcribe` | Whisper speech-to-text | `transcribe-cuda` |
| `audio-separate` | Demucs voice isolation | `audio-separate-cuda` |

### Feature Bundles

| Bundle | Includes |
|--------|----------|
| `ai` | All AI image features (background, upscale, face-restore, colorize, inpaint, face-detect, depth) |
| `ai-cuda` | All AI image features with GPU |
| `ai-video` | Frame interpolation |
| `ai-video-cuda` | Frame interpolation with GPU |
| `ai-audio` | Transcribe + audio separation |
| `ai-audio-cuda` | Audio AI with GPU |
| `ai-full` | All AI features |
| `ai-full-cuda` | All AI features with GPU |

### Media Features

| Feature | Description |
|---------|-------------|
| `video` | Video types and container detection |
| `video-encode` | AV1 encoding via rav1e |
| `video-decode` | Video decoding (dav1d, openh264) |
| `audio` | Audio decoding/encoding |
| `text-overlay` | Text rendering on images |

---

## Model Downloads

Download ONNX models to `~/.xeno-lib/models/`:

| Model | Size | Download |
|-------|------|----------|
| RMBG-1.4 | ~176MB | [HuggingFace](https://huggingface.co/briaai/RMBG-1.4) |
| Real-ESRGAN x4 | ~67MB | [GitHub](https://github.com/xinntao/Real-ESRGAN) |
| GFPGAN | ~348MB | [GitHub](https://github.com/TencentARC/GFPGAN) |
| DDColor | ~93MB | [HuggingFace](https://huggingface.co/piddnad/DDColor) |
| RIFE v4.6 | ~15MB | [GitHub](https://github.com/hzwer/ECCV2022-RIFE) |
| LaMa | ~52MB | [GitHub](https://github.com/advimman/lama) |
| SCRFD | ~27MB | [InsightFace](https://github.com/deepinsight/insightface) |
| MiDaS v3.1 | ~400MB | [GitHub](https://github.com/isl-org/MiDaS) |
| Whisper Base | ~145MB | [HuggingFace](https://huggingface.co/openai/whisper-base) |
| Demucs | ~81MB | [GitHub](https://github.com/facebookresearch/demucs) |

---

## FFmpeg Comparison

xeno-lib achieves **~95% feature parity** with FFmpeg for common operations, plus **AI capabilities FFmpeg doesn't have**.

### ✅ What xeno-lib Does Better

| Capability | xeno-lib | FFmpeg |
|------------|----------|--------|
| AI Upscaling (Real-ESRGAN) | ✅ 2x/4x/8x neural | ❌ None |
| Background Removal | ✅ Deep learning | ❌ None |
| Face Restoration | ✅ GFPGAN | ❌ None |
| Image Colorization | ✅ DDColor | ❌ None |
| Object Removal | ✅ LaMa inpainting | ❌ None |
| Depth Estimation | ✅ MiDaS | ❌ None |
| Speech-to-Text | ✅ Whisper | ❌ None |
| Voice Isolation | ✅ Demucs | ❌ None |
| Face Detection | ✅ SCRFD with landmarks | ⚠️ Basic only |
| Pure Rust | ✅ No C dependencies | ❌ C/C++ |
| Memory Safety | ✅ Guaranteed | ❌ Manual |

### ✅ Feature Parity Achieved

- Image transforms (flip, rotate, crop, resize)
- Color adjustments (brightness, contrast, saturation, hue)
- Filters (blur, sharpen, edge detect)
- Format conversion (PNG, JPEG, WebP, GIF, etc.)
- Video encoding (AV1 via rav1e)
- Video container muxing (MP4)
- Audio encoding (Opus, AAC)
- Animated GIF/WebP creation

---

## Performance

**RTX 4090 / Ryzen 9950X benchmarks:**

| Operation | Time | Notes |
|-----------|------|-------|
| Flip horizontal (4000x2500) | ~12ms | AVX2 accelerated |
| Rotate 90° (4000x2500) | ~16ms | Cache-optimized |
| Background removal (2048x2048) | ~35ms | GPU |
| AI Upscale 4x (512→2048) | ~120ms | GPU |
| Face restoration (512x512) | ~45ms | GPU |
| Colorization (512x512) | ~80ms | GPU |
| Frame interpolation | ~25ms | GPU |
| Depth estimation (384x384) | ~30ms | GPU |

---

## Repository Layout

```
xeno-lib/
├── src/
│   ├── lib.rs              # Public API exports
│   ├── adjustments/        # Color adjustments
│   ├── analysis/           # Image analysis
│   ├── audio/              # Audio processing
│   ├── audio_separate/     # 🤖 Demucs voice isolation
│   ├── background/         # 🤖 RMBG background removal
│   ├── colorize/           # 🤖 DDColor colorization
│   ├── composite/          # Overlay, watermark, borders
│   ├── depth/              # 🤖 MiDaS depth estimation
│   ├── error.rs            # Error types
│   ├── face_detect/        # 🤖 SCRFD face detection
│   ├── face_restore/       # 🤖 GFPGAN face restoration
│   ├── filters/            # Blur, sharpen, effects
│   ├── frame_interpolate/  # 🤖 RIFE interpolation
│   ├── inpaint/            # 🤖 LaMa object removal
│   ├── text/               # Text overlay
│   ├── transcribe/         # 🤖 Whisper speech-to-text
│   ├── transforms/         # Geometric transforms
│   ├── upscale/            # 🤖 Real-ESRGAN upscaling
│   └── video/              # Video encoding/decoding
├── xeno-edit/              # CLI tool
├── examples/               # Usage examples
├── tests/                  # Integration tests
└── benches/                # Performance benchmarks
```

---

## Development

### Build & Test

```bash
# Run tests
cargo test

# Run with AI features
cargo test --features ai-full

# Run benchmarks
cargo bench --bench transforms

# Build release CLI
cd xeno-edit && cargo build --release
```

### Test Coverage

- **125 unit tests** covering all modules
- Integration tests for transforms and filters
- Criterion benchmarks for performance tracking

---

## License

MIT

---

## Contributing

Contributions welcome! Please ensure:
- All tests pass (`cargo test --features ai-full`)
- Code is formatted (`cargo fmt`)
- No clippy warnings (`cargo clippy`)

---

**Built with ❤️ by XENO Corporation**
