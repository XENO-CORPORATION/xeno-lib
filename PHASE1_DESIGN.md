# Phase 1: Complete Geometric Transformations - Architecture Design Document

## Status: Implementation Ready
**Author:** Staff Rust Engineer + Image Processing Architect
**Date:** 2025-11-08
**Target:** 52 operations across 12 categories for image + video processing

---

## 1. Executive Summary

This document defines the architecture for implementing a complete geometric transformation library supporting both single images and video frame sequences. The implementation will provide 52 operations across 12 categories with SIMD optimization, parallel processing, and strict performance targets.

**Success Criteria:**
- ✅ All 52 operations implemented and tested
- ✅ <10ms per frame for basic operations (flip, rotate 90°, etc.)
- ✅ Process 1800-frame video in <20 seconds
- ✅ Handle 4K frames efficiently (memory < 2x input size where possible)
- ✅ Zero panics on invalid inputs
- ✅ Support RGB, RGBA, grayscale, and all standard formats (JPEG, PNG, WebP)

---

## 2. Current State Analysis

### Existing Implementation (✅ = Complete)

**Category 1: Flips**
- ✅ `flip_horizontal` - src/transforms/flip.rs:16 (SIMD accelerated)
- ✅ `flip_vertical` - src/transforms/flip.rs:21 (SIMD accelerated)
- ❌ `flip_both` - MISSING

**Category 2: Rotations**
- ✅ `rotate_90` - src/transforms/rotate.rs:14 (optimized)
- ✅ `rotate_180` - src/transforms/rotate.rs:19 (optimized)
- ✅ `rotate_270` - src/transforms/rotate.rs:24 (optimized)
- ✅ `rotate` - src/transforms/rotate.rs:30 (arbitrary angle, nearest/bilinear)
- ❌ `rotate_90_ccw` - MISSING (alias needed)
- ❌ `rotate_bounded` - MISSING (expand canvas to fit)
- ❌ `rotate_cropped` - MISSING (maintain dimensions)

**Category 3: Crops**
- ✅ `crop` - src/transforms/crop.rs:10
- ❌ `crop_center`, `crop_to_aspect`, `crop_percentage`, `autocrop`, `crop_to_content` - MISSING

**Category 4: Resize/Scale**
- ✅ `resize_exact` - src/transforms/resize.rs:12
- ✅ `resize_by_percent` - src/transforms/resize.rs:34
- ✅ `resize_to_width` - src/transforms/resize.rs:50
- ✅ `resize_to_height` - src/transforms/resize.rs:76
- ✅ `resize_to_fit` - src/transforms/resize.rs:102
- ✅ `thumbnail` - src/transforms/resize.rs:137
- ❌ `resize_fill`, `resize_cover`, `scale`, `downscale`, `upscale` - MISSING

**Categories 5-12: All MISSING** (24 operations)

### Existing Architecture Patterns

**1. Type Dispatch Pattern**
```rust
// Macro-based dispatch over DynamicImage variants
dispatch_on_dynamic_image!(image, flip_horizontal_impl)
```

**2. Parallel Processing Pattern**
```rust
// Rayon-based parallel row processing
output_data.par_chunks_mut(row_stride)
    .enumerate()
    .for_each(|(row_idx, dst_row)| { ... });
```

**3. SIMD Acceleration Pattern**
```rust
// Separate SIMD module with fallback to scalar
flip_simd::horizontal_rows(src_row, dst_row, width, channels, ...);
flip_horizontal_scalar_tail(...); // Handle remainder
```

**4. Interpolation Strategy Pattern**
```rust
// Trait-based interpolation kernels
trait InterpolationKernel {
    fn sample_into(input: &[u8], ...) -> bool;
}
```

---

## 3. Architecture Design

### 3.1 Module Organization

```
src/transforms/
├── mod.rs                 # Public API exports
├── utils.rs               # Shared utilities (allocation, dispatch)
├── flip.rs                # Category 1: Flips
├── flip_simd.rs          # SIMD optimizations for flips
├── rotate.rs              # Category 2: Rotations (ENHANCED)
├── crop.rs                # Category 3: Crops (ENHANCED)
├── resize.rs              # Category 4: Resize/Scale (ENHANCED)
├── matrix.rs             # Category 5: Transpose/Transverse (NEW)
├── affine.rs             # Category 6: Affine transforms (NEW)
├── perspective.rs        # Category 7: Perspective (NEW)
├── canvas.rs             # Category 8: Canvas operations (NEW)
├── alignment.rs          # Category 9: Alignment (NEW)
├── batch.rs              # Category 10: Batch processing (NEW)
├── sequence.rs           # Category 11: Frame sequences (NEW)
├── config.rs             # Category 12: Configuration (NEW)
├── interpolation.rs      # Interpolation kernels
└── tests.rs              # Integration tests
```

### 3.2 Design Patterns

#### Pattern 1: Configuration Context
```rust
// Global mutable config for interpolation/background settings
pub struct TransformConfig {
    default_interpolation: Interpolation,
    background_color: [u8; 4],
    preserve_alpha: bool,
}

static CONFIG: Mutex<TransformConfig> = Mutex::new(TransformConfig::default());
```

#### Pattern 2: Builder Pattern for Complex Operations
```rust
pub struct PadBuilder {
    image: DynamicImage,
    top: u32, right: u32, bottom: u32, left: u32,
    color: Option<[u8; 4]>,
}
```

#### Pattern 3: Zero-Copy Streaming for Video
```rust
pub fn stream_transform<F>(
    stdin: impl Read,
    stdout: impl Write,
    transform_fn: F,
) -> Result<(), TransformError>
where F: Fn(DynamicImage) -> Result<DynamicImage, TransformError>;
```

#### Pattern 4: Pipeline Composition
```rust
pub struct TransformPipeline {
    operations: Vec<Box<dyn Fn(DynamicImage) -> Result<DynamicImage, TransformError>>>,
}
```

### 3.3 Memory Management Strategy

**Target:** < 2x input size for most operations

**Strategies:**
1. **In-place operations** where possible (flips, rotations)
2. **Streaming row processing** for large images
3. **Parallel allocation** with rayon for multi-threaded performance
4. **Lazy evaluation** for pipeline operations (only allocate when executing)

### 3.4 Performance Optimization Strategy

**SIMD Acceleration:**
- AVX2 for x86_64 (already implemented for flips)
- NEON for ARM (optional, feature-gated)
- Fallback to scalar operations

**Parallelization:**
- Row-level parallelism via rayon (already implemented)
- Batch processing with configurable thread pool
- Per-operation grain size tuning

**Cache Optimization:**
- Tile-based processing for large images
- Minimize cache misses with sequential memory access
- Aligned memory allocations for SIMD

---

## 4. Implementation Plan

### Phase 1.1: Complete Basic Operations (Days 1-2)

**Milestone:** Categories 1-4 complete (26 operations)

1. **Category 1: Flips** - Add `flip_both`
2. **Category 2: Rotations** - Add `rotate_90_ccw`, `rotate_bounded`, `rotate_cropped`
3. **Category 3: Crops** - Add 5 new crop variants
4. **Category 4: Resize/Scale** - Add 5 new resize variants

**Deliverables:**
- `src/transforms/flip.rs` - Add flip_both
- `src/transforms/rotate.rs` - Enhance with new rotation modes
- `src/transforms/crop.rs` - Add all crop variants
- `src/transforms/resize.rs` - Add scale and resize variants
- Tests for all new operations

### Phase 1.2: Advanced Transformations (Days 3-4)

**Milestone:** Categories 5-9 complete (16 operations)

5. **Category 5: Matrix Ops** - Transpose, transverse
6. **Category 6: Affine** - Shear, affine transform, translate
7. **Category 7: Perspective** - Perspective transform, correction, homography
8. **Category 8: Canvas** - Pad, expand, trim operations
9. **Category 9: Alignment** - Center, align operations

**Deliverables:**
- `src/transforms/matrix.rs` - NEW
- `src/transforms/affine.rs` - NEW (with SIMD acceleration)
- `src/transforms/perspective.rs` - NEW
- `src/transforms/canvas.rs` - NEW
- `src/transforms/alignment.rs` - NEW
- Tests for all operations

### Phase 1.3: Video/Batch Processing (Day 5)

**Milestone:** Categories 10-12 complete (10 operations)

10. **Category 10: Batch/Video** - Batch, sequence, parallel, stream, pipeline
11. **Category 11: Frame Utils** - Load/save sequences, sequence info
12. **Category 12: Control** - Interpolation/background config

**Deliverables:**
- `src/transforms/batch.rs` - NEW
- `src/transforms/sequence.rs` - NEW
- `src/transforms/config.rs` - NEW
- Video processing examples
- Performance benchmarks

### Phase 1.4: Testing & Optimization (Day 6)

**Milestone:** All performance targets met

- Comprehensive test suite (all 52 operations)
- Golden image tests
- Memory usage tests
- Performance benchmarks
- Documentation updates

---

## 5. API Design

### 5.1 Core Operations (Functional Style)

```rust
// Existing pattern - maintain consistency
pub fn flip_horizontal(image: &DynamicImage) -> Result<DynamicImage, TransformError>;
pub fn crop(image: &DynamicImage, x: u32, y: u32, w: u32, h: u32) -> Result<DynamicImage, TransformError>;
```

### 5.2 New Operations - Consistent Naming

**Crop Operations:**
```rust
pub fn crop_center(image: &DynamicImage, width: u32, height: u32) -> Result<DynamicImage, TransformError>;
pub fn crop_to_aspect(image: &DynamicImage, aspect_ratio: f32, anchor: Anchor) -> Result<DynamicImage, TransformError>;
pub fn crop_percentage(image: &DynamicImage, top: f32, right: f32, bottom: f32, left: f32) -> Result<DynamicImage, TransformError>;
pub fn autocrop(image: &DynamicImage, tolerance: u8) -> Result<DynamicImage, TransformError>;
pub fn crop_to_content(image: &DynamicImage) -> Result<DynamicImage, TransformError>;
```

**Affine Operations:**
```rust
pub fn shear_horizontal(image: &DynamicImage, factor: f32) -> Result<DynamicImage, TransformError>;
pub fn shear_vertical(image: &DynamicImage, factor: f32) -> Result<DynamicImage, TransformError>;
pub fn affine_transform(image: &DynamicImage, matrix: [[f32; 3]; 2]) -> Result<DynamicImage, TransformError>;
pub fn translate(image: &DynamicImage, x: i32, y: i32) -> Result<DynamicImage, TransformError>;
```

**Batch Operations:**
```rust
pub fn batch_transform<F>(images: &[DynamicImage], transform_fn: F) -> Result<Vec<DynamicImage>, TransformError>
where F: Fn(&DynamicImage) -> Result<DynamicImage, TransformError> + Sync;

pub struct TransformPipeline {
    operations: Vec<Box<dyn Fn(DynamicImage) -> Result<DynamicImage, TransformError>>>,
}

impl TransformPipeline {
    pub fn new() -> Self;
    pub fn add<F>(self, op: F) -> Self where F: Fn(DynamicImage) -> Result<DynamicImage, TransformError> + 'static;
    pub fn execute(&self, image: &DynamicImage) -> Result<DynamicImage, TransformError>;
}
```

### 5.3 Configuration API

```rust
pub fn set_interpolation(interpolation: Interpolation);
pub fn set_background(color: [u8; 4]);
pub fn get_interpolation() -> Interpolation;
pub fn get_background() -> [u8; 4];
```

---

## 6. Testing Strategy

### 6.1 Unit Tests
- Each operation tested with RGB, RGBA, grayscale
- Edge cases (0x0, 1x1, large images)
- Invalid inputs (bounds checking, NaN values)

### 6.2 Integration Tests
- Pipeline operations
- Batch processing
- Frame sequence handling

### 6.3 Performance Tests
- Criterion benchmarks for all operations
- Target: <10ms for basic ops, <50ms for complex ops (10MP RGBA)
- Memory profiling (ensure < 2x input size)

### 6.4 Golden Image Tests
- Visual regression testing
- Generate reference images for all operations

---

## 7. Performance Targets

| Operation Category | Target (10MP RGBA) | Strategy |
|-------------------|-------------------|----------|
| Flips | <5 ms | SIMD + parallel rows |
| Rotate 90/180/270 | <5 ms | Parallel rows |
| Arbitrary rotation | <50 ms | SIMD interpolation |
| Crop | <2 ms | Zero-copy where possible |
| Resize | <30 ms | SIMD + parallel rows |
| Affine | <50 ms | SIMD + parallel rows |
| Perspective | <80 ms | Parallel rows |
| Batch (100 frames) | <2 s | Parallel processing |

---

## 8. Risk Assessment & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| SIMD complexity | High | Medium | Provide scalar fallback for all SIMD ops |
| Memory overruns | High | Low | Strict bounds checking + tests |
| Performance targets | Medium | Medium | Profile early, optimize hot paths |
| API breaking changes | Low | Low | Maintain existing API, add new functions |

---

## 9. Dependencies

### Current Dependencies
- `image` (v0.25) - Core image handling
- `rayon` (v1.10) - Parallel processing
- `nom-exif` (v2.5) - EXIF parsing
- `thiserror` (v1.0) - Error handling
- `fast_image_resize` (v5.3, optional) - Fast resize (not yet integrated)

### Additional Dependencies Needed
None - all operations can be implemented with existing dependencies.

---

## 10. Documentation Plan

1. **API Documentation**
   - Rustdoc for all public functions
   - Code examples for each operation
   - Performance characteristics

2. **Examples**
   - Basic transformations
   - Batch processing
   - Video frame processing
   - Pipeline composition

3. **Guides**
   - Migration guide (if breaking changes)
   - Performance tuning guide
   - SIMD optimization guide

---

## 11. Success Metrics

**Quantitative:**
- ✅ 52/52 operations implemented
- ✅ 100% test coverage for new operations
- ✅ All performance targets met
- ✅ Zero panics on fuzzing

**Qualitative:**
- ✅ Clean, maintainable code
- ✅ Consistent API design
- ✅ Comprehensive documentation
- ✅ Easy to extend for future operations

---

## 12. Next Steps

1. ✅ Design document review (THIS DOCUMENT)
2. Start implementation with Phase 1.1 (Categories 1-4)
3. Iterate with testing and optimization
4. Complete all phases within 6 days

---

## Appendix A: Complete Operation List

### Category 1: Flips (3)
1. flip_horizontal ✅
2. flip_vertical ✅
3. flip_both ❌

### Category 2: Rotations (7)
4. rotate_90_cw ✅
5. rotate_90_ccw ❌
6. rotate_180 ✅
7. rotate_270_cw ✅
8. rotate (arbitrary) ✅
9. rotate_bounded ❌
10. rotate_cropped ❌

### Category 3: Crops (6)
11. crop ✅
12. crop_center ❌
13. crop_to_aspect ❌
14. crop_percentage ❌
15. autocrop ❌
16. crop_to_content ❌

### Category 4: Resize/Scale (10)
17. resize ✅ (resize_exact)
18. resize_exact ✅
19. resize_fit ✅ (resize_to_fit)
20. resize_fill ❌
21. resize_cover ❌
22. scale ❌
23. scale_width ✅ (resize_to_width)
24. scale_height ✅ (resize_to_height)
25. downscale ❌
26. upscale ❌

### Category 5: Matrix Ops (2)
27. transpose ❌
28. transverse ❌

### Category 6: Affine (4)
29. shear_horizontal ❌
30. shear_vertical ❌
31. affine_transform ❌
32. translate ❌

### Category 7: Perspective (3)
33. perspective_transform ❌
34. perspective_correct ❌
35. homography ❌

### Category 8: Canvas (5)
36. pad ❌
37. pad_to_size ❌
38. pad_to_aspect ❌
39. expand_canvas ❌
40. trim ❌

### Category 9: Alignment (2)
41. center_on_canvas ❌
42. align ❌

### Category 10: Batch/Video (5)
43. batch_transform ❌
44. sequence_transform ❌
45. parallel_batch ❌
46. stream_transform ❌
47. pipeline_transform ❌

### Category 11: Frame Utils (3)
48. load_sequence ❌
49. save_sequence ❌
50. sequence_info ❌

### Category 12: Control (2)
51. set_interpolation ❌
52. set_background ❌

**Total:** 52 operations (16 complete, 36 to implement)
