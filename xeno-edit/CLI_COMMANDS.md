# xeno-edit CLI Commands

`xeno-edit` is the command-line interface for `xeno-lib`.

For full flags and defaults on any command:

```bash
xeno-edit <command> --help
```

## Quick Start

```bash
# Remove background
xeno-edit remove-bg photo.jpg

# Convert format
xeno-edit convert webp photo.png

# Recenter subject and resize canvas output
xeno-edit recenter logo.png --resize 512x512

# Batch image filter
xeno-edit image-filter --brightness 10 --contrast 15 *.png
```

## Command Index

| Command | Alias | Purpose |
|---|---|---|
| `remove-bg` | `rmbg` | AI background removal |
| `convert` | `cvt` | Convert images between formats |
| `recenter` | `rc` | Center transparent subject content on canvas |
| `image-filter` | `imgf` | Apply image processing pipeline |
| `gif` | - | Build animated GIF from images |
| `awebp` | `webp-anim` | Build animated WebP from images |
| `video-info` | `vinfo` | Show video metadata |
| `video-encode` | `av1` | Encode image sequence to AV1 |
| `gpu-info` | `nvdec` | Show GPU decode capabilities |
| `audio-info` | `ainfo` | Show audio metadata |
| `extract-audio` | `xaudio` | Extract audio to WAV |
| `video-frames` | `vframes` | Extract frames from video |
| `video-to-gif` | `v2gif` | Convert video to GIF |
| `video-thumbnail` | `vthumb` | Generate video thumbnail |
| `encode-sequence` | `eseq` | Encode pattern sequence to AV1 |
| `video-transcode` | `vtrans` | Decode/transform/re-encode video |
| `capabilities` | `caps` | Output enabled library capabilities |
| `video-trim` | `vtrim` | Trim video by time range |
| `video-concat` | `vcat` | Concatenate videos |
| `text-overlay` | `drawtext` | Draw text onto images |
| `h264-encode` | `h264` | Encode image sequence to H.264/MP4 |
| `audio-encode` | `aenc` | Encode WAV/FLAC audio |
| `exec` | `run` | Execute JSON operation config |
| `template` | `tpl` | Generate JSON config template |

## Image Commands

### `remove-bg` (`rmbg`)

```bash
xeno-edit remove-bg input.png --output-dir out
```

### `convert` (`cvt`)

```bash
xeno-edit convert webp input.png --quality 90
```

### `recenter` (`rc`)

Centers the non-transparent subject inside the original canvas.  
Optional resize happens after recenter.

```bash
# Recenter only
xeno-edit recenter input.png

# Recenter + resize to exact dimensions
xeno-edit recenter input.png --resize 512x512

# Batch with threshold and interpolation
xeno-edit recenter *.png --output-dir out --alpha-threshold 8 --interpolation bilinear
```

### `image-filter` (`imgf`)

```bash
xeno-edit image-filter input.jpg --brightness 10 --contrast 10 --blur 2
```

### `gif`

```bash
xeno-edit gif output.gif frame1.png frame2.png frame3.png --delay 100 --loops 0
```

### `awebp` (`webp-anim`)

```bash
xeno-edit awebp output.webp frame1.png frame2.png frame3.png --delay 100 --quality 80
```

### `text-overlay` (`drawtext`)

```bash
xeno-edit text-overlay image.png --text "Hello" --font ./font.ttf --x 40 --y 40
```

## Video Commands

### `video-info` (`vinfo`)

```bash
xeno-edit video-info input.ivf --json
```

### `video-encode` (`av1`)

```bash
xeno-edit video-encode output.ivf frame1.png frame2.png --fps 30 --speed 6 --quality 80
```

### `h264-encode` (`h264`)

```bash
xeno-edit h264-encode output.mp4 frame1.png frame2.png --fps 30 --bitrate 4000
```

### `video-frames` (`vframes`)

```bash
xeno-edit video-frames input.ivf out_frames --format png --every 1
```

### `video-to-gif` (`v2gif`)

```bash
xeno-edit video-to-gif input.ivf output.gif --fps 12 --width 640
```

### `video-thumbnail` (`vthumb`)

```bash
xeno-edit video-thumbnail input.ivf thumb.jpg --position 1.5 --width 640
```

### `encode-sequence` (`eseq`)

```bash
xeno-edit encode-sequence "frame_%04d.png" output.ivf --start 1 --end 240 --fps 24
```

### `video-transcode` (`vtrans`)

```bash
xeno-edit video-transcode input.ivf output.ivf --width 1280 --height 720 --codec av1
```

### `video-trim` (`vtrim`)

```bash
xeno-edit video-trim input.ivf output_trim.ivf --start 2.0 --end 8.0
```

### `video-concat` (`vcat`)

```bash
xeno-edit video-concat output.ivf clip1.ivf clip2.ivf clip3.ivf
```

## Audio Commands

### `audio-info` (`ainfo`)

```bash
xeno-edit audio-info input.wav --json
```

### `extract-audio` (`xaudio`)

```bash
xeno-edit extract-audio input.mp4 output.wav --mono
```

### `audio-encode` (`aenc`)

```bash
xeno-edit audio-encode output.flac input1.wav input2.wav --format flac --sample-rate 48000 --bits 16
```

## Agent and Automation Commands

### `capabilities` (`caps`)

```bash
xeno-edit capabilities
```

### `gpu-info` (`nvdec`)

```bash
xeno-edit gpu-info --device 0 --json
```

### `exec` (`run`)

Executes a JSON operation config (file path, `-` for stdin, or inline JSON string).

```bash
xeno-edit exec config.json
```

### `template` (`tpl`)

Generates JSON templates for operations.

```bash
xeno-edit template transcode
```
