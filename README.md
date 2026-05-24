# ClearShot — Training Data Extractor

Extract sharp, face-focused frames from video for AI/ML training data. Cross-platform Gradio app.

## Features

- **Smart frame sampling** — Configurable FPS extraction rate
- **Blur rejection** — Laplacian variance filter removes motion-blurred frames
- **Face detection** — MediaPipe-powered face detection with confidence tuning
- **Crop modes** — Face-focused or full-body crop
- **Square output** — Center-crop or letterbox to square, at configurable resolution
- **De-duplication** — Perceptual hashing skips near-identical frames
- **Batch download** — ZIP export of all extracted frames

## Requirements

- **Python 3.10+**
- **FFmpeg** (optional but recommended for accurate video metadata)

### Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
```bash
# Using winget
winget install ffmpeg

# Or using choco
choco install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

## Setup

```bash
# Clone / navigate to the project
cd clear-shot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will be available at **http://localhost:7860**

## Usage

1. Upload a video file
2. Adjust extraction settings (FPS, blur threshold, crop mode, etc.)
3. Click **Extract Frames**
4. Review results in the gallery
5. Download all frames as a ZIP

## Settings Guide

| Setting | Default | Description |
|---------|---------|-------------|
| Extraction FPS | 2.0 | Frames sampled per second of video |
| Blur Threshold | 100 | Higher = stricter blur rejection |
| Detection Confidence | 0.5 | Minimum face detection score |
| Crop Mode | Face | Face-focused or full body |
| Padding % | 20% | Extra margin around detected region |
| Square Method | Center Crop | Center-crop or letterbox padding |
| Output Size | 512px | Final image dimensions |
| Output Format | PNG | PNG (lossless) or JPG |
| De-dup Sensitivity | 8 | 0 = off, higher = more aggressive |
