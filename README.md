# Sight-Plate — ML-Based License Plate Detection & Enhancement

> A complete end-to-end computer vision pipeline for detecting and restoring Indian vehicle license plates from images and video footage, built with YOLOv8, Real-ESRGAN v3, and Blind Motion Deblurring.

**Stack:** Python · YOLOv8s · Real-ESRGAN v3 · TensorFlow · OpenCV · PySide6 · CUDA

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why This Project](#2-why-this-project)
3. [System Architecture](#3-system-architecture)
4. [Tech Stack](#4-tech-stack)
5. [Repository Structure](#5-repository-structure)
6. [Setup & Installation](#6-setup--installation)
7. [Dataset Preparation](#7-dataset-preparation)
8. [Training the Detection Model](#8-training-the-detection-model)
9. [Enhancement Pipeline — How It Works](#9-enhancement-pipeline--how-it-works)
10. [Scripts Explained](#10-scripts-explained)
11. [GUI Application](#11-gui-application)
12. [Model Performance](#12-model-performance)
13. [Running the Project](#13-running-the-project)
14. [Output Structure](#14-output-structure)
15. [Known Limitations](#15-known-limitations)

---

## 1. Project Overview

Sight-Plate is a fully functional Automatic License Plate Recognition (ALPR) pipeline built specifically for **Indian road conditions**. Most open-source ALPR systems are trained on European or US plates and perform poorly on Indian roads due to differences in plate fonts, lighting conditions, camera angles, and motion blur from dashcam/CCTV footage.

This project addresses that gap by:

- Training a custom YOLOv8s model on **8,000+ Indian-specific plate images** including real-world photos taken on Indian roads
- Building a **multi-stage enhancement pipeline** that restores blurry, low-resolution, or tilted plates to a clean, readable binary output
- Packaging everything into a **professional desktop GUI** (PySide6) that works on both images and videos

---

## 2. Why This Project

Standard downloaded datasets contain clean, well-lit, front-facing plate images. Real CCTV footage and dashcam recordings produce plates that are:

- **Motion blurred** — due to vehicle movement
- **Low resolution** — plate occupies a small portion of the frame
- **Tilted** — camera angle, vehicle angle, or mounting position
- **Unevenly lit** — shadows, glare, nighttime

Each of these problems is addressed by a dedicated stage in the enhancement pipeline.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT (Image or Video)                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1 — YOLOv8s Detection                        │
│                                                                  │
│  • Scans entire image for license plates                         │
│  • Outputs bounding box coordinates + confidence score          │
│  • Crops each detected plate with 5px padding                   │
│  • Draws annotated overlay on original image/video              │
│                                                                  │
│  Model: lp_detector_v33/weights/best.pt                         │
│  Input size: 640×640  |  Confidence threshold: 0.25             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 2 — Blind Motion Deblur                       │
│                                                                  │
│  • Computes FFT magnitude spectrum of the plate crop            │
│  • Feeds FFT image into two CNN models:                          │
│      - angle_model.hdf5  → estimates blur angle (0–180°)        │
│      - length_model.hdf5 → estimates blur length (1–30px)       │
│  • Builds a motion PSF (Point Spread Function) kernel           │
│  • Applies Wiener deconvolution to mathematically reverse blur  │
│                                                                  │
│  Skips deconvolution if estimated blur length ≤ 1px (no blur)   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3 — Deskew                                    │
│                                                                  │
│  • Applies Canny edge detection to find dominant lines          │
│  • Uses Hough Line Transform to measure plate tilt angle        │
│  • Rotates the image to correct tilt if angle > 1.5°            │
│  • Expands canvas after rotation to avoid clipping corners      │
│                                                                  │
│  Skips if no lines detected or tilt is within tolerance         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 4 — Smart Upscaling                           │
│                                                                  │
│  Small plate  (w ≤ 120px or h ≤ 40px):                          │
│    → Pre-upscale 2× bicubic → Real-ESRGAN v3 4× SR              │
│    → Post-sharpen with unsharp mask                             │
│                                                                  │
│  Larger plate (w > 120px and h > 40px):                          │
│    → Bicubic 4× only (ESRGAN over-smooths already-sharp plates) │
│    → Light unsharp mask                                          │
│                                                                  │
│  Model: realesr-general-x4v3.pth (4.7MB, CCTV-optimised)        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 5 — Binary Conversion                         │
│                                                                  │
│  • CLAHE (Contrast Limited Adaptive Histogram Equalisation)     │
│    → boosts local contrast without blowing out highlights       │
│  • Fast Non-Local Means Denoising → removes upscale artifacts   │
│  • Gaussian blur → smooths CLAHE edges before thresholding      │
│  • Otsu global threshold → works best when contrast is uniform  │
│  • Adaptive Gaussian threshold → fallback for uneven lighting   │
│  • Auto-invert if background came out dark (non-standard plates)│
│  • Morphological close → fills small gaps in characters         │
│  • Connected component cleanup → removes noise specks < 20px    │
│                                                                  │
│  Output: clean white background, black characters               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT                                        │
│                                                                  │
│  output/detected/                                                │
│    ├── original_plate1.jpg       (cropped plate)                 │
│    ├── original_annotated.jpg    (full image with bbox overlay)  │
│    └── original_annotated.mp4    (for video input)              │
│                                                                  │
│  output/enhanced/                                                │
│    ├── original_p1_color.jpg     (ESRGAN upscaled, full colour)  │
│    └── original_p1_binary.jpg    (clean B&W, OCR-ready)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Tech Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| Detection | Ultralytics YOLOv8s | Plate localisation in full images |
| Deblurring | TensorFlow / Keras | CNN-based blur parameter estimation |
| Super Resolution | Real-ESRGAN v3 | 4× upscaling of small plate crops |
| Image Processing | OpenCV | Deskew, CLAHE, thresholding, morphology |
| GPU Acceleration | PyTorch + CUDA 12.4 | ESRGAN inference on RTX 4060 |
| GUI | PySide6 | Desktop application |
| Training | Ultralytics YOLO CLI | Transfer learning from YOLOv8s.pt |
| Annotation | Roboflow | Bounding box annotation of real-world photos |

---

## 5. Repository Structure

```
Sight-Plate/
│
├── scripts/
│   ├── app.py                  # PySide6 GUI — main desktop application
│   ├── detect_plate.py         # Standalone detection script (image + video)
│   ├── enhance_plate.py        # Standalone enhancement pipeline
│   ├── train_yolo_v3.py        # Latest training script (v3 — real-world data)
│   ├── train_yolo_v2.py        # v2 training script (kept for reference)
│   ├── train_yolo.py           # v1 original training (from scratch)
│   ├── merge_my_dataset.py     # Merges 88 real-world annotated photos
│   ├── merge_datasets.py       # Merges downloaded Roboflow/Kaggle datasets
│   ├── coco_to_yolo.py         # Converts COCO JSON annotations to YOLO .txt
│   ├── split_val.py            # Splits dataset into train/val (80/20)
│   └── check_voc_classes.py    # Inspects Pascal VOC XML class labels
│
├── models/
│   ├── yolo/
│   │   └── lp_detector_v33/
│   │       └── weights/
│   │           └── best.pt     # Best trained model (mAP50 = 0.895)
│   ├── realesrgan/
│   │   └── realesr-general-x4v3.pth
│   └── Blind-Motion-Deblurring-for-Legible-License-Plates-using-Deep-Learning/
│       └── pretrained_models/
│           ├── angle_model.hdf5
│           └── length_model.hdf5
│
├── output/
│   ├── detected/               # Detection outputs (crops + annotated images)
│   └── enhanced/               # Enhancement outputs (colour + binary)
│
├── dataset/
│   └── detection/
│       ├── images/
│       │   ├── train/          # ~6,800 training images
│       │   └── val/            # ~1,400 validation images
│       ├── labels/
│       │   ├── train/          # YOLO format .txt label files
│       │   └── val/
│       └── data.yaml           # Dataset config for YOLO training
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** Model weight files (`.pt`, `.pth`, `.hdf5`) are not included in the repository due to size limits. Download them from the [Releases](../../releases) page and place them in the corresponding `models/` subdirectories.

---

## 6. Setup & Installation

### Prerequisites

- Windows 10/11 (tested on Windows 11)
- Python 3.10.x
- NVIDIA GPU with CUDA support (CUDA 12.4 tested)
- Minimum 8GB VRAM recommended for ESRGAN inference

### Step-by-Step Installation

**Step 1 — Clone the repository**

```bash
git clone https://github.com/alwinajai/Sight-Plate--Machine-Learning-based-Plate-detection-and-Enhancement.git
cd Sight-Plate--Machine-Learning-based-Plate-detection-and-Enhancement
```

**Step 2 — Create a virtual environment**

```bash
python -m venv lp_env
lp_env\Scripts\activate
```

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — Download model weights**

Download the following files from the [Releases](../../releases) page:

| File | Destination |
|---|---|
| `best.pt` | `models/yolo/lp_detector_v33/weights/` |
| `realesr-general-x4v3.pth` | `models/realesrgan/` |
| `angle_model.hdf5` | `models/Blind-Motion-Deblurring.../pretrained_models/` |
| `length_model.hdf5` | `models/Blind-Motion-Deblurring.../pretrained_models/` |

**Step 5 — Verify GPU is available**

```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output: `GPU: NVIDIA GeForce RTX 4060 Laptop GPU`

---

## 7. Dataset Preparation

### Overview

The detection model was trained on a combined dataset of **8,280 images** across three sources:

| Source | Images | Format | Notes |
|---|---|---|---|
| Roboflow Universe (6 datasets) | ~6,800 | YOLO | Mixed Indian + international plates |
| Kaggle Indian plate datasets | ~1,400 | YOLO / COCO | Indian-specific plates |
| Real-world photos (Alwin Ajai) | 86 | YOLO (annotated via Roboflow) | Actual Indian road conditions |

### Dataset Format

Every image has a paired `.txt` label file in YOLO format:

```
# class  cx     cy     width  height
  0      0.523  0.812  0.187  0.064
```

All values are normalised (0–1) relative to image width and height. Class `0` = `license_plate`.

### Converting COCO to YOLO — `coco_to_yolo.py`

Some downloaded datasets use COCO JSON format. This script converts them:

```python
# What it does:
# 1. Reads _annotations.coco.json
# 2. For each image, finds all bounding boxes
# 3. Converts [x, y, width, height] (COCO pixels) 
#    to [cx, cy, w, h] (YOLO normalised)
# 4. Writes one .txt file per image

python scripts/coco_to_yolo.py
```

### Merging Datasets — `merge_datasets.py`

After downloading multiple datasets, this script merges them into the main training folder:

```python
# What it does:
# 1. Scans source folders (downloaded_datasets/1, /3/archive etc.)
# 2. Copies images to dataset/detection/images/train
# 3. Copies labels to dataset/detection/labels/train  
# 4. Skips files that already exist (no duplicates)
# 5. Prints before/after counts

python scripts/merge_datasets.py
```

### Adding Real-World Data — `merge_my_dataset.py`

After annotating 88 personal photos via Roboflow and downloading the YOLO export:

```python
# What it does:
# 1. Reads from my_dataset/Annoted/train/ and /valid/
# 2. Copies 69 images to training set, 17 to validation
# 3. Preserves Roboflow's 80/20 split
# 4. Zero naming conflicts — Roboflow uses unique hash filenames

python scripts/merge_my_dataset.py
```

### Splitting Validation Set — `split_val.py`

If a dataset has no pre-built val split:

```python
# What it does:
# 1. Takes 20% of training images randomly
# 2. Moves them (with their labels) to val/ folders
# 3. Ensures no image appears in both train and val

python scripts/split_val.py
```

---

## 8. Training the Detection Model

### Version History

| Version | Base Model | Epochs | Dataset Size | mAP50 | Key Change |
|---|---|---|---|---|---|
| v1 | YOLOv8s.pt | 50 | 4,111 | 0.859 | Trained from scratch |
| v2 | lp_detector5/best.pt | 50 | 8,194 | 0.906 | Transfer learning, 2× data |
| v3 | lp_detector_v2/best.pt | 40 | 8,280 | 0.895 | + 86 real Indian road photos, frozen backbone |

### Training Script — `train_yolo_v3.py`

This is the most recent training script. It performs transfer learning from the v2 checkpoint:

```python
from ultralytics import YOLO

model = YOLO(r"models\yolo\lp_detector_v2\weights\best.pt")
# ↑ Starts from already-trained v2 weights, not from scratch.
# This means the model already knows what a license plate looks like.
# Training only fine-tunes it to also recognise real Indian road conditions.

model.train(
    data=r"dataset\detection\data.yaml",  # Points to train/val image folders
    
    epochs=40,          # Fewer epochs than v1/v2 — fine-tuning needs less
    imgsz=640,          # All images resized to 640×640 before training
    batch=16,           # 16 images processed per GPU step
    name="lp_detector_v3",
    
    freeze=10,          # CRITICAL: Freezes the first 10 backbone layers.
                        # These layers detect basic features (edges, shapes).
                        # Already learned perfectly — no need to retrain.
                        # Only the detection head (final layers) gets updated.
    
    lr0=0.005,          # Learning rate: 10× lower than v1 (0.05).
                        # Small LR = gentle nudges to weights.
                        # Large LR would overwrite v2's learned knowledge.
    
    lrf=0.01,           # Final LR = lr0 × lrf = 0.005 × 0.01
    warmup_epochs=3,    # Gradually ramps up LR for first 3 epochs
    
    # Augmentation — tuned for real Indian road conditions:
    hsv_v=0.5,          # Brightness variation (harsh sun, shadows, night)
    degrees=15.0,       # More rotation (plates at angles on real roads)
    scale=0.6,          # Scale variation (near/far vehicles)
    mosaic=0.8,         # Stitches 4 images into 1 (dataset diversity)
    copy_paste=0.2,     # Pastes plate crops onto other backgrounds
                        # (effectively multiplies small real-world dataset)
    
    patience=20,        # Stops early if no improvement for 20 epochs
    workers=0,          # Windows fix: disables multiprocessing
    amp=True,           # Mixed precision — speeds up RTX 4060 training
)
```

### How Transfer Learning Works Here

```
YOLOv8s.pt (generic)
      │
      ▼  50 epochs, lr=0.01, 4,111 images
lp_detector5/best.pt  → mAP50 = 0.859
      │
      ▼  50 epochs, lr=0.001, 8,194 images (2× data)
lp_detector_v2/best.pt → mAP50 = 0.906
      │
      ▼  40 epochs, lr=0.005, 8,280 images (+ real photos)
         freeze=10 backbone layers
lp_detector_v33/best.pt → mAP50 = 0.895  ← deployed model
```

The slight dip from v2 (0.906) to v3 (0.895) on the validation metric is expected — the v3 val set now contains 17 harder real-world images that v2 never saw. In practice, v3 performs significantly better on actual Indian road photos.

---

## 9. Enhancement Pipeline — How It Works

### Stage 1: Blind Motion Deblur (`enhance_plate.py`)

Motion blur in CCTV footage has a specific pattern — it smears pixels in a straight line at a particular angle and length. The blind deblur stage figures out these parameters automatically using two pretrained CNN models.

```python
def _blind_deblur(img):
    # Step 1: Convert plate to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Compute FFT magnitude spectrum
    # The FFT of a motion-blurred image shows a distinctive streak
    # pattern perpendicular to the blur direction
    fft = compute_fft_image(gray)   # Outputs 224×224 grayscale FFT image
    
    # Step 3: Feed FFT into CNN models
    inp = fft.reshape(1, 224, 224, 1) / 255.0   # Normalise, add batch dim
    angle  = argmax(angle_model.predict(inp))    # 0–179 degrees
    length = argmax(length_model.predict(inp))+1 # 1–30 pixels
    
    # Step 4: Build PSF (Point Spread Function) kernel
    # This is a mathematical model of how the blur happened
    kernel = zeros((length, length))
    kernel[length//2, :] = 1.0 / length         # Horizontal motion kernel
    kernel = rotate(kernel, angle)               # Rotate to blur angle
    
    # Step 5: Wiener deconvolution in frequency domain
    # Mathematically reverses the blur by dividing out the PSF
    # noise_level=0.01 prevents amplifying noise during deconvolution
    for each colour channel:
        wiener_filter = conj(PSF_fft) / (|PSF_fft|² + 0.01)
        restored = ifft(fft(channel) × wiener_filter)
```

### Stage 2: Deskew

```python
def _deskew(img):
    # Step 1: Find edges
    edges = Canny(grayscale(img), threshold1=50, threshold2=150)
    
    # Step 2: Detect line segments using Hough Transform
    lines = HoughLinesP(edges, minLineLength=img.width/5)
    
    # Step 3: Calculate median angle of all near-horizontal lines
    # (ignores near-vertical lines which are character strokes)
    angles = [arctan2(y2-y1, x2-x1) for each line if -45° < angle < 45°]
    tilt = median(angles)
    
    # Step 4: Rotate to correct tilt (only if > 1.5° — avoids over-correction)
    # Canvas is expanded after rotation to prevent corner clipping
    M = getRotationMatrix2D(centre, tilt, scale=1.0)
    corrected = warpAffine(img, M, expanded_size)
```

### Stage 3: Smart Upscaling

```python
def _upscale(img):
    h, w = img.shape[:2]
    
    if w > 120 and h > 40:
        # Plate already has sufficient resolution
        # ESRGAN would over-smooth the existing detail
        # Simple bicubic 4× preserves sharpness better
        upscaled = resize(img, (w*4, h*4), INTER_CUBIC)
        return unsharp_mask(upscaled, strength=1.4)
    else:
        # Genuinely small plate — ESRGAN provides real detail
        # Pre-upscale 2× first so ESRGAN has more pixels to work with
        pre = resize(img, (w*2, h*2), INTER_CUBIC)
        enhanced = esrgan_model.enhance(pre, outscale=4)
        return unsharp_mask(enhanced, strength=1.4)
```

### Stage 4: Binary Conversion

```python
def _binary(img):
    # 1. CLAHE — boosts local contrast in 4×4 grid tiles
    #    clipLimit=2.5 prevents over-amplifying noise
    gray = CLAHE(clipLimit=2.5, tileGridSize=(4,4)).apply(grayscale(img))
    
    # 2. Denoise — removes ESRGAN upscaling artifacts
    gray = fastNlMeansDenoising(gray, h=5)
    
    # 3. Gaussian blur — smooths before thresholding
    gray = GaussianBlur(gray, (3,3))
    
    # 4. Try Otsu threshold first (best for uniform lighting)
    _, otsu = threshold(gray, 0, 255, THRESH_BINARY + THRESH_OTSU)
    
    # 5. Check white pixel ratio — plates should be mostly white background
    white_ratio = count(otsu == 255) / total_pixels
    
    if 0.45 < white_ratio < 0.85:
        binary = otsu                    # Otsu worked well
    else:
        binary = adaptiveThreshold(...)  # Fall back to local thresholding
    
    # 6. Auto-invert if plate came out dark background
    if white_ratio < 0.5:
        binary = bitwise_not(binary)
    
    # 7. Connected component analysis — remove noise specks
    #    Any black blob smaller than 20px is noise, not a character
    for each component:
        if area >= 20px: keep it
        else: erase it
```

---

## 10. Scripts Explained

### `detect_plate.py` — Standalone Detection

Runs YOLO detection on a single image or video. Opens a file picker dialog.

- **Image mode:** Detects plates, saves crops + annotated image, shows result
- **Video mode:** Processes frame by frame, saves annotated video + plate crops. Asks user for frame skip rate (1 = every frame, 2 = every 2nd frame) and whether to save frame snapshots

```bash
python scripts/detect_plate.py
```

### `enhance_plate.py` — Standalone Enhancement

Takes a cropped plate image and runs the full 4-stage enhancement pipeline. Opens a file picker pointing to `output/detected/`.

```bash
python scripts/enhance_plate.py
```

**Important:** Select `_plate1.jpg` style crops. Do NOT select `_annotated.jpg` — the script checks for this and shows an error.

### `app.py` — Full GUI Application

The main desktop application. Combines detection and enhancement into one interface. See [Section 11](#11-gui-application) for details.

```bash
python scripts/app.py
```

---

## 11. GUI Application

The GUI (`app.py`) is built with PySide6 and runs the full pipeline in a background thread so the interface stays responsive during processing.

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  TOPBAR: App title · Tech badges · System status chip               │
├──────────────┬──────────────────────────────────────────────────────┤
│              │  OVERVIEW: Plates Detected · Enhanced · Session Files │
│  SIDEBAR:    ├─────────────────────────────────────────────────────  │
│              │  PROGRESS: Message · % · Progress bar                 │
│  Upload      ├───────────────────────┬─────────────────────────────  │
│  File Info   │  RESULTS:             │  LOG:                         │
│  Options     │  Detected plate rows  │  Timestamped system messages  │
│  Analyze     │  with conf badges     │  colour-coded by level        │
│  Clear/Open  │                       │                               │
│              │                       │                               │
└──────────────┴───────────────────────┴─────────────────────────────  │
│  FOOTER: Version · Developed by Alwin Ajai · LinkedIn ↗              │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Features

- **Drag-and-drop** file upload or click to browse
- **Dynamic options** — "Save frame snapshots" only appears for video files
- **Live stats** — Plates Detected and Enhanced counters update in real time as analysis runs
- **Confidence badges** — green (≥ 80%), amber (60–80%), red (< 60%) for each result
- **Output folder menu** — clicking "Open Output" shows a dropdown: Detected Folder, Enhanced Folder, or Both
- **Threading** — YOLO, TensorFlow, and ESRGAN all run in a `QThread` so the UI never freezes

### Threading Architecture

```
Main Thread (UI)             Worker Thread (QThread)
      │                              │
      │──── worker.start() ─────>    │
      │                              │── YOLO predict()
      │<─── sig_prog(10, "...") ──   │
      │  [update progress bar]       │── _do_enhance()
      │<─── sig_plate(path, conf) ─  │
      │  [add result row, update     │── write annotated video
      │   stats counters]            │
      │<─── sig_done(True, "...") ─  │
      │  [re-enable Analyze btn,     │
      │   set chip to COMPLETE]      │
```

---

## 12. Model Performance

### Detection Metrics

| Version | mAP50 | mAP50-95 | Precision | Recall | Training Time |
|---|---|---|---|---|---|
| v1 | 0.859 | 0.550 | 0.889 | 0.790 | 1.65 hrs |
| v2 | 0.906 | 0.663 | 0.934 | 0.847 | 2.77 hrs |
| v3 | 0.895 | 0.646 | 0.935 | 0.828 | 1.92 hrs |

**Hardware:** NVIDIA RTX 4060 Laptop GPU (8GB VRAM), CUDA 12.4, PyTorch 2.5.1

**Note on v3 mAP:** The slight drop from v2 (0.906) to v3 (0.895) is because the v3 validation set includes 17 harder real-world Indian road images that v2 never saw. Precision actually improved (+0.001). In real-world usage on Indian road photos, v3 produces higher confidence detections than v2.

### Dataset Composition

| Dataset | Images | Source | Notes |
|---|---|---|---|
| Roboflow Indian plates (6 sources) | ~4,100 | Roboflow Universe | Base training set |
| Kaggle datasets (2 sources) | ~4,100 | Kaggle | Additional variety |
| Real-world Indian road photos | 86 | Personal — annotated via Roboflow | Actual deployment conditions |
| **Total** | **8,280** | | **80% train / 20% val** |

---

## 13. Running the Project

### Option A — Full GUI (Recommended)

```bash
# Activate environment
lp_env\Scripts\activate

# Launch application
python scripts/app.py
```

1. The app loads all models on startup (takes ~15–20 seconds)
2. Status chip changes from `● LOADING MODELS` → `● READY`
3. Drop or browse for an image/video
4. Check/uncheck enhancement options
5. Click **ANALYZE**
6. Results appear live in the right panel
7. Click **Open Output** → choose detected or enhanced folder

### Option B — Detection Only

```bash
python scripts/detect_plate.py
```

Useful for batch processing or when enhancement is not needed. Significantly faster.

### Option C — Enhance Existing Crops

```bash
python scripts/enhance_plate.py
```

Run this on crops already saved in `output/detected/`. Opens file picker in that directory.

### Option D — Retrain the Model

```bash
# 1. Merge any new data first
python scripts/merge_my_dataset.py

# 2. Run transfer learning from v3
python scripts/train_yolo_v3.py

# 3. Update MODEL_PATH in app.py, detect_plate.py to point to new weights
```

---

## 14. Output Structure

After running analysis on `car_photo.jpg`:

```
output/
├── detected/
│   ├── car_photo_plate1.jpg          # Cropped plate 1
│   ├── car_photo_plate2.jpg          # Cropped plate 2 (if multiple)
│   └── car_photo_annotated.jpg       # Full image with bounding boxes
│
└── enhanced/
    ├── car_photo_p1_color.jpg         # ESRGAN upscaled colour output
    ├── car_photo_p1_binary.jpg        # Clean black-on-white binary (OCR-ready)
    ├── car_photo_p2_color.jpg
    └── car_photo_p2_binary.jpg
```

For video input (`traffic.mp4`):

```
output/
├── detected/
│   ├── traffic_f00042_p1.jpg         # Plate crop from frame 42
│   ├── traffic_f00084_p2.jpg         # Plate crop from frame 84
│   ├── traffic_annotated.mp4         # Full annotated video
│   └── traffic_snapshots/            # (if snapshot option enabled)
│       ├── f00042_p1.jpg
│       └── f00084_p2.jpg
│
└── enhanced/
    ├── traffic_p1_color.jpg
    ├── traffic_p1_binary.jpg
    └── ...
```

---

## 15. Known Limitations

- **Very distant plates (< 30px wide in crop):** Even ESRGAN cannot recover enough detail at this resolution. The binary output will be noisy. A warning in the log would indicate low-confidence detection (conf < 0.40).
- **Night-time footage:** The model handles moderate low-light well due to `hsv_v` augmentation during training, but extreme darkness (no street lighting) degrades detection confidence significantly.
- **Non-standard plate formats:** Temporary plates, military plates, or heavily damaged plates may not be detected. The model was trained primarily on standard IND white/yellow plates.
- **Enhancement speed on video:** Running full ESRGAN enhancement on every plate crop in a long video is time-intensive. The pipeline enhances every 15th plate live during video processing and then runs a batch pass at the end. For very long videos (> 30 min), consider using detection-only mode.
- **No OCR:** The pipeline produces a clean binary image optimised for OCR but does not include a character recognition stage. Integration with EasyOCR or PaddleOCR is a planned future addition.

---

## Author

**Alwin Ajai**  
LinkedIn: [linkedin.com/in/alwin-ajai-817436201](https://www.linkedin.com/in/alwin-ajai-817436201/)  
GitHub: [github.com/alwinajai](https://github.com/alwinajai)

---

*Built during internship — 2026*
