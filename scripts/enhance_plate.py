import cv2
import os
import sys
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# -- TensorFlow (blind deblur models) -----------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# -- Paths --------------------------------------------------------------------
ESRGAN_MODEL = r"E:\intern\project\Plate detection\models\realesrgan\realesr-general-x4v3.pth"
ANGLE_MODEL  = r"E:\intern\project\Plate detection\models\Blind-Motion-Deblurring-for-Legible-License-Plates-using-Deep-Learning\pretrained_models\angle_model.hdf5"
LENGTH_MODEL = r"E:\intern\project\Plate detection\models\Blind-Motion-Deblurring-for-Legible-License-Plates-using-Deep-Learning\pretrained_models\length_model.hdf5"
SIDEKICK_DIR = r"E:\intern\project\Plate detection\models\Blind-Motion-Deblurring-for-Legible-License-Plates-using-Deep-Learning\sidekick"
OUTPUT_DIR   = r"E:\intern\project\Plate detection\output\enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, SIDEKICK_DIR)

# Plates larger than this skip ESRGAN (already sharp enough - ESRGAN hurts them)
ESRGAN_SIZE_THRESHOLD = (120, 40)   # (width, height) in pixels


# =============================================================================
#  LOAD MODELS
# =============================================================================
def load_blind_deblur_models():
    print("  Loading blind deblur models (TensorFlow)...")
    angle_model  = tf.keras.models.load_model(ANGLE_MODEL)
    length_model = tf.keras.models.load_model(LENGTH_MODEL)
    print("  Blind deblur models loaded.")
    return angle_model, length_model


def load_esrgan():
    print("  Loading Real-ESRGAN v3...")
    model = SRVGGNetCompact(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_conv=32,
        upscale=4, act_type='prelu'
    )
    upsampler = RealESRGANer(
        scale=4, model_path=ESRGAN_MODEL, model=model,
        tile=0, tile_pad=10, pre_pad=0,
        half=True, device=torch.device('cuda')
    )
    print("  Real-ESRGAN v3 loaded on GPU.")
    return upsampler


# =============================================================================
#  STEP 1 - BLIND MOTION DEBLUR
# =============================================================================
def compute_fft_image(img_gray):
    f      = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    mag    = 20 * np.log(np.abs(fshift) + 1e-8)
    mag    = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag    = cv2.resize(mag.astype(np.uint8), (224, 224))
    return mag


def estimate_blur_params(img, angle_model, length_model):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fft  = compute_fft_image(gray)

    # Model expects (1, 224, 224, 1) - grayscale
    inp = fft.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=-1)  # -> (224, 224, 1)
    inp = np.expand_dims(inp, axis=0)   # -> (1, 224, 224, 1)

    angle_pred  = angle_model.predict(inp,  verbose=0)
    length_pred = length_model.predict(inp, verbose=0)

    angle  = int(np.argmax(angle_pred[0]))
    length = int(np.argmax(length_pred[0])) + 1

    length = max(1, min(length, 30))
    angle  = angle % 180

    return length, angle


def wiener_deconvolve(img, length, angle, noise_level=0.01):
    # If length==1 there's no meaningful blur - skip deconvolution
    if length <= 1:
        return img

    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length

    center = (length // 2, length // 2)
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    kernel = kernel / (kernel.sum() + 1e-8)

    result = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        channel    = img[:, :, c].astype(np.float32) / 255.0
        kernel_pad = np.zeros_like(channel)
        kh, kw     = kernel.shape
        kernel_pad[:kh, :kw] = kernel

        img_fft     = np.fft.fft2(channel)
        kernel_fft  = np.fft.fft2(kernel_pad)
        kernel_conj = np.conj(kernel_fft)
        wiener      = kernel_conj / (np.abs(kernel_fft) ** 2 + noise_level)
        restored    = np.fft.ifft2(img_fft * wiener).real
        result[:, :, c] = (np.clip(restored, 0, 1) * 255).astype(np.uint8)

    return result.astype(np.uint8)


def blind_deblur(img, angle_model, length_model):
    length, angle = estimate_blur_params(img, angle_model, length_model)
    print(f"  Estimated blur -> length: {length}px, angle: {angle} deg")
    if length <= 1:
        print("  No significant blur detected - skipping Wiener deconvolution")
        return img
    return wiener_deconvolve(img, length, angle, noise_level=0.01)


# =============================================================================
#  STEP 2 - DESKEW  (NEW - fixes tilted plates before upscaling)
# =============================================================================
def deskew(img):
    """
    Detect plate tilt using Hough Line Transform and correct it.
    This must run BEFORE ESRGAN - upscaling a tilted plate makes it worse.
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=20,
        minLineLength=img.shape[1] // 5,
        maxLineGap=10
    )

    if lines is None:
        print("  Deskew: no lines detected, skipping")
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < angle < 45:   # ignore near-vertical lines
                angles.append(angle)

    if not angles:
        print("  Deskew: could not determine angle, skipping")
        return img

    median_angle = float(np.median(angles))

    # Only rotate if tilt is meaningful (>1.5 degrees)
    if abs(median_angle) < 1.5:
        print(f"  Deskew: tilt is {median_angle:.1f} deg - within tolerance, skipping")
        return img

    print(f"  Deskew: correcting {median_angle:.1f} deg tilt")

    h, w   = img.shape[:2]
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    # Expand canvas so corners aren't clipped after rotation
    cos_a = abs(M[0, 0]); sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    deskewed = cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed


# =============================================================================
#  STEP 3 - SMART UPSCALE  (ESRGAN only when it actually helps)
# =============================================================================
def smart_upscale(img, upsampler):
    """
    - Small plates  (w <= 120 or h <= 40): use ESRGAN - genuine upscaling needed
    - Larger plates (w > 120 and h > 40) : use bicubic only - ESRGAN over-smooths
    """
    h, w = img.shape[:2]
    thresh_w, thresh_h = ESRGAN_SIZE_THRESHOLD

    if w > thresh_w and h > thresh_h:
        print(f"  Upscale: plate is {w}x{h}px -> bicubic 4x (ESRGAN skipped - plate already sharp)")
        upscaled = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        # Mild unsharp mask to crisp edges without over-sharpening
        blur     = cv2.GaussianBlur(upscaled, (0, 0), 1.5)
        upscaled = cv2.addWeighted(upscaled, 1.4, blur, -0.4, 0)
        return upscaled, "Bicubic 4x"
    else:
        print(f"  Upscale: plate is {w}x{h}px -> ESRGAN 4x (small plate, needs SR)")
        # Pre-upscale 2x with bicubic so ESRGAN has more to work with
        img_pre = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        enhanced, _ = upsampler.enhance(img_pre, outscale=4)
        blur     = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)
        return enhanced, "ESRGAN 4x"


# =============================================================================
#  STEP 4 - BINARY CONVERSION  (improved)
# =============================================================================
def make_binary(img):
    """
    Convert upscaled color plate to clean black-on-white binary.
    Uses both Otsu and Adaptive threshold, picks the better one.
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE - boost local contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)

    # Light denoise before thresholding
    gray = cv2.fastNlMeansDenoising(gray, h=5,
                                     templateWindowSize=7,
                                     searchWindowSize=15)

    # Slight blur to smooth CLAHE artifacts
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Strategy 1: Otsu global threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Strategy 2: Adaptive Gaussian (better for uneven lighting / shadows)
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=10
    )

    # Pick best: Otsu works well when white ratio is in the "plate zone" (50-85%)
    white_ratio = np.sum(otsu == 255) / otsu.size
    if 0.45 < white_ratio < 0.85:
        binary = otsu
        method = "Otsu"
    else:
        binary = adaptive
        method = "Adaptive"

    print(f"  Binary: {method} threshold selected (white ratio: {white_ratio:.2f})")

    # Auto-invert if plate came out inverted (characters should be dark/black)
    if np.sum(binary == 255) / binary.size < 0.5:
        binary = cv2.bitwise_not(binary)
        print("  Binary: auto-inverted (dark background detected)")

    # Morphological cleanup - close small gaps in characters
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    # Connected component cleanup - remove noise specks smaller than 20px
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(binary), connectivity=8)
    cleaned = np.full_like(binary, 255)
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] >= 20:
            cleaned[labels == i] = 0

    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR), method


# =============================================================================
#  MAIN ENHANCEMENT FUNCTION
# =============================================================================
def enhance_plate(image_path, angle_model, length_model, upsampler):
    img_original = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_original is None:
        print(f"Could not read: {image_path}")
        return None

    if "annotated" in os.path.basename(image_path).lower():
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Wrong File",
            "Select _plate1.jpg / _plate2.jpg\nNOT the _annotated.jpg")
        root.destroy()
        return None

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    h_orig, w_orig = img_original.shape[:2]
    print(f"\nEnhancing: {os.path.basename(image_path)} ({w_orig}x{h_orig} px)")

    # -- Step 1: Blind motion deblur ------------------------------------------
    print("  [1/4] Blind deblur (CNN estimates blur + Wiener deconvolution)...")
    deblurred = blind_deblur(img_original.copy(), angle_model, length_model)

    # -- Step 2: Deskew (straighten tilted plate) -----------------------------
    print("  [2/4] Deskewing (correcting plate tilt)...")
    deskewed = deskew(deblurred)

    # -- Step 3: Smart upscale (ESRGAN or bicubic based on plate size) --------
    print("  [3/4] Smart upscale...")
    enhanced_color, upscale_method = smart_upscale(deskewed, upsampler)

    # -- Step 4: Binary conversion --------------------------------------------
    print("  [4/4] Binary conversion (CLAHE -> Threshold -> Cleanup)...")
    enhanced_binary, binary_method = make_binary(enhanced_color.copy())

    # -- Save outputs ---------------------------------------------------------
    color_path  = os.path.join(OUTPUT_DIR, f"{base_name}_color.jpg")
    binary_path = os.path.join(OUTPUT_DIR, f"{base_name}_binary.jpg")
    cv2.imwrite(color_path,  enhanced_color)
    cv2.imwrite(binary_path, enhanced_binary)

    # 4-panel comparison: Original | Deskewed | Color Enhanced | Binary
    target_h = 300
    def rp(im):
        s = target_h / max(im.shape[0], 1)
        return cv2.resize(im, (max(int(im.shape[1] * s), 1), target_h),
                          interpolation=cv2.INTER_CUBIC)

    div = np.ones((target_h, 6, 3), dtype=np.uint8) * 160

    p_orig    = rp(img_original)
    p_desk    = rp(deskewed)
    p_color   = rp(enhanced_color)
    p_binary  = rp(enhanced_binary)

    comparison = np.hstack([p_orig, div, p_desk, div, p_color, div, p_binary])

    font = cv2.FONT_HERSHEY_SIMPLEX
    x0 = 10
    x1 = p_orig.shape[1] + 14
    x2 = x1 + p_desk.shape[1] + 14
    x3 = x2 + p_color.shape[1] + 14

    cv2.putText(comparison, "Original",            (x0, 30), font, 0.9, (0, 0, 255),    2)
    cv2.putText(comparison, "Deskewed",             (x1, 30), font, 0.9, (0, 180, 255),  2)
    cv2.putText(comparison, upscale_method,         (x2, 30), font, 0.9, (0, 200, 0),    2)
    cv2.putText(comparison, f"Binary [{binary_method}]", (x3, 30), font, 0.9, (30, 160, 255), 2)

    comp_path = os.path.join(OUTPUT_DIR, f"{base_name}_comparison.jpg")
    cv2.imwrite(comp_path, comparison)

    print(f"\n  Results saved:")
    print(f"  Color    -> {color_path}")
    print(f"  Binary   -> {binary_path}")
    print(f"  Comparison (4-panel) -> {comp_path}")
    print(f"\n  Pipeline summary:")
    print(f"    Input size   : {w_orig}x{h_orig} px")
    print(f"    Upscale used : {upscale_method}")
    print(f"    Binary method: {binary_method}")

    return binary_path


# =============================================================================
#  FILE PICKER
# =============================================================================
def pick_image():
    root = tk.Tk(); root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title="Select cropped plate (_plate1.jpg, _plate2.jpg ...)",
        initialdir=r"E:\intern\project\Plate detection\output\detected",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"),
                   ("All files",   "*.*")]
    )
    root.destroy()
    return path


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    print("License Plate Enhancer - Blind Deblur + Deskew + Smart Upscale + Binary")
    print("=" * 72)
    print("Select _plate1.jpg / _plate2.jpg  (NOT _annotated.jpg)\n")

    print("Loading models...")
    angle_model, length_model = load_blind_deblur_models()
    upsampler                 = load_esrgan()
    print("All models ready.\n")

    while True:
        image_path = pick_image()
        if not image_path:
            print("No file selected. Exiting.")
            break

        print(f"Selected: {image_path}")
        result = enhance_plate(image_path, angle_model, length_model, upsampler)

        if result:
            root = tk.Tk(); root.withdraw()
            again = messagebox.askyesno("Enhance Another?",
                f"Saved to:\n{OUTPUT_DIR}\n\nEnhance another plate?")
            root.destroy()
            if not again:
                break

    print("Done!")