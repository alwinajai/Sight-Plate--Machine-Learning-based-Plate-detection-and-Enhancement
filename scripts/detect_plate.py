import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from ultralytics import YOLO
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH = r"E:\intern\project\Plate detection\models\yolo\lp_detector_v33\weights\best.pt"
OUTPUT_DIR = r"E:\intern\project\Plate detection\output\detected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v'}


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_image(image_path, model, conf=0.25):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return []

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    results   = model.predict(source=image_path, conf=conf, device=0, verbose=False)
    boxes     = results[0].boxes
    cropped_paths = []

    if len(boxes) == 0:
        print(f"No plate detected in: {os.path.basename(image_path)}")
        return []

    print(f"Found {len(boxes)} plate(s) in {os.path.basename(image_path)}")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf_score      = float(box.conf[0])
        pad = 5
        h, w = img.shape[:2]
        crop = img[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
        crop_path = os.path.join(OUTPUT_DIR, f"{base_name}_plate{i+1}.jpg")
        cv2.imwrite(crop_path, crop)
        cropped_paths.append(crop_path)
        print(f"  Plate {i+1}: conf={conf_score:.2f} -> {crop_path}")
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f"LP {conf_score:.2f}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    ann_path = os.path.join(OUTPUT_DIR, f"{base_name}_annotated.jpg")
    cv2.imwrite(ann_path, img)
    print(f"  Annotated -> {ann_path}")

    print(f"  Done! Open annotated image to review:")
    print(f"  -> {ann_path}")
    return cropped_paths


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE PLATE SNAPSHOT  -  full frame + zoomed inset
# ══════════════════════════════════════════════════════════════════════════════
def save_plate_snapshot(frame, box, frame_idx, plate_idx, out_dir, conf_score):
    x1, y1, x2, y2 = box
    h_f, w_f = frame.shape[:2]
    pad = 8

    scale      = 640 / w_f
    frame_disp = cv2.resize(frame, (640, int(h_f * scale)))

    sx1 = int(x1 * scale); sy1 = int(y1 * scale)
    sx2 = int(x2 * scale); sy2 = int(y2 * scale)
    cv2.rectangle(frame_disp, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
    cv2.putText(frame_disp, f"LP {conf_score:.2f}",
                (sx1, max(sy1-8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame_disp, f"Frame {frame_idx}",
                (8, frame_disp.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    crop = frame[max(0,y1-pad):min(h_f,y2+pad),
                 max(0,x1-pad):min(w_f,x2+pad)]
    target_h   = 240
    zoom_scale = target_h / max(crop.shape[0], 1)
    zoom_w     = max(int(crop.shape[1] * zoom_scale), 1)
    zoomed     = cv2.resize(crop, (zoom_w, target_h), interpolation=cv2.INTER_CUBIC)
    cv2.putText(zoomed, "PLATE ZOOM", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    panel_h = max(frame_disp.shape[0], target_h)
    fp = frame_disp
    if fp.shape[0] < panel_h:
        fp = cv2.copyMakeBorder(fp, 0, panel_h - fp.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(30,30,30))
    zp = zoomed
    if zp.shape[0] < panel_h:
        zp = cv2.copyMakeBorder(zp, 0, panel_h - zp.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(30,30,30))

    divider   = 255 * np.ones((panel_h, 4, 3), dtype=np.uint8)
    composite = np.hstack([fp, divider, zp])

    snap_path = os.path.join(out_dir, f"frame{frame_idx:05d}_plate{plate_idx}.jpg")
    cv2.imwrite(snap_path, composite)
    return snap_path


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_video(video_path, model, conf=0.25, skip_frames=2,
                 save_frames=True, progress_callback=None):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None

    base_name    = os.path.splitext(os.path.basename(video_path))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = os.path.join(OUTPUT_DIR, f"{base_name}_annotated.mp4")
    writer = cv2.VideoWriter(out_video_path,
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    snaps_dir = os.path.join(OUTPUT_DIR, f"{base_name}_snapshots")
    if save_frames:
        os.makedirs(snaps_dir, exist_ok=True)

    print(f"\nProcessing: {os.path.basename(video_path)}")
    print(f"  {w}x{h} @ {fps:.1f}fps  |  {total_frames} frames")
    print(f"  Detection every {skip_frames} frame(s)")

    frame_idx   = 0
    plate_count = 0
    last_boxes  = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_frames == 0:
            results    = model.predict(source=frame, conf=conf,
                                       device=0, verbose=False, stream=False)
            last_boxes = results[0].boxes

        for i, box in enumerate(last_boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score      = float(box.conf[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"LP {conf_score:.2f}", (x1, max(y1-8,12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)

            if save_frames and frame_idx % skip_frames == 0:
                plate_count += 1
                save_plate_snapshot(
                    frame.copy(), (x1,y1,x2,y2),
                    frame_idx, plate_count, snaps_dir, conf_score
                )

        cv2.putText(frame, f"Frame {frame_idx}/{total_frames}",
                    (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        writer.write(frame)

        if progress_callback:
            progress_callback(frame_idx, total_frames)

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"\n  Done! Frames={frame_idx} | Plates={plate_count}")
    print(f"  Annotated video -> {out_video_path}")
    if save_frames:
        print(f"  Snapshots       -> {snaps_dir}")

    return out_video_path, snaps_dir, plate_count


# ══════════════════════════════════════════════════════════════════════════════
#  PROGRESS WINDOW
# ══════════════════════════════════════════════════════════════════════════════
class ProgressWindow:
    def __init__(self, title="Processing..."):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("440x110")
        self.root.resizable(False, False)
        self.root.attributes('-topmost', True)
        tk.Label(self.root, text="Processing video - please wait...",
                 font=("Arial", 11)).pack(pady=(14,4))
        self.bar = ttk.Progressbar(self.root, length=400, mode='determinate')
        self.bar.pack(pady=4)
        self.lbl = tk.Label(self.root, text="", font=("Arial", 9))
        self.lbl.pack()

    def update(self, cur, total):
        pct = int(cur / max(total,1) * 100)
        self.bar['value'] = pct
        self.lbl.config(text=f"{cur} / {total} frames  ({pct}%)")
        self.root.update()

    def close(self):
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  FILE PICKER
# ══════════════════════════════════════════════════════════════════════════════
def pick_file():
    root = tk.Tk(); root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title="Select an Image or Video File",
        filetypes=[
            ("All supported", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv *.wmv"),
            ("Images",        "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("Videos",        "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v"),
            ("All files",     "*.*")
        ]
    )
    root.destroy()
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("License Plate Detector - Image & Video  [v3 model: mAP50=0.895]")
    print("=" * 60)

    model = YOLO(MODEL_PATH)

    while True:
        print("\nOpening file picker...")
        file_path = pick_file()
        if not file_path:
            print("No file selected. Exiting.")
            break

        ext = os.path.splitext(file_path)[1].lower()
        print(f"Selected: {file_path}")

        # ── IMAGE ────────────────────────────────────────────────────────────
        if ext in IMAGE_EXTS:
            crops = detect_image(file_path, model, conf=0.25)
            root = tk.Tk(); root.withdraw()
            messagebox.showinfo("Done",
                f"Found {len(crops)} plate(s)\nSaved to:\n{OUTPUT_DIR}")
            root.destroy()

        # ── VIDEO ────────────────────────────────────────────────────────────
        elif ext in VIDEO_EXTS:
            root = tk.Tk(); root.withdraw()

            skip = simpledialog.askinteger(
                "Detection Speed",
                "Run detection every N frames:\n"
                "  1 = every frame (slowest, most detections)\n"
                "  2 = every 2nd frame  <- recommended\n"
                "  5 = every 5th frame  (fastest)",
                initialvalue=2, minvalue=1, maxvalue=10, parent=root)
            if skip is None:
                skip = 2

            save_frames = messagebox.askyesno(
                "Save Annotated Frames?",
                "Save annotated frame snapshots?\n\n"
                "Each snapshot shows:\n"
                "  - Full frame with detection box\n"
                "  - Zoomed-in plate on the right\n\n"
                "Yes = save snapshots (uses more disk space)\n"
                "No  = only save the annotated video"
            )
            root.destroy()

            pw = ProgressWindow(f"Processing: {os.path.basename(file_path)}")
            result = detect_video(file_path, model, conf=0.25,
                                  skip_frames=skip, save_frames=save_frames,
                                  progress_callback=pw.update)
            pw.close()

            if result:
                out_video, snaps_dir, count = result
                root = tk.Tk(); root.withdraw()
                msg = (f"Processing complete!\n\n"
                       f"Plates detected : {count}\n"
                       f"Annotated video -> {out_video}")
                if save_frames:
                    msg += f"\nSnapshots       -> {snaps_dir}"
                messagebox.showinfo("Video Done", msg)
                root.destroy()
        else:
            print(f"Unsupported file type: {ext}")

        root = tk.Tk(); root.withdraw()
        again = messagebox.askyesno("Process Another?",
                                    "Process another image or video?")
        root.destroy()
        if not again:
            break

    print("\nExiting.")