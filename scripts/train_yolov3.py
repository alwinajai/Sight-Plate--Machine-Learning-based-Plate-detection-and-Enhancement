"""
train_yolo_v3.py  (updated)
----------------------------
Transfer learning from lp_detector_v2/best.pt
Fine-tuned on merged dataset including 86 real-world Indian road photos.

Key fixes vs previous v3 attempt:
  - lr0 raised from 0.0005 -> 0.005  (was too cautious, model couldn't learn)
  - freeze=10 added  (backbone stays fixed, only detection head adapts)
  - patience raised to 20  (gives more room to improve)
  - epochs raised to 40    (previous run stopped at 16 - need more room)
  - copy_paste raised to 0.2  (more effective for small real-world additions)
  - mosaic raised to 0.8   (better dataset mixing)
"""

from ultralytics import YOLO

if __name__ == '__main__':

    # Start from v2 best checkpoint
    model = YOLO(r"E:\intern\project\Plate detection\models\yolo\lp_detector_v2\weights\best.pt")

    model.train(
        data=r"E:\intern\project\Plate detection\dataset\detection\data.yaml",

        epochs=40,
        imgsz=640,
        batch=16,
        name="lp_detector_v3",
        project=r"E:\intern\project\Plate detection\models\yolo",

        # ── Device ───────────────────────────────────────────────────────────
        device=0,

        # ── Freeze backbone (key fix) ─────────────────────────────────────────
        # Freezes the first 10 layers (backbone - edge/shape detectors)
        # Only the detection head (final layers) gets updated.
        # This protects everything v2 learned while adapting to real photos.
        freeze=10,

        # ── Learning rate (key fix) ───────────────────────────────────────────
        # Previous attempt used 0.0005 - too small, model couldn't move.
        # 0.005 gives enough room to adapt without overwriting v2 knowledge.
        lr0=0.005,
        lrf=0.01,
        warmup_epochs=3,       # longer warmup to ease into training

        # ── Augmentation tuned for real-world Indian road photos ──────────────
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,             # stronger brightness variation (outdoor lighting)
        degrees=15.0,          # more rotation (plates at angles on real roads)
        translate=0.1,
        scale=0.6,             # more scale variation (near/far vehicles)
        fliplr=0.5,
        mosaic=0.8,            # raised from 0.5 - better dataset mixing
        erasing=0.4,
        copy_paste=0.2,        # raised from 0.1 - very effective for small additions
                               # pastes real plate crops onto other backgrounds

        # ── Training stability ────────────────────────────────────────────────
        patience=20,           # raised from 15 - give model more room to improve
        save=True,
        save_period=10,
        val=True,
        plots=True,

        # ── Performance ──────────────────────────────────────────────────────
        workers=0,             # Windows multiprocessing fix
        cache=False,
        amp=True,              # mixed precision for RTX 4060
    )

    print("\nTraining complete!")
    print(r"Best model: E:\intern\project\Plate detection\models\yolo\lp_detector_v3\weights\best.pt")
    print("\nCompare with v2 baseline:")
    print("  v2: mAP50=0.906  Precision=0.934  Recall=0.847")
    print("  v3: check above results")