from ultralytics import YOLO

if __name__ == '__main__':

    # ── Transfer learning from your existing best model ────────────────────────
    model = YOLO(r"E:\intern\project\Plate detection\models\yolo\lp_detector5\weights\best.pt")

    model.train(
        data=r"E:\intern\project\Plate detection\dataset\detection\data.yaml",

        epochs=50,
        imgsz=640,
        batch=16,
        name="lp_detector_v2",
        project=r"E:\intern\project\Plate detection\models\yolo",

        # ── Device ──────────────────────────────────────────────────────────────
        device=0,

        # ── Transfer learning settings ──────────────────────────────────────────
        # Lower LR since we're fine-tuning not training from scratch
        lr0=0.001,          # initial LR (was 0.01 in v1)
        lrf=0.01,           # final LR fraction
        warmup_epochs=2,    # shorter warmup for fine-tuning

        # ── Augmentation (same as v1) ────────────────────────────────────────────
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        erasing=0.4,

        # ── Training stability ───────────────────────────────────────────────────
        patience=20,
        save=True,
        save_period=10,
        val=True,
        plots=True,

        # ── Performance ─────────────────────────────────────────────────────────
        workers=0,          # Windows fix
        cache=False,
        amp=True,
    )

    print("\nTraining complete!")
    print(r"Best model: E:\intern\project\Plate detection\models\yolo\lp_detector_v2\weights\best.pt")