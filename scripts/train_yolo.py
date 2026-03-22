from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO(r"E:\intern\project\Plate detection\yolov8s.pt")

    model.train(
        data=r"E:\intern\project\Plate detection\dataset\detection\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="lp_detector",
        project=r"E:\intern\project\Plate detection\models\yolo",

        # --- Device ---
        device=0,               # RTX 4060 GPU

        # --- Augmentation ---
        hsv_h=0.015,            # hue variation
        hsv_s=0.7,              # saturation (simulates weather)
        hsv_v=0.4,              # brightness (simulates night/day)
        degrees=10.0,           # random rotation
        translate=0.1,          # random translation
        scale=0.5,              # random scale (far/close plates)
        fliplr=0.5,             # horizontal flip
        mosaic=1.0,             # mosaic augmentation
        erasing=0.4,            # random erasing (occlusion)

        # --- Training stability ---
        patience=20,            # early stop if no improvement
        save=True,
        save_period=10,         # checkpoint every 10 epochs
        val=True,
        plots=True,             # save training graphs

        # --- Performance ---
        workers=0,              # fix for Windows multiprocessing
        cache=False,
        amp=True,               # mixed precision for RTX 4060
    )

    print("\n✅ Training complete!")
    print(r"Best model: E:\intern\project\Plate detection\models\yolo\lp_detector\weights\best.pt")