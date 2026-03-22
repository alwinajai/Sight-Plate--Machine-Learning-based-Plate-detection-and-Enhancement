import json
import os
import shutil
from pathlib import Path

def coco_to_yolo(coco_json_path, images_src_dir, output_img_dir, output_lbl_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # Build image id → filename map
    img_map = {img['id']: img for img in coco['images']}

    # Build image id → annotations map
    ann_map = {}
    for ann in coco['annotations']:
        iid = ann['image_id']
        if iid not in ann_map:
            ann_map[iid] = []
        ann_map[iid].append(ann)

    converted = 0
    skipped = 0

    for img_id, img_info in img_map.items():
        fname = img_info['file_name']
        # Handle subfolders in filename
        fname_base = os.path.basename(fname)
        w = float(img_info['width'])
        h = float(img_info['height'])

        # Find source image (search recursively)
        src_path = None
        for root, dirs, files in os.walk(images_src_dir):
            if fname_base in files:
                src_path = os.path.join(root, fname_base)
                break

        if src_path is None:
            skipped += 1
            continue

        # Copy image
        dst_img = os.path.join(output_img_dir, fname_base)
        shutil.copy2(src_path, dst_img)

        # Write YOLO label
        stem = Path(fname_base).stem
        lbl_path = os.path.join(output_lbl_dir, stem + '.txt')

        with open(lbl_path, 'w') as lf:
            if img_id in ann_map:
                for ann in ann_map[img_id]:
                    x, y, bw, bh = [float(v) for v in ann['bbox']]  # COCO: x_min, y_min, width, height
                    # Convert to YOLO: x_center, y_center, width, height (normalized)
                    xc = (x + bw / 2) / w
                    yc = (y + bh / 2) / h
                    nw = bw / w
                    nh = bh / h
                    lf.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

        converted += 1

    print(f"  ✅ Converted: {converted} images")
    print(f"  ⚠️  Skipped : {skipped} images (not found)")
    return converted


# ── Dataset configs ──────────────────────────────────────────────
BASE = r"E:\intern\project\Plate detection\dataset"

DATASETS = [
    {
        "name": "indian_1",
        "json_train": f"{BASE}/raw_indian_1/train/_annotations.coco.json",
        "json_val":   f"{BASE}/raw_indian_1/valid/_annotations.coco.json",
        "img_train":  f"{BASE}/raw_indian_1/train",
        "img_val":    f"{BASE}/raw_indian_1/valid",
    },
    {
        "name": "rain",
        "json_train": f"{BASE}/raw_rain/train/_annotations.coco.json",
        "json_val":   f"{BASE}/raw_rain/valid/_annotations.coco.json",
        "img_train":  f"{BASE}/raw_rain/train",
        "img_val":    f"{BASE}/raw_rain/valid",
    },
    {
        "name": "indian_2",
        "json_train": f"{BASE}/raw_indian_2/train/_annotations.coco.json",
        "json_val":   f"{BASE}/raw_indian_2/valid/_annotations.coco.json",
        "img_train":  f"{BASE}/raw_indian_2/train",
        "img_val":    f"{BASE}/raw_indian_2/valid",
    },
    {
        "name": "indian_3",
        "json_train": f"{BASE}/raw_indian_3/train/_annotations.coco.json",
        "json_val":   f"{BASE}/raw_indian_3/valid/_annotations.coco.json",
        "img_train":  f"{BASE}/raw_indian_3/train",
        "img_val":    f"{BASE}/raw_indian_3/valid",
    },
    {
        "name": "vehicle",
        "json_train": f"{BASE}/raw_vehicle/train/_annotations.coco.json",
        "json_val":   f"{BASE}/raw_vehicle/valid/_annotations.coco.json",
        "img_train":  f"{BASE}/raw_vehicle/train",
        "img_val":    f"{BASE}/raw_vehicle/valid",
    },
    {
        "name": "cctv",
        "json_train": f"{BASE}/raw_cctv/train/_annotations.coco.json",
        "json_val":   f"{BASE}/raw_cctv/valid/_annotations.coco.json",
        "img_train":  f"{BASE}/raw_cctv/train",
        "img_val":    f"{BASE}/raw_cctv/valid",
    },
]

OUT_TRAIN_IMG = f"{BASE}/detection/images/train"
OUT_TRAIN_LBL = f"{BASE}/detection/labels/train"
OUT_VAL_IMG   = f"{BASE}/detection/images/val"
OUT_VAL_LBL   = f"{BASE}/detection/labels/val"

# ── Run conversion ───────────────────────────────────────────────
total = 0
for ds in DATASETS:
    print(f"\n📂 Processing: {ds['name']}")

    if os.path.exists(ds['json_train']):
        print("  [train]")
        total += coco_to_yolo(ds['json_train'], ds['img_train'], OUT_TRAIN_IMG, OUT_TRAIN_LBL)
    else:
        print(f"  ⚠️  No train JSON found, skipping")

    if os.path.exists(ds['json_val']):
        print("  [val]")
        total += coco_to_yolo(ds['json_val'], ds['img_val'], OUT_VAL_IMG, OUT_VAL_LBL)
    else:
        print(f"  ⚠️  No val JSON found, skipping")

print(f"\n🎉 Total images converted: {total}")
print(f"   Train: {len(os.listdir(OUT_TRAIN_IMG))} images")
print(f"   Val  : {len(os.listdir(OUT_VAL_IMG))} images")