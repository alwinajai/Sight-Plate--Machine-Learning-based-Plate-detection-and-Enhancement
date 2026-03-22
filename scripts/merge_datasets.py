import os
import shutil
import random

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE         = r"E:\intern\project\Plate detection"
DATASETS_DIR = r"E:\intern\project\Plate detection\downloaded_datasets"
TRAIN_IMGS   = os.path.join(BASE, "dataset", "detection", "images",  "train")
TRAIN_LBLS   = os.path.join(BASE, "dataset", "detection", "labels",  "train")
VAL_IMGS     = os.path.join(BASE, "dataset", "detection", "images",  "val")
VAL_LBLS     = os.path.join(BASE, "dataset", "detection", "labels",  "val")

os.makedirs(TRAIN_IMGS, exist_ok=True)
os.makedirs(TRAIN_LBLS, exist_ok=True)
os.makedirs(VAL_IMGS,   exist_ok=True)
os.makedirs(VAL_LBLS,   exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def copy_yolo_dataset(images_dir, labels_dir, split_ratio=0.9):
    """Copy images+labels to train/val split."""
    if not os.path.exists(images_dir):
        print(f"  Skipping (not found): {images_dir}")
        return 0, 0

    img_files = [f for f in os.listdir(images_dir)
                 if os.path.splitext(f)[1].lower() in IMG_EXTS]

    random.shuffle(img_files)
    split_idx = int(len(img_files) * split_ratio)
    train_files = img_files[:split_idx]
    val_files   = img_files[split_idx:]

    t_count = v_count = 0

    for fname in train_files:
        base = os.path.splitext(fname)[0]
        lbl  = base + ".txt"
        lbl_src = os.path.join(labels_dir, lbl)
        if not os.path.exists(lbl_src):
            continue
        shutil.copy2(os.path.join(images_dir, fname),
                     os.path.join(TRAIN_IMGS, fname))
        shutil.copy2(lbl_src, os.path.join(TRAIN_LBLS, lbl))
        t_count += 1

    for fname in val_files:
        base = os.path.splitext(fname)[0]
        lbl  = base + ".txt"
        lbl_src = os.path.join(labels_dir, lbl)
        if not os.path.exists(lbl_src):
            continue
        shutil.copy2(os.path.join(images_dir, fname),
                     os.path.join(VAL_IMGS, fname))
        shutil.copy2(lbl_src, os.path.join(VAL_LBLS, lbl))
        v_count += 1

    return t_count, v_count

total_train = total_val = 0

# ── Dataset 1 (already YOLO format) ───────────────────────────────────────────
print("Processing Dataset 1...")
d1_images = os.path.join(DATASETS_DIR, "1", "images")
d1_labels = os.path.join(DATASETS_DIR, "1", "labels")

# Dataset 1 has flat structure — images and labels in same folder
if not os.path.exists(d1_images):
    d1_images = os.path.join(DATASETS_DIR, "1")
    d1_labels = os.path.join(DATASETS_DIR, "1", "labels")

t, v = copy_yolo_dataset(d1_images, d1_labels)
total_train += t; total_val += v
print(f"  Dataset 1: {t} train, {v} val")

# ── Dataset 3 archive (already YOLO format with train/val split) ───────────────
print("Processing Dataset 3 (archive)...")
d3_train_imgs = os.path.join(DATASETS_DIR, "3", "archive", "images", "train")
d3_train_lbls = os.path.join(DATASETS_DIR, "3", "archive", "labels", "train")
d3_val_imgs   = os.path.join(DATASETS_DIR, "3", "archive", "images", "val")
d3_val_lbls   = os.path.join(DATASETS_DIR, "3", "archive", "labels", "val")

# Copy train directly
if os.path.exists(d3_train_imgs):
    img_files = [f for f in os.listdir(d3_train_imgs)
                 if os.path.splitext(f)[1].lower() in IMG_EXTS]
    for fname in img_files:
        base    = os.path.splitext(fname)[0]
        lbl_src = os.path.join(d3_train_lbls, base + ".txt")
        if not os.path.exists(lbl_src):
            continue
        shutil.copy2(os.path.join(d3_train_imgs, fname),
                     os.path.join(TRAIN_IMGS, fname))
        shutil.copy2(lbl_src, os.path.join(TRAIN_LBLS, base + ".txt"))
        total_train += 1

# Copy val directly
if os.path.exists(d3_val_imgs):
    img_files = [f for f in os.listdir(d3_val_imgs)
                 if os.path.splitext(f)[1].lower() in IMG_EXTS]
    for fname in img_files:
        base    = os.path.splitext(fname)[0]
        lbl_src = os.path.join(d3_val_lbls, base + ".txt")
        if not os.path.exists(lbl_src):
            continue
        shutil.copy2(os.path.join(d3_val_imgs, fname),
                     os.path.join(VAL_IMGS, fname))
        shutil.copy2(lbl_src, os.path.join(VAL_LBLS, base + ".txt"))
        total_val += 1

print(f"  Dataset 3 archive added to existing split")

print(f"\nMerge complete!")
print(f"  Total train: {total_train + len(os.listdir(TRAIN_IMGS))}")
print(f"  Total val  : {total_val   + len(os.listdir(VAL_IMGS))}")