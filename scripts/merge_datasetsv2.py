"""
merge_my_dataset.py
-------------------
Merges your 88 real-world annotated images from Roboflow into the
existing detection dataset, then prints final counts.

Source : E:\intern\project\Plate detection\my_dataset\Annoted\
Target : E:\intern\project\Plate detection\dataset\detection\
"""

import os
import shutil

# ── Paths ─────────────────────────────────────────────────────────────────────
SOURCE_TRAIN_IMG = r"E:\intern\project\Plate detection\my_dataset\Annoted\train\images"
SOURCE_TRAIN_LBL = r"E:\intern\project\Plate detection\my_dataset\Annoted\train\labels"
SOURCE_VAL_IMG   = r"E:\intern\project\Plate detection\my_dataset\Annoted\valid\images"
SOURCE_VAL_LBL   = r"E:\intern\project\Plate detection\my_dataset\Annoted\valid\labels"

TARGET_TRAIN_IMG = r"E:\intern\project\Plate detection\dataset\detection\images\train"
TARGET_TRAIN_LBL = r"E:\intern\project\Plate detection\dataset\detection\labels\train"
TARGET_VAL_IMG   = r"E:\intern\project\Plate detection\dataset\detection\images\val"
TARGET_VAL_LBL   = r"E:\intern\project\Plate detection\dataset\detection\labels\val"

# ── Helpers ───────────────────────────────────────────────────────────────────
def count_files(folder, ext=".jpg"):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(ext)])


def copy_folder(src, dst, label=""):
    if not os.path.exists(src):
        print(f"  [SKIP] Source not found: {src}")
        return 0

    os.makedirs(dst, exist_ok=True)
    files   = os.listdir(src)
    copied  = 0
    skipped = 0

    for fname in files:
        src_path = os.path.join(src, fname)
        dst_path = os.path.join(dst, fname)

        if os.path.exists(dst_path):
            skipped += 1
            continue

        shutil.copy2(src_path, dst_path)
        copied += 1

    print(f"  {label}: {copied} copied, {skipped} already existed (skipped)")
    return copied


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Merging real-world dataset into detection dataset")
    print("=" * 60)

    # -- Counts before merge --------------------------------------------------
    before_train = count_files(TARGET_TRAIN_IMG)
    before_val   = count_files(TARGET_VAL_IMG)
    print(f"\nBefore merge:")
    print(f"  Train images : {before_train}")
    print(f"  Val   images : {before_val}")

    # -- Copy train images + labels -------------------------------------------
    print("\nCopying train split...")
    copy_folder(SOURCE_TRAIN_IMG, TARGET_TRAIN_IMG, "Train images")
    copy_folder(SOURCE_TRAIN_LBL, TARGET_TRAIN_LBL, "Train labels")

    # -- Copy val images + labels ---------------------------------------------
    print("\nCopying val split...")
    copy_folder(SOURCE_VAL_IMG, TARGET_VAL_IMG, "Val images")
    copy_folder(SOURCE_VAL_LBL, TARGET_VAL_LBL, "Val labels")

    # -- Counts after merge ---------------------------------------------------
    after_train = count_files(TARGET_TRAIN_IMG)
    after_val   = count_files(TARGET_VAL_IMG)

    print(f"\nAfter merge:")
    print(f"  Train images : {after_train}  (+{after_train - before_train})")
    print(f"  Val   images : {after_val}  (+{after_val - before_val})")
    print(f"\n  Total dataset : {after_train + after_val} images")
    print("\nMerge complete! Ready to run train_yolo_v2.py")
    print("=" * 60)