import os
import shutil
import random

BASE        = r"E:\intern\project\Plate detection\dataset"
TRAIN_IMG   = os.path.join(BASE, "detection", "images", "train")
TRAIN_LBL   = os.path.join(BASE, "detection", "labels", "train")
VAL_IMG     = os.path.join(BASE, "detection", "images", "val")
VAL_LBL     = os.path.join(BASE, "detection", "labels", "val")

os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(VAL_LBL, exist_ok=True)

# Get all images that have a matching label
all_images = [f for f in os.listdir(TRAIN_IMG)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

paired = []
for img in all_images:
    stem = os.path.splitext(img)[0]
    lbl  = stem + ".txt"
    if os.path.exists(os.path.join(TRAIN_LBL, lbl)):
        paired.append(img)

print(f"Total paired images: {len(paired)}")

# Shuffle and split 20% to val
random.seed(42)
random.shuffle(paired)
val_count = int(len(paired) * 0.2)
val_files  = paired[:val_count]
train_kept = paired[val_count:]

print(f"Moving {val_count} images to val...")

for img in val_files:
    stem = os.path.splitext(img)[0]
    lbl  = stem + ".txt"

    # Move image
    shutil.move(os.path.join(TRAIN_IMG, img),
                os.path.join(VAL_IMG,   img))
    # Move label
    shutil.move(os.path.join(TRAIN_LBL, lbl),
                os.path.join(VAL_LBL,   lbl))

print(f"\n✅ Split complete!")
print(f"   Train images : {len(os.listdir(TRAIN_IMG))}")
print(f"   Val   images : {len(os.listdir(VAL_IMG))}")
print(f"   Total        : {len(os.listdir(TRAIN_IMG)) + len(os.listdir(VAL_IMG))}")