import os
import xml.etree.ElementTree as ET
import shutil
import random

# ── Paths ──────────────────────────────────────────────────────────────────────
VOC_DIR    = r"E:\intern\project\Plate detection\downloaded_datasets\3\Indian_vehicle_dataset"
OUT_IMAGES = r"E:\intern\project\Plate detection\dataset\detection\images\train"
OUT_LABELS = r"E:\intern\project\Plate detection\dataset\detection\labels\train"

os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_LABELS, exist_ok=True)

# Class names that represent license plates in VOC annotations
PLATE_CLASSES = {
    "license plate", "licence plate", "numberplate", "number plate",
    "number_plate", "license_plate", "plate", "lp", "registration plate"
}

converted = 0
skipped   = 0
no_plate  = 0

for xml_file in os.listdir(VOC_DIR):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(VOC_DIR, xml_file)
    base_name = os.path.splitext(xml_file)[0]

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image dimensions
        size = root.find("size")
        if size is None:
            skipped += 1
            continue

        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        if img_w == 0 or img_h == 0:
            skipped += 1
            continue

        # Get image filename from XML
        filename_elem = root.find("filename")
        img_filename  = filename_elem.text if filename_elem is not None else None

        # Find matching image file
        img_src = None
        if img_filename:
            candidate = os.path.join(VOC_DIR, img_filename)
            if os.path.exists(candidate):
                img_src = candidate

        # If not found by XML filename, try matching by base name
        if img_src is None:
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                candidate = os.path.join(VOC_DIR, base_name + ext)
                if os.path.exists(candidate):
                    img_src = candidate
                    break

        if img_src is None:
            skipped += 1
            continue

        # Parse bounding boxes
        yolo_lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip().lower()

            # Accept any plate-related class name
            if not any(pc in class_name for pc in PLATE_CLASSES):
                continue

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Convert to YOLO format (normalized cx, cy, w, h)
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h

            # Clamp to [0,1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))

            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            no_plate += 1
            continue

        # Copy image and write label
        img_ext  = os.path.splitext(img_src)[1]
        img_dest = os.path.join(OUT_IMAGES, base_name + img_ext)
        lbl_dest = os.path.join(OUT_LABELS, base_name + ".txt")

        shutil.copy2(img_src, img_dest)
        with open(lbl_dest, "w") as f:
            f.write("\n".join(yolo_lines))

        converted += 1

    except Exception as e:
        print(f"Error processing {xml_file}: {e}")
        skipped += 1

print(f"\nPascal VOC → YOLO conversion complete!")
print(f"  Converted : {converted}")
print(f"  No plate  : {no_plate} (skipped — no license plate annotation)")
print(f"  Errors    : {skipped}")
print(f"  Output    → {OUT_IMAGES}")