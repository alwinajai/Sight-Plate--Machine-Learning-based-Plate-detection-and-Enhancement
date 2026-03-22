import xml.etree.ElementTree as ET
import os

VOC_DIR = r"E:\intern\project\Plate detection\downloaded_datasets\3\Indian_vehicle_dataset"

names    = set()
examined = 0

for f in os.listdir(VOC_DIR):
    if not f.endswith(".xml"):
        continue
    try:
        root = ET.parse(os.path.join(VOC_DIR, f)).getroot()
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is not None:
                names.add(name_elem.text.strip())
        examined += 1
    except Exception as e:
        print(f"Error in {f}: {e}")

print(f"Examined {examined} XML files")
print(f"\nAll class names found:")
for n in sorted(names):
    print(f"  '{n}'")