import json
import re
import subprocess
from tkinter import Tk, filedialog

KAGGLE_PATTERN = r"/kaggle/input/([a-zA-Z0-9\-_]+)"


def extract_kaggle_datasets(notebook):
    datasets = set()

    with open(notebook, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            for line in cell.get("source", []):
                matches = re.findall(KAGGLE_PATTERN, line)
                for m in matches:
                    datasets.add(m)

    return list(datasets)


def find_dataset_owner(dataset_name):

    print(f"\nSearching Kaggle for: {dataset_name}")

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "list", "-s", dataset_name],
            capture_output=True,
            text=True
        )

        lines = result.stdout.split("\n")

        for line in lines:
            if "/" in line and dataset_name in line:

                dataset_id = line.strip().split()[0]

                if "/" in dataset_id:
                    print(f"Found dataset: {dataset_id}")
                    return dataset_id

    except Exception as e:
        print("Search error:", e)

    return None


def download_dataset(dataset_name, save_folder):

    dataset_id = find_dataset_owner(dataset_name)

    if dataset_id is None:
        print(f"Dataset not found on Kaggle: {dataset_name}")
        return

    print(f"\nDownloading: {dataset_id}")

    try:
        subprocess.run([
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_id,
            "-p",
            save_folder,
            "--unzip"
        ], check=True)

        print(f"Download complete: {dataset_id}")

    except Exception as e:
        print(f"Download failed: {dataset_id}")
        print(e)


# File selection
root = Tk()
root.withdraw()

print("Select notebook files")

notebooks = filedialog.askopenfilenames(
    title="Select Jupyter notebooks",
    filetypes=[("Jupyter Notebook", "*.ipynb")]
)

if not notebooks:
    print("No notebooks selected")
    exit()


print("Select folder to save datasets")

save_folder = filedialog.askdirectory()

if not save_folder:
    print("No folder selected")
    exit()


datasets_found = set()

for nb in notebooks:
    print(f"\nScanning {nb}")

    datasets = extract_kaggle_datasets(nb)

    for d in datasets:
        datasets_found.add(d)


print("\nDatasets detected:")
for d in datasets_found:
    print("-", d)


print("\nStarting downloads")

for dataset in datasets_found:
    download_dataset(dataset, save_folder)


print("\nAll downloads finished.")
