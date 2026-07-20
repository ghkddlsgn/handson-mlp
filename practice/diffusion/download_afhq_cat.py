from pathlib import Path
import shutil
import kaggle

DATASETS_DIR = Path("datasets")
RAW_DIR = DATASETS_DIR / "animal-faces-raw"
CAT_DIR = DATASETS_DIR / "afhq_cat"

kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    "andrewmvd/animal-faces",
    path=str(RAW_DIR),
    unzip=True
)

for split in ["train", "val"]:
    src = RAW_DIR / "afhq" / split / "cat"

    if not src.exists():
        continue

    dst = CAT_DIR / split / "cat"
    dst.mkdir(parents=True, exist_ok=True)

    for img in src.glob("*.jpg"):
        shutil.copy(img, dst / img.name)

n_train = len(list((CAT_DIR / "train" / "cat").glob("*.jpg")))
n_val = len(list((CAT_DIR / "val" / "cat").glob("*.jpg")))

print(f"cat images -> train: {n_train}, val: {n_val}")
print(f"saved to: {CAT_DIR.resolve()}")
