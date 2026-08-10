from pathlib import Path

import kagglehub


DATASET = "awsaf49/coco-2017-dataset"
OUTPUT_DIR = Path("datasets/coco")


def main() -> None:
    path = kagglehub.dataset_download(
        DATASET,
        output_dir=str(OUTPUT_DIR),
    )
    print(f"COCO 2017 downloaded to: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
