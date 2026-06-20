from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def blur_score(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=Path("data/blur_scores.csv"))
    args = parser.parse_args()

    images = sorted(
        path for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "blur_score", "status"])
        writer.writeheader()
        for image in images:
            score = blur_score(image)
            writer.writerow(
                {
                    "filename": image.name,
                    "blur_score": f"{score:.4f}",
                    "status": "keep" if score >= args.threshold else "discard",
                }
            )


if __name__ == "__main__":
    main()

