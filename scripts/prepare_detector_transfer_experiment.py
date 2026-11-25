"""Rebuild a paired development split using the original legacy camera pages."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from app.pipeline.image_ops import load_image
from prepare_yolo_dataset import write_data_yaml
from validate_yolo_dataset import parse_yolo_segmentation_line


_ROOT = Path(__file__).resolve().parents[1]
_IMAGE_COMMIT = "99847d97be19d729ea5234a03673449488cd6b95"
_LABEL_COMMIT = "46e94be8a33c6e4d84a0b4879f9c460524157c27"
_STOCK_INVENTORY_SHA256 = "819bce5454be794fd8fdfa6fa72e28d059405f6ca500ebe53ab27704a87e9d73"
_GROUP = "legacy_album_context_01"
_ARCHIVE_PAGES = {
    "train": ("page_03", "page_23", "page_27", "page_30", "page_38"),
    "val": ("page_19", "page_22"),
}
_GAMMAS = (.92, 1.08)


def _git_file(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=_ROOT)


def _verified_file(root: Path, record: dict, key: str) -> bytes:
    path = (root / record[key]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Source file is outside the synthetic dataset: {record[key]}")
    contents = path.read_bytes()
    if sha256(contents).hexdigest() != record[f"{key}_sha256"]:
        raise ValueError(f"Source hash differs from inventory: {record[key]}")
    return contents


def _write_case(root: Path, name: str, split: str, image: bytes, label: bytes, provenance: dict) -> dict:
    image_path, label_path = f"images/{split}/{name}.jpg", f"labels/{split}/{name}.txt"
    (root / image_path).write_bytes(image)
    (root / label_path).write_bytes(label)
    return {
        **provenance,
        "split": split,
        "image": image_path,
        "image_sha256": sha256(image).hexdigest(),
        "label": label_path,
        "label_sha256": sha256(label).hexdigest(),
    }


def _transformed_label(polygons: list[np.ndarray], turns: int) -> bytes:
    outlines = []
    for polygon in polygons:
        x, y = polygon.T
        coordinates = ((x, y), (y, 1 - x), (1 - x, 1 - y), (1 - y, x))[turns]
        values = np.column_stack(coordinates).ravel()
        outlines.append("0 " + " ".join(f"{value:.8f}" for value in values))
    return ("\n".join(outlines) + "\n").encode("ascii")


def prepare(stock_root: Path, destination: Path) -> None:
    cv2.setNumThreads(2)
    inventory = (stock_root / "sources.json").read_bytes()
    if sha256(inventory).hexdigest() != _STOCK_INVENTORY_SHA256:
        raise ValueError("Use the pinned input inventory from the stock detector experiment")
    stock = json.loads(inventory)
    stock_training = sorted((item for item in stock if item["split"] == "train"), key=lambda item: item["image"])
    stock_validation = [item for item in stock if item["split"] == "val"]
    if len(stock_training) != 96 or len(stock_validation) != 24:
        raise ValueError("Use the original 96/24 synthetic stock experiment inputs")
    archive_split = json.loads(_git_file(_LABEL_COMMIT, "data/yolo_split_manifest.json"))
    if any(page not in archive_split[split] or page in archive_split["golden"]
           for split, pages in _ARCHIVE_PAGES.items() for page in pages):
        raise ValueError("Archived source differs from the legacy train/validation split")

    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "native"
    archive.mkdir()
    real_training = []
    real_validation = []
    for split, pages in _ARCHIVE_PAGES.items():
        for page in pages:
            image_source = f"data/golden_fixtures/{page}.jpg"
            label_source = f"data/yolo_dataset/labels/{split}/{page}.txt"
            image, label = _git_file(_IMAGE_COMMIT, image_source), _git_file(_LABEL_COMMIT, label_source)
            native_path = archive / f"{page}.jpg"
            native_path.write_bytes(image)
            native = load_image(native_path)
            polygons = [np.array(parse_yolo_segmentation_line(line, class_count=1))
                        for line in label.decode().splitlines() if line.strip()]
            receipt = {
                "source_group": _GROUP,
                "source": f"{_IMAGE_COMMIT}:{image_source}",
                "source_label": f"{_LABEL_COMMIT}:{label_source}",
                "source_image_sha256": sha256(image).hexdigest(),
                "source_label_sha256": sha256(label).hexdigest(),
                "native_size": list(native.shape[1::-1]),
                "prints": len(polygons),
                "annotation": "legacy_development_label",
            }
            if split == "val":
                real_validation.append((f"legacy-{page}", image, label, receipt))
                continue
            for turns in range(4):
                rotated = np.rot90(native, turns)
                for index, gamma in enumerate(_GAMMAS):
                    curve = np.rint(255 * (np.arange(256, dtype=np.float64) / 255) ** gamma).astype(np.uint8)
                    adjusted = cv2.LUT(rotated, curve)
                    encoded = cv2.imencode(".jpg", adjusted, [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tobytes()
                    real_training.append((
                        f"legacy-{page}-r{turns}-g{index}", encoded, _transformed_label(polygons, turns),
                        {**receipt, "rotation_ccw_quarters": turns, "gamma": gamma, "jpeg_quality": 94},
                    ))

    replaced = set(np.random.default_rng(716250).choice(len(stock_training), len(real_training), replace=False))
    for cohort in ("synthetic-control", "mixed-legacy"):
        root = destination / cohort
        for folder in ("images/train", "images/val", "labels/train", "labels/val"):
            (root / folder).mkdir(parents=True)
        inventory = []
        for index, item in enumerate([*stock_training, *stock_validation]):
            if cohort == "mixed-legacy" and index in replaced:
                continue
            image, label = (_verified_file(stock_root, item, key) for key in ("image", "label"))
            inventory.append(_write_case(root, Path(item["image"]).stem, item["split"], image, label, item))
        for split, cases in (("train", real_training if cohort == "mixed-legacy" else []), ("val", real_validation)):
            for name, image, label, receipt in cases:
                inventory.append(_write_case(root, name, split, image, label, receipt))
        write_data_yaml(root / "data.yaml", root)
        (root / "sources.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stock_data", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.stock_data.resolve(), args.destination.resolve())


if __name__ == "__main__":
    main()
