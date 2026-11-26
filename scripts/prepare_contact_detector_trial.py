"""Prepare matched development cohorts for dense contact-print artwork."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from app.evaluation.crop_fixtures import _photograph
from prepare_detector_transfer_experiment import _verified_file, _write_case
from prepare_yolo_dataset import write_data_yaml


_TRANSFER_INVENTORY_SHA256 = "31b7db998453c763a867aeb97d018d981fe4804f273f1367d298a94aed8b79ae"
_ARMS = ("sparse-control", "dense-contact")
_SPLITS = {"train": (1848100, 16), "val": (1849100, 4)}


def _contact_art(width: int, height: int, seed: int, rows: int, columns: int) -> np.ndarray:
    """Render contrast, film perforations and regular image subdivisions on physical stock."""
    rng = np.random.default_rng(seed)
    margin_x, margin_y = round(width * .055), round(height * .071)
    source = _photograph(height, width, rng, white=False)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    canvas = np.full((height, width), int(rng.integers(198, 248)), np.uint8)
    right, bottom = width - margin_x, height - margin_y // 2
    xs = np.rint(np.linspace(margin_x, right, columns + 1)).astype(int)
    ys = np.rint(np.linspace(margin_y, bottom, rows + 1)).astype(int)
    rail = max(3, round((ys[1] - ys[0]) * .10))
    for y1, y2 in zip(ys[:-1], ys[1:], strict=True):
        canvas[y1:y2 - 2, margin_x:right] = int(rng.integers(24, 56))
        for x1, x2 in zip(xs[:-1], xs[1:], strict=True):
            panel = gray[y1 + rail:y2 - rail - 2, x1 + 1:x2 - 1]
            canvas[y1 + rail:y2 - rail - 2, x1 + 1:x2 - 1] = cv2.convertScaleAbs(
                panel, alpha=rng.uniform(.70, 1.15), beta=rng.uniform(0, 22),
            )
        for x in range(margin_x + 3, right - 3, max(5, width // 37)):
            canvas[y1 + 1:y1 + rail - 1, x:x + 2] = 211
            canvas[y2 - rail:y2 - 3, x:x + 2] = 211
    cv2.putText(canvas, f"{seed % 10000:04d}", (margin_x, margin_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                max(.35, width / 540), 67, 1, cv2.LINE_AA)
    return canvas


def _render(seed: int, separated: bool, dense: bool) -> tuple[np.ndarray, bytes, dict]:
    rng = np.random.default_rng(seed)
    width, height = 720, 560
    size = (width, height)
    noise = cv2.resize(rng.normal(0, 6, (22, 34)).astype(np.float32), size)
    grain = np.sin(np.arange(width) * rng.uniform(.17, .32)) * rng.uniform(0, 5)
    background = np.clip((rng.uniform(128, 223) + noise + grain)[:, :, None] + rng.uniform(-16, 16, 3),
                         0, 255).astype(np.uint8)
    paper_width, paper_height = int(rng.integers(247, 304)), int(rng.integers(362, 448))
    x, y = int(rng.integers(387, 407)), int(rng.integers(44, 69))
    sleeve_width = int(rng.integers(218, 279))
    sleeve = _contact_art(sleeve_width, paper_height, seed + 31, 7, 5)
    sleeve_row, sleeve_column = y + int(rng.integers(-8, 14)), int(rng.integers(44, 96))
    backing = background[sleeve_row:sleeve_row + paper_height, sleeve_column:sleeve_column + sleeve_width]
    transmission = (.30 + .62 * sleeve.astype(np.float32) / 255)[:, :, None]
    backing[:] = np.rint(backing * transmission * [.84, .91, 1.]).astype(np.uint8)

    rows, columns = (7, 5) if dense or separated else (2, 2)
    stock = _contact_art(paper_width, paper_height, seed + 23, rows, columns)
    paper = cv2.cvtColor(stock, cv2.COLOR_GRAY2BGR)
    owners = np.zeros((height, width), np.uint8)
    gap = int(rng.choice([1, 2, 3, 6]))
    header = round(.071 * paper_height)
    film_height = paper_height - header - header // 2
    split = header + round(film_height * int(rng.choice([3, 4])) / 7)
    spans = ((0, split), (split + gap, paper_height)) if separated else ((0, paper_height),)
    for owner, (start, stop) in enumerate(spans, 1):
        offset = 0 if owner == 1 else int(rng.integers(-3, 4))
        mask = np.zeros_like(owners)
        block = np.s_[y + start:y + stop, x + offset:x + offset + paper_width]
        mask[block] = 1
        shade = cv2.GaussianBlur(cv2.warpAffine(mask.astype(np.float32),
                                                np.float32([[1, 0, 3], [0, 1, 3]]), size), (0, 0), 1.6)
        background[:] = np.rint(background * (1 - .19 * shade[:, :, None])).astype(np.uint8)
        material = paper[start:stop].copy()
        if separated:
            material[-1 if owner == 1 else 0] = 219
        background[block] = material
        owners[block] = owner

    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    projection = cv2.getPerspectiveTransform(corners, corners + rng.uniform(-29, 29, (4, 2)).astype(np.float32))
    image = cv2.warpPerspective(background, projection, size, borderMode=cv2.BORDER_REPLICATE)
    owners = cv2.warpPerspective(owners, projection, size, flags=cv2.INTER_NEAREST)
    if rng.random() < .55:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(.42, 1.1))
    turns = int(rng.integers(0, 4))
    image, owners = np.rot90(image, turns), np.rot90(owners, turns)
    label = []
    for owner in range(1, len(spans) + 1):
        contours, _ = cv2.findContours(np.uint8(owners == owner), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = cv2.approxPolyDP(max(contours, key=cv2.contourArea), .8, True).reshape(-1, 2)
        points = points / (np.asarray(image.shape[1::-1]) - 1)
        label.append("0 " + " ".join(f"{value:.8f}" for value in points.ravel()) + "\n")
    return image, "".join(label).encode("ascii"), {"jpeg_quality": int(rng.choice([65, 75, 84, 96])),
                                                   "rotation_ccw_quarters": turns}


def prepare(transfer_root: Path, destination: Path) -> None:
    cv2.setNumThreads(1)
    inventory_data = (transfer_root / "sources.json").read_bytes()
    if sha256(inventory_data).hexdigest() != _TRANSFER_INVENTORY_SHA256:
        raise ValueError("Use the pinned mixed-legacy transfer inventory")
    inventory = json.loads(inventory_data)
    stock = [row for row in inventory if row["split"] == "train" and row["annotation"] == "generated_fixture"]
    if len(stock) != 56 or sum(row["split"] == "train" for row in inventory) != 96:
        raise ValueError("The matched cohorts require the original 96 training images")
    removed = set(np.random.default_rng(1848).choice(len(stock), 32, replace=False))
    removed = {stock[index]["image"] for index in removed}
    retained = [row for row in inventory if row["image"] not in removed]
    destination.mkdir(parents=True, exist_ok=False)

    for arm in _ARMS:
        root = destination / arm
        for location in ("images/train", "images/val", "labels/train", "labels/val"):
            (root / location).mkdir(parents=True)
        sources = []
        for row in retained:
            image, label = (_verified_file(transfer_root, row, key) for key in ("image", "label"))
            sources.append(_write_case(root, Path(row["image"]).stem, row["split"], image, label, row))
        for split, (start, pair_count) in _SPLITS.items():
            for seed in range(start, start + pair_count):
                for separated in (False, True):
                    dense = arm == "dense-contact" or split == "val" or separated
                    image, label, parameters = _render(seed, separated, dense)
                    encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, parameters["jpeg_quality"]])[1]
                    name = f"contact-{seed}-{'separate' if separated else 'joined'}"
                    provenance = {**parameters, "source_group": f"contact-layout-{seed}", "seed": seed,
                                  "source": "procedural contact stock fixture",
                                  "case": "separate_prints" if separated else "shared_multipanel_stock",
                                  "artwork": "dense" if dense else "sparse", "prints": 2 if separated else 1,
                                  "annotation": "generated_fixture"}
                    sources.append(_write_case(root, name, split, encoded.tobytes(), label, provenance))
        write_data_yaml(root / "data.yaml", root)
        (root / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_legacy_data", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.mixed_legacy_data.resolve(), args.destination.resolve())


if __name__ == "__main__":
    main()
