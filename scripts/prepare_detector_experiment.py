"""Build a synthetic detector development dataset with exact stock labels."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from app.evaluation.crop_fixtures import _page, _photograph
from prepare_yolo_dataset import write_data_yaml


_SPLITS = {"train": (731253, 96), "val": (831253, 24), "challenge": (931253, 24)}
_PHOTOGRAPH_CASES = (
    "white_margin", "white_on_white", "white_motif", "cast_shadow", "glare", "camera_tilt",
    "touching", "overlap", "photo_corners", "clipped", "curl", "borderless", "portrait",
)


def _scene(rng: np.random.Generator, count: int, shared_stock: bool) -> tuple[np.ndarray, list[np.ndarray]]:
    width, height = 672, 520
    size = (width, height)
    paper = rng.uniform(115, 244)
    grain = cv2.resize(rng.normal(0, 4, (13, 15)).astype(np.float32), size)
    tint = rng.normal(0, 7, 3)
    page = np.clip((paper + grain)[:, :, None] + tint, 0, 255).astype(np.uint8)
    spine = int(rng.integers(15, 48))
    page[:, :spine] = np.rint(page[:, :spine] * rng.uniform(.56, .86)).astype(np.uint8)

    panel_width = int(rng.integers(104, 146))
    panel_height = int(rng.integers(124, 218))
    tone = int(rng.uniform(192, 252))
    margin = int(rng.integers(2, 12))
    gap = int(rng.choice([1, 2, 3, 6, 12]))
    run_width = count * panel_width + (count - 1) * gap
    start_x = int(rng.integers(spine + 6, max(spine + 7, width - run_width - 14)))
    start_y = int(rng.integers(56, height - panel_height - 62))
    stock = np.full((panel_height, run_width, 3), tone, dtype=np.uint8)
    owner = np.zeros((height, width), np.uint8)
    shift = np.float32([[1, 0, 3], [0, 1, 4]])
    shadow_alpha = rng.uniform(.06, .36)

    for index in range(count):
        left = index * (panel_width + gap)
        ink = _photograph(panel_height - 2 * margin, panel_width - 2 * margin, rng, white=bool(index % 2))
        gain = rng.uniform(.73, 1.24, 3)
        bias = rng.uniform(-21, 21, 3)
        if rng.random() < .33:
            ink = cv2.cvtColor(cv2.cvtColor(ink, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        stock[margin:-margin, left + margin:left + panel_width - margin] = np.clip(ink * gain + bias, 0, 255)

    for index in range(1 if shared_stock else count):
        left = index * (panel_width + gap)
        tile = stock if shared_stock else stock[:, left:left + panel_width].copy()
        depth = 0 if shared_stock else int(rng.integers(-15, 16))
        x, y = start_x + left, start_y + depth
        mask = np.zeros_like(owner)
        mask[y:y + panel_height, x:x + tile.shape[1]] = 1
        shadow = cv2.warpAffine(mask.astype(np.float32), shift, size)
        shading = 1 - shadow_alpha * cv2.GaussianBlur(shadow, (0, 0), 1.8)
        page = np.rint(page * shading[:, :, None]).astype(np.uint8)
        page[y:y + panel_height, x:x + tile.shape[1]] = tile
        owner[mask != 0] = index + 1

    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(source, source + rng.uniform(-24, 24, (4, 2)).astype(np.float32))
    page = cv2.warpPerspective(page, transform, size, borderMode=cv2.BORDER_REPLICATE)
    owner = cv2.warpPerspective(owner, transform, size, flags=cv2.INTER_NEAREST)
    if rng.random() < .45:
        page = cv2.GaussianBlur(page, (0, 0), rng.uniform(.4, 1.14))
    contours = []
    for index in range(1, int(owner.max()) + 1):
        choices, _ = cv2.findContours((owner == index).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(choices, key=cv2.contourArea)
        outline = cv2.approxPolyDP(contour, .72, True).reshape(-1, 2)
        contours.append(outline / [width - 1, height - 1])
    return page, contours


def prepare(root: Path) -> None:
    cv2.setNumThreads(2)
    sources = []
    for split, (first_seed, count) in _SPLITS.items():
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True, exist_ok=False)
        labels.mkdir(parents=True, exist_ok=False)
        for index in range(count):
            seed = first_seed + index
            rng = np.random.default_rng(seed)
            if index % 3 == 2 and split != "challenge":
                case = _PHOTOGRAPH_CASES[index // 3 % len(_PHOTOGRAPH_CASES)]
                fixture = _page(case, rng)
                image, polygons = fixture.image, [photo.visible_polygon for photo in fixture.prints]
            else:
                shared = index % 3 == 0 if split != "challenge" else index % 2 == 0
                case = "shared_multipanel_stock" if shared else "neighboring_prints"
                image, polygons = _scene(rng, int(rng.integers(2, 5)), shared)

            name = f"synthetic-stock-{seed}"
            encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(82, 99))])[1].tobytes()
            outline = "".join(
                "0 " + " ".join(f"{x:.8f} {y:.8f}" for x, y in polygon) + "\n"
                for polygon in polygons
            ).encode("ascii")
            (images / f"{name}.jpg").write_bytes(encoded)
            (labels / f"{name}.txt").write_bytes(outline)
            sources.append({
                "image": f"images/{split}/{name}.jpg",
                "label": f"labels/{split}/{name}.txt",
                "image_sha256": sha256(encoded).hexdigest(),
                "label_sha256": sha256(outline).hexdigest(),
                "source_group": name,
                "source": "procedural stock fixture",
                "split": split,
                "case": case,
                "seed": seed,
                "prints": len(polygons),
                "annotation": "generated_fixture",
            })
    write_data_yaml(root / "data.yaml", root)
    (root / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.destination.resolve())


if __name__ == "__main__":
    main()
