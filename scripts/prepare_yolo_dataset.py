from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class SplitManifest:
    source_images: str
    source_labels: str | None
    output_dataset: str
    golden_dir: str
    train: list[str]
    val: list[str]
    golden: list[str]
    missing_labels: list[str]


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_stem_list(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [Path(line.strip()).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]


def choose_split_stems(images: list[Path], *, seed: int, val_count: int, golden_count: int, val_list: Path | None, golden_list: Path | None) -> tuple[list[str], list[str], list[str]]:
    stems = [path.stem for path in images]
    stem_set = set(stems)
    golden = read_stem_list(golden_list)
    val = read_stem_list(val_list)
    unknown = sorted((set(golden) | set(val)) - stem_set)
    if unknown:
        raise SystemExit(f"Split list references unknown images: {', '.join(unknown)}")
    overlap = sorted(set(golden) & set(val))
    if overlap:
        raise SystemExit(f"Images cannot be both validation and golden: {', '.join(overlap)}")

    remaining = [stem for stem in stems if stem not in set(golden) and stem not in set(val)]
    rng = random.Random(seed)
    shuffled = list(remaining)
    rng.shuffle(shuffled)
    if not golden:
        golden = sorted(shuffled[:golden_count])
        shuffled = shuffled[golden_count:]
    else:
        shuffled = [stem for stem in shuffled if stem not in set(golden)]
    if not val:
        val = sorted(shuffled[:val_count])
        shuffled = shuffled[val_count:]
    else:
        shuffled = [stem for stem in shuffled if stem not in set(val)]
    train = sorted(stem for stem in stems if stem not in set(golden) and stem not in set(val))
    return train, sorted(val), sorted(golden)


def clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_split(images_by_stem: dict[str, Path], labels_by_stem: dict[str, Path], stems: list[str], *, image_dir: Path, label_dir: Path | None) -> list[str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    if label_dir is not None:
        label_dir.mkdir(parents=True, exist_ok=True)
    missing_labels: list[str] = []
    for stem in stems:
        source_image = images_by_stem[stem]
        shutil.copy2(source_image, image_dir / source_image.name)
        label = labels_by_stem.get(stem)
        if label is not None and label_dir is not None:
            shutil.copy2(label, label_dir / f"{stem}.txt")
        elif label_dir is not None:
            missing_labels.append(stem)
    return missing_labels


def write_data_yaml(path: Path, dataset_dir: Path) -> None:
    path.write_text(
        f"path: {dataset_dir.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['photo']\n",
        encoding="utf-8",
    )


def prepare_dataset(
    *,
    source_images: Path,
    source_labels: Path | None,
    output_dataset: Path,
    golden_dir: Path,
    seed: int,
    val_count: int,
    golden_count: int,
    val_list: Path | None = None,
    golden_list: Path | None = None,
    require_labels: bool = False,
) -> SplitManifest:
    images = image_files(source_images)
    if not images:
        raise SystemExit(f"No images found in {source_images}")
    labels = sorted(source_labels.glob("*.txt")) if source_labels is not None and source_labels.exists() else []
    images_by_stem = {path.stem: path for path in images}
    labels_by_stem = {path.stem: path for path in labels}
    train, val, golden = choose_split_stems(
        images,
        seed=seed,
        val_count=val_count,
        golden_count=golden_count,
        val_list=val_list,
        golden_list=golden_list,
    )

    clean_directory(output_dataset / "images" / "train")
    clean_directory(output_dataset / "images" / "val")
    clean_directory(output_dataset / "labels" / "train")
    clean_directory(output_dataset / "labels" / "val")
    golden_dir.mkdir(parents=True, exist_ok=True)
    for existing in golden_dir.glob("*.jpg"):
        existing.unlink()
    for existing in golden_dir.glob("*.jpeg"):
        existing.unlink()
    for existing in golden_dir.glob("*.png"):
        existing.unlink()
    for existing in golden_dir.glob("*.webp"):
        existing.unlink()

    missing_labels: list[str] = []
    missing_labels.extend(copy_split(images_by_stem, labels_by_stem, train, image_dir=output_dataset / "images" / "train", label_dir=output_dataset / "labels" / "train"))
    missing_labels.extend(copy_split(images_by_stem, labels_by_stem, val, image_dir=output_dataset / "images" / "val", label_dir=output_dataset / "labels" / "val"))
    copy_split(images_by_stem, labels_by_stem, golden, image_dir=golden_dir, label_dir=golden_dir / "labels" if source_labels is not None else None)
    if require_labels and missing_labels:
        raise SystemExit(f"Missing train/val labels: {', '.join(sorted(missing_labels))}")

    data_yaml = output_dataset.parent / "data.yaml"
    write_data_yaml(data_yaml, output_dataset)
    manifest = SplitManifest(
        source_images=str(source_images),
        source_labels=str(source_labels) if source_labels else None,
        output_dataset=str(output_dataset),
        golden_dir=str(golden_dir),
        train=train,
        val=val,
        golden=golden,
        missing_labels=sorted(missing_labels),
    )
    manifest_path = output_dataset.parent / "yolo_split_manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic train/val/golden YOLO segmentation splits for PhoSca.")
    parser.add_argument("--source-images", type=Path, default=Path("data/raw_album_pages"))
    parser.add_argument("--source-labels", type=Path, default=None, help="Directory containing YOLO .txt labels keyed by image stem.")
    parser.add_argument("--output-dataset", type=Path, default=Path("data/yolo_dataset"))
    parser.add_argument("--golden-dir", type=Path, default=Path("data/golden_fixtures"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-count", type=int, default=5)
    parser.add_argument("--golden-count", type=int, default=10)
    parser.add_argument("--val-list", type=Path, default=None, help="Optional newline-delimited image stems for the validation split.")
    parser.add_argument("--golden-list", type=Path, default=None, help="Optional newline-delimited image stems for locked golden fixtures.")
    parser.add_argument("--require-labels", action="store_true")
    args = parser.parse_args()

    manifest = prepare_dataset(
        source_images=args.source_images,
        source_labels=args.source_labels,
        output_dataset=args.output_dataset,
        golden_dir=args.golden_dir,
        seed=args.seed,
        val_count=args.val_count,
        golden_count=args.golden_count,
        val_list=args.val_list,
        golden_list=args.golden_list,
        require_labels=args.require_labels,
    )
    print(json.dumps(asdict(manifest), indent=2))


if __name__ == "__main__":
    main()
