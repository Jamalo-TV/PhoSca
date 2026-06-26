from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from export_label_studio_tasks import export_tasks
from prepare_yolo_dataset import prepare_dataset
from validate_yolo_dataset import validate_dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_WORKSPACE = Path("data/segmentation_training")


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def label_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.txt") if path.is_file())


def ensure_workspace(workspace: Path) -> dict[str, Path]:
    paths = {
        "images": workspace / "images",
        "labels": workspace / "labels",
        "exports": workspace / "exports",
        "tasks": workspace / "label_studio_tasks.json",
    }
    for key in ("images", "labels", "exports"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def seed_images(source: Path, destination: Path, *, limit: int = 0, overwrite: bool = False) -> list[str]:
    copied: list[str] = []
    images = image_files(source)
    if limit > 0:
        images = images[:limit]
    for image in images:
        target = destination / image.name
        if target.exists() and not overwrite:
            continue
        shutil.copy2(image, target)
        copied.append(image.name)
    return copied


def build_label_studio_tasks(
    *,
    images_dir: Path,
    output: Path,
    document_root: Path,
    preannotate: bool,
    min_confidence: float,
) -> int:
    tasks = export_tasks(
        source_images=images_dir,
        output=output,
        document_root=document_root,
        url_prefix="/data/local-files/?d=",
        preannotate=preannotate,
        min_confidence=min_confidence,
    )
    return len(tasks)


def workspace_status(workspace: Path) -> dict:
    paths = ensure_workspace(workspace)
    return {
        "workspace": str(workspace),
        "images_dir": str(paths["images"]),
        "labels_dir": str(paths["labels"]),
        "exports_dir": str(paths["exports"]),
        "label_studio_tasks": str(paths["tasks"]),
        "images": len(image_files(paths["images"])),
        "labels": len(label_files(paths["labels"])),
    }


def prepare_workspace_dataset(args: argparse.Namespace) -> dict:
    paths = ensure_workspace(args.workspace)
    labels = label_files(paths["labels"])
    if not labels:
        raise SystemExit(
            f"No YOLO labels found in {paths['labels']}. "
            "Export reviewed Label Studio JSON, then convert it with scripts/convert_label_studio_to_yolo.py."
        )

    manifest = prepare_dataset(
        source_images=paths["images"],
        source_labels=paths["labels"],
        output_dataset=args.output_dataset,
        golden_dir=args.golden_dir,
        seed=args.seed,
        val_count=args.val_count,
        golden_count=args.golden_count,
        val_list=args.val_list,
        golden_list=args.golden_list,
        require_labels=True,
    )
    report = validate_dataset(args.data_yaml, root=args.output_dataset, golden_dir=args.golden_dir)
    return {
        "manifest": asdict(manifest),
        "validation": report.as_dict(),
    }


def train_workspace_model(args: argparse.Namespace) -> dict:
    prepared = prepare_workspace_dataset(args)
    if not prepared["validation"]["valid"]:
        raise SystemExit("YOLO dataset validation failed; fix labels before training.")

    command = [
        sys.executable,
        str(Path(__file__).with_name("train_segmentation_model.py")),
        "--data",
        str(args.data_yaml),
        "--base-model",
        args.base_model,
        "--project",
        str(args.project),
        "--name",
        args.name,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--patience",
        str(args.patience),
        "--export",
        str(args.export),
    ]
    if args.device:
        command.extend(["--device", args.device])

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return {
        "prepared": prepared,
        "train_command": command,
        "export": str(args.export),
    }


def bootstrap_workspace(args: argparse.Namespace) -> dict:
    paths = ensure_workspace(args.workspace)
    copied: list[str] = []
    existing_images = image_files(paths["images"])
    if args.copy_examples and (args.overwrite or not existing_images):
        copied = seed_images(args.seed_images, paths["images"], limit=args.limit, overwrite=args.overwrite)

    images = image_files(paths["images"])
    if not images:
        raise SystemExit(f"No images found in {paths['images']}. Add album page images there and rerun this command.")

    task_count = build_label_studio_tasks(
        images_dir=paths["images"],
        output=paths["tasks"],
        document_root=args.document_root,
        preannotate=args.preannotate,
        min_confidence=args.min_confidence,
    )
    return {
        **workspace_status(args.workspace),
        "copied_examples": copied,
        "tasks": task_count,
        "next_commands": [
            "docker run -it -p 8080:8080 -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data -v ${PWD}/data:/label-studio/data heartexlabs/label-studio:latest",
            f"python scripts/convert_label_studio_to_yolo.py --input {paths['exports'] / 'label_studio_export.json'} --output {paths['labels']}",
            "python scripts/setup_segmentation_training.py train",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap, prepare, and train PhoSca's one-class YOLO photo segmentation workspace."
    )
    parser.add_argument("command", nargs="?", choices=["bootstrap", "status", "prepare", "train"], default="bootstrap")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--seed-images", type=Path, default=Path("data/raw_album_pages"))
    parser.add_argument("--copy-examples", action=argparse.BooleanOptionalAction, default=True, help="Seed examples when images/ is empty.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images when copying examples.")
    parser.add_argument("--limit", type=int, default=0, help="Limit copied seed images; 0 copies all available examples.")
    parser.add_argument("--document-root", type=Path, default=Path("data"))
    parser.add_argument("--preannotate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--output-dataset", type=Path, default=Path("data/yolo_dataset"))
    parser.add_argument("--golden-dir", type=Path, default=Path("data/golden_fixtures"))
    parser.add_argument("--data-yaml", type=Path, default=Path("data/data.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-count", type=int, default=5)
    parser.add_argument("--golden-count", type=int, default=10)
    parser.add_argument("--val-list", type=Path, default=None)
    parser.add_argument("--golden-list", type=Path, default=None)
    parser.add_argument("--base-model", default="yolo11n-seg.pt")
    parser.add_argument("--project", type=Path, default=Path("runs/phosca-segmentation"))
    parser.add_argument("--name", default="photo-seg")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--export", type=Path, default=Path("models/yolov8-seg-album.onnx"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "status":
        payload = workspace_status(args.workspace)
    elif args.command == "prepare":
        payload = prepare_workspace_dataset(args)
    elif args.command == "train":
        payload = train_workspace_model(args)
    else:
        payload = bootstrap_workspace(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
