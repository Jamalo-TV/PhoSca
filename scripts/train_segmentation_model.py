from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_yolo_dataset import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export PhoSca's one-class photo segmentation model.")
    parser.add_argument("--data", type=Path, default=Path("data/data.yaml"), help="Ultralytics dataset YAML with polygon labels.")
    parser.add_argument("--base-model", default="yolo11n-seg.pt", help="YOLO segmentation checkpoint to fine-tune.")
    parser.add_argument("--project", type=Path, default=Path("runs/phosca-segmentation"), help="Training output directory.")
    parser.add_argument("--name", default="photo-seg", help="Run name inside --project.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. 0, cpu, or mps.")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--export", type=Path, default=Path("models/yolov8-seg-album.onnx"), help="Destination ONNX path.")
    parser.add_argument("--skip-train", action="store_true", help="Export an existing --weights checkpoint instead of training.")
    parser.add_argument("--weights", type=Path, default=None, help="Existing .pt checkpoint for --skip-train exports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Dataset YAML not found: {args.data}")
    dataset_report = validate_dataset(args.data)
    if not dataset_report.valid:
        raise SystemExit("YOLO dataset is not trainable yet. Run scripts/validate_yolo_dataset.py for details.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required for training. Install it with: pip install ultralytics") from exc

    if args.skip_train:
        if args.weights is None or not args.weights.exists():
            raise SystemExit("--skip-train requires --weights pointing to an existing .pt checkpoint")
        model = YOLO(str(args.weights))
        train_result = {"skipped": True, "weights": str(args.weights)}
        export_model = model
    else:
        model = YOLO(args.base_model)
        train_result = model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=str(args.project),
            name=args.name,
            patience=args.patience,
            device=args.device,
            task="segment",
        )
        best_weights = args.project / args.name / "weights" / "best.pt"
        export_model = YOLO(str(best_weights)) if best_weights.exists() else model

    args.export.parent.mkdir(parents=True, exist_ok=True)
    export_result = export_model.export(format="onnx", imgsz=args.imgsz, dynamic=True, simplify=True)
    exported = Path(str(export_result))
    if exported.resolve() != args.export.resolve():
        args.export.write_bytes(exported.read_bytes())

    report = {
        "data": str(args.data),
        "base_model": args.base_model,
        "training": str(train_result),
        "export_weights": str(getattr(export_model, "ckpt_path", "") or args.weights or "in_memory_model"),
        "onnx_export": str(args.export),
        "imgsz": args.imgsz,
    }
    report_path = args.export.with_suffix(".training.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
