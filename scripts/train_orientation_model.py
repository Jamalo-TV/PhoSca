from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROTATIONS = (0, 90, 180, 270)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    normalized = angle % 360
    if normalized == 0:
        return image
    if normalized == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported right-angle rotation: {angle}")


@dataclass(frozen=True)
class Split:
    train: list[Path]
    val: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export PhoSca's 0/90/180/270 photo orientation classifier.")
    parser.add_argument("--images", type=Path, default=Path("data/orientation_photos"), help="Directory of manually upright photo crops.")
    parser.add_argument("--output", type=Path, default=Path("models/photo-orientation.onnx"), help="Destination ONNX file.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--min-images", type=int, default=8, help="Minimum upright source images required before training.")
    return parser.parse_args()


def find_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file())


def split_images(paths: list[Path], *, val_ratio: float, seed: int) -> Split:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    return Split(train=shuffled[val_count:], val=shuffled[:val_count])


def load_tensor(path: Path, *, applied_rotation: int, image_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    image = rotate_image(image, applied_rotation)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0


def correction_label(applied_rotation: int) -> int:
    correction = (-applied_rotation) % 360
    return ROTATIONS.index(correction)


def build_model():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required for orientation training. Install torch in the training environment.") from exc

    class OrientationNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 160, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(160),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(160, 4)

        def forward(self, x):
            return self.classifier(self.features(x).flatten(1))

    return torch, nn, OrientationNet()


def batch_iterator(paths: list[Path], *, image_size: int, batch_size: int, shuffle: bool, seed: int):
    import torch

    samples = [(path, rotation) for path in paths for rotation in ROTATIONS]
    if shuffle:
        random.Random(seed).shuffle(samples)
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        tensors = [load_tensor(path, applied_rotation=rotation, image_size=image_size) for path, rotation in chunk]
        labels = [correction_label(rotation) for _, rotation in chunk]
        yield torch.from_numpy(np.stack(tensors)), torch.tensor(labels, dtype=torch.long)


def evaluate(model, paths: list[Path], *, image_size: int, batch_size: int, device: str) -> float:
    import torch

    if not paths:
        return 0.0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in batch_iterator(paths, image_size=image_size, batch_size=batch_size, shuffle=False, seed=0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = model(inputs).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return correct / total if total else 0.0


def main() -> None:
    args = parse_args()
    paths = find_images(args.images)
    if len(paths) < args.min_images:
        raise SystemExit(
            f"Need at least {args.min_images} manually upright images under {args.images}; found {len(paths)}. "
            "Use reviewed/corrected photo crops, not raw album pages."
        )

    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise SystemExit("The onnx package is required for export. Install it with: pip install onnx") from exc

    torch, nn, model = build_model()
    torch.manual_seed(args.seed)
    split = split_images(paths, val_ratio=args.val_ratio, seed=args.seed)
    device = args.device
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for inputs, labels in batch_iterator(split.train, image_size=args.imgsz, batch_size=args.batch, shuffle=True, seed=args.seed + epoch):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            total_items += int(labels.numel())
        val_accuracy = evaluate(model, split.val, image_size=args.imgsz, batch_size=args.batch, device=device)
        train_loss = total_loss / total_items if total_items else 0.0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_accuracy})
        print(json.dumps(history[-1]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["image"],
        output_names=["rotation_logits"],
        dynamic_axes={"image": {0: "batch"}, "rotation_logits": {0: "batch"}},
        opset_version=17,
    )
    report = {
        "source_images": len(paths),
        "train_images": len(split.train),
        "val_images": len(split.val),
        "rotations": list(ROTATIONS),
        "label_semantics": "output class is the clockwise rotation to apply to make the input upright",
        "output": str(args.output),
        "history": history,
    }
    report_path = args.output.with_suffix(".training.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
