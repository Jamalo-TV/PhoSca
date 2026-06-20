from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw_album_pages"))
    parser.add_argument("--output", type=Path, default=Path("data/load_dataset"))
    parser.add_argument("--variants", type=int, default=10)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for existing in args.output.glob("*.jpg"):
        existing.unlink()

    for image_path in sorted(args.source.glob("*.jpg")):
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            for index in range(args.variants):
                scale = 0.90 + (index % 5) * 0.025
                rotation = [-3, -2, -1, 1, 2, 3, 0, -1.5, 1.5, 0.5][index % 10]
                resized = image.resize((int(image.width * scale), int(image.height * scale)))
                rotated = resized.rotate(rotation, expand=True, fillcolor=(245, 245, 245))
                output_path = args.output / f"{image_path.stem}_variant_{index + 1:02d}.jpg"
                rotated.save(output_path, format="JPEG", quality=85, optimize=True)

    print({"generated": len(list(args.output.glob("*.jpg"))), "output": str(args.output)})


if __name__ == "__main__":
    main()

