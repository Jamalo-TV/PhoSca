"""Score an ONNX orientation model on source-pinned, reviewed photo crops."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys


ANGLES = (0, 90, 180, 270)


def read_manifest(path: Path) -> tuple[list[dict], str]:
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"Expected a schema-1 orientation manifest in {path}")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"No orientation samples in {path}")

    identities, digests = set(), set()
    validated = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError(f"Invalid orientation sample in {path}")
        if not all(isinstance(sample.get(key), str) and sample[key].strip()
                   for key in ("id", "source_group", "image", "image_sha256")):
            raise ValueError(f"An orientation sample lacks its ID, group, path or digest in {path}")
        identity = sample["id"].strip()
        group = sample["source_group"].strip().casefold()
        correction = sample.get("upright_clockwise_degrees")
        if type(correction) is not int or correction not in ANGLES:
            raise ValueError(f"Invalid clockwise orientation label for {identity}")
        image = (path.parent / sample["image"]).resolve()
        digest = sha256(image.read_bytes()).hexdigest()
        if digest != sample["image_sha256"].lower():
            raise ValueError(f"Image digest mismatch for {identity}")
        if identity in identities or digest in digests:
            raise ValueError(f"Duplicate sample or image for {identity}")
        identities.add(identity)
        digests.add(digest)
        validated.append({"id": identity, "source_group": group, "image": image,
                          "image_sha256": digest, "upright_clockwise_degrees": correction})
    return validated, sha256(payload).hexdigest()


def check_disjoint(test: list[dict], training: list[dict]) -> None:
    for key in ("source_group", "image_sha256"):
        overlap = {sample[key] for sample in test} & {sample[key] for sample in training}
        if overlap:
            raise ValueError(f"Training and evaluation overlap by {key}: {', '.join(sorted(overlap))}")


def summarize(cases: list[dict]) -> dict:
    return {
        "cases": len(cases),
        "correct_final": sum(row["rotation_degrees"] == row["expected_clockwise_degrees"] for row in cases),
        "accepted": sum(row["accepted"] for row in cases),
        "accepted_wrong": sum(row["accepted"] and row["rotation_degrees"] != row["expected_clockwise_degrees"] for row in cases),
        "wrong_rotations": sum(row["rotation_degrees"] not in (0, row["expected_clockwise_degrees"]) for row in cases),
        "wrongly_rotated_upright": sum(row["expected_clockwise_degrees"] == 0 and row["rotation_degrees"] != 0 for row in cases),
        "review": sum(not row["accepted"] for row in cases),
    }


def evaluate(model: Path, samples: list[dict]) -> dict:
    import numpy as np

    from app.pipeline.image_ops import load_image
    from app.pipeline.orientation import correct_photo_orientation, rotate_image

    cases = []
    groups = defaultdict(list)
    for sample in samples:
        payload = sample["image"].read_bytes()
        if sha256(payload).hexdigest() != sample["image_sha256"]:
            raise ValueError(f"Image changed since preflight for {sample['id']}")
        image = load_image(BytesIO(payload))
        for applied in ANGLES:
            expected = (sample["upright_clockwise_degrees"] - applied) % 360
            oriented = rotate_image(image, applied)
            result, decision = correct_photo_orientation(oriented, model_path=model)
            if decision.get("method") != "onnx_orientation":
                raise ValueError(f"The runtime did not use the selected ONNX model for {sample['id']}")
            if not np.array_equal(result, rotate_image(oriented, decision["rotation_degrees"])):
                raise ValueError(f"Output pixels contradict the recorded rotation for {sample['id']}")
            row = {
                "id": sample["id"], "source_group": sample["source_group"],
                "applied_clockwise_degrees": applied, "expected_clockwise_degrees": expected,
                "rotation_degrees": decision["rotation_degrees"], "confidence": decision["confidence"],
                "accepted": decision["decision"] in {"auto_rotate", "confidently_preserve"},
            }
            cases.append(row)
            groups[sample["source_group"]].append(row)
    return {"originals": len(samples), "overall": summarize(cases),
            "as_acquired": summarize([row for row in cases if row["applied_clockwise_degrees"] == 0]),
            "by_source_group": {name: summarize(rows) for name, rows in sorted(groups.items())}, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Reviewed orientation samples.")
    parser.add_argument("--training-manifest", type=Path, help="Check the evaluation split against these training sources.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    try:
        samples, digest = read_manifest(args.manifest)
        training_digest = None
        if args.training_manifest:
            training, training_digest = read_manifest(args.training_manifest)
            check_disjoint(samples, training)

        model = args.model.resolve()
        model_digest = sha256(model.read_bytes()).hexdigest()
        sources = {model, args.manifest.resolve(), *(row["image"] for row in samples)}
        if args.training_manifest:
            sources.update([args.training_manifest.resolve(), *(row["image"] for row in training)])
        if args.output.resolve() in sources:
            raise ValueError("Use an output path separate from the model, manifests and source images")
        results = evaluate(model, samples)
        if sha256(model.read_bytes()).hexdigest() != model_digest:
            raise ValueError("The orientation model changed during evaluation")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    report = {"schema_version": 1, "model_sha256": model_digest, "manifest_sha256": digest,
              "training_manifest_sha256": training_digest, **results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"originals": report["originals"], **report["overall"]}, indent=2))


if __name__ == "__main__":
    main()
