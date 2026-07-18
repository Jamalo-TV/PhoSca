"""Compare the frozen legacy enhancer with the conservative production route."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.evaluation.enhancement import measure_enhancement  # noqa: E402
from app.evaluation.legacy_enhancement import legacy_conservative_enhance_photo  # noqa: E402


def _clean_scene(*, monochrome: bool = False, sepia: bool = False) -> np.ndarray:
    height, width = 360, 520
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    sky = np.clip(205.0 - y * 0.18 + x * 0.025, 0, 255)
    green = np.clip(72.0 + y * 0.22 + 18.0 * np.sin(x / 19.0), 0, 255)
    image = np.stack((sky * 0.82, sky, sky * 1.04), axis=2)
    ground = y > height * 0.52
    image[ground, 0] = green[ground] * 0.65
    image[ground, 1] = green[ground]
    image[ground, 2] = green[ground] * 0.72
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.rectangle(image, (55, 58), (215, 270), (65, 95, 135), -1)
    cv2.rectangle(image, (65, 68), (205, 260), (125, 165, 205), -1)
    cv2.ellipse(image, (335, 175), (62, 82), -8, 0, 360, (105, 160, 205), -1)
    cv2.circle(image, (317, 160), 7, (55, 75, 95), -1)
    cv2.circle(image, (353, 154), 7, (55, 75, 95), -1)
    cv2.putText(image, "PHOSCA 1993", (238, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2, cv2.LINE_AA)
    rng = np.random.default_rng(20260718)
    texture = rng.normal(0.0, 2.0, image.shape[:2]).astype(np.float32)
    textured = image.astype(np.float32)
    textured[ground] += texture[ground, None]
    image = np.clip(textured, 0, 255).astype(np.uint8)
    if monochrome or sepia:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if sepia:
            image = cv2.merge(
                (
                    np.clip(gray.astype(np.float32) * 0.72, 0, 255).astype(np.uint8),
                    np.clip(gray.astype(np.float32) * 0.90, 0, 255).astype(np.uint8),
                    np.clip(gray.astype(np.float32) * 1.08, 0, 255).astype(np.uint8),
                )
            )
        else:
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return image


def _synthetic_cases() -> list[dict]:
    clean = _clean_scene()
    faded = cv2.convertScaleAbs(clean, alpha=0.58, beta=58)
    faded = faded.astype(np.int16)
    faded[:, :, 2] += 18
    faded[:, :, 1] += 8
    faded = np.clip(faded, 0, 255).astype(np.uint8)

    normalized = clean.astype(np.float32) / 255.0
    underexposed = np.clip(np.power(normalized, 1.45) * 195.0, 0, 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(clean, (0, 0), 1.15)
    noise = np.random.default_rng(42).normal(0.0, 7.0, blurred.shape).astype(np.float32)
    noisy_soft = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    encoded = cv2.imencode(".jpg", clean, [int(cv2.IMWRITE_JPEG_QUALITY), 36])[1]
    jpeg = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    monochrome = _clean_scene(monochrome=True)
    sepia = _clean_scene(sepia=True)
    return [
        {
            "id": "faded_yellow_cast",
            "reference": clean,
            "source": faded,
            "tags": ["generated_fixture", "fading", "colour_cast"],
        },
        {
            "id": "underexposed",
            "reference": clean,
            "source": underexposed,
            "tags": ["generated_fixture", "underexposure"],
        },
        {
            "id": "noisy_soft",
            "reference": clean,
            "source": noisy_soft,
            "tags": ["generated_fixture", "noise", "scanner_softness"],
        },
        {"id": "jpeg_artifacts", "reference": clean, "source": jpeg, "tags": ["generated_fixture", "jpeg_artifacts"]},
        {
            "id": "already_good",
            "reference": clean,
            "source": clean.copy(),
            "tags": ["generated_fixture", "already_good"],
        },
        {
            "id": "intentional_monochrome",
            "reference": monochrome,
            "source": monochrome.copy(),
            "tags": ["generated_fixture", "monochrome_intent"],
        },
        {
            "id": "intentional_sepia",
            "reference": sepia,
            "source": sepia.copy(),
            "tags": ["generated_fixture", "sepia_intent"],
        },
    ]


def _hash_pixels(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _save(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def _html_report(output_dir: Path, cases: list[dict]) -> None:
    rows = []
    for case in cases:
        cells = []
        for label, filename in case["artifacts"].items():
            cells.append(
                f'<figure><img loading="lazy" src="{html.escape(filename)}" alt="">'
                f"<figcaption>{html.escape(label)}</figcaption></figure>"
            )
        rows.append(
            f"<section><h2>{html.escape(case['id'])}</h2><p>{html.escape(', '.join(case['tags']))}</p>"
            f'<div class="grid">{"".join(cells)}</div><details><summary>Metrics and provenance</summary>'
            f"<pre>{html.escape(json.dumps(case['metrics'], indent=2))}</pre></details></section>"
        )
    document = (
        '<!doctype html>\n<html><head><meta charset="utf-8"><title>PhoSca enhancement evaluation</title>\n'
        "<style>body{font:15px system-ui;margin:2rem;background:#111;color:#eee}"
        "section{margin:0 0 3rem}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:1rem}"
        "figure{margin:0}img{width:100%;height:auto;background:#222}"
        "figcaption{margin-top:.35rem;color:#bbb}pre{white-space:pre-wrap;overflow:auto}"
        "@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>\n"
        "</head><body><h1>PhoSca enhancement evaluation</h1>"
        "<p>Synthetic paired fixtures are regression evidence only; inspect every image.</p>"
        f"{''.join(rows)}</body></html>"
    )
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate faithful enhancement against paired synthetic references.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/enhancement-evaluation"))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    from app.pipeline.enhancement import conservative_enhance_photo

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    for fixture in _synthetic_cases():
        source = fixture["source"]
        reference = fixture["reference"]
        legacy_started = perf_counter()
        legacy, legacy_metadata = legacy_conservative_enhance_photo(source)
        legacy_runtime = (perf_counter() - legacy_started) * 1000.0
        current_started = perf_counter()
        current, current_metadata = conservative_enhance_photo(source)
        current_runtime = (perf_counter() - current_started) * 1000.0
        difference = cv2.convertScaleAbs(cv2.absdiff(source, current), alpha=3.0)

        prefix = fixture["id"]
        artifacts = {
            "reference": f"{prefix}-reference.png",
            "source": f"{prefix}-source.png",
            "legacy": f"{prefix}-legacy.png",
            "conservative": f"{prefix}-conservative.png",
            "difference_x3": f"{prefix}-difference-x3.png",
        }
        for name, image in (
            (artifacts["reference"], reference),
            (artifacts["source"], source),
            (artifacts["legacy"], legacy),
            (artifacts["conservative"], current),
            (artifacts["difference_x3"], difference),
        ):
            _save(args.output_dir / name, image)
        metrics = {
            "source": measure_enhancement(source, reference).as_dict(),
            "legacy": measure_enhancement(legacy, reference).as_dict(),
            "conservative": measure_enhancement(current, reference).as_dict(),
            "runtime_ms": {"legacy": legacy_runtime, "conservative": current_runtime},
            "hashes": {
                "source": _hash_pixels(source),
                "legacy": _hash_pixels(legacy),
                "conservative": _hash_pixels(current),
            },
            "legacy_provenance": legacy_metadata,
            "conservative_provenance": current_metadata,
        }
        cases.append({"id": prefix, "tags": fixture["tags"], "artifacts": artifacts, "metrics": metrics})

    def aggregate(route: str, metric: str) -> float:
        values = [case["metrics"][route][metric] for case in cases if case["metrics"][route][metric] is not None]
        return float(np.mean(values)) if values else 0.0

    preservation_ids = {"already_good", "intentional_monochrome", "intentional_sepia"}
    preservation = [case for case in cases if case["id"] in preservation_ids]
    summary = {
        "schema_version": 1,
        "dataset": {"kind": "generated_fixture", "cases": len(cases), "real_photos": 0},
        "mean": {
            route: {
                "psnr": aggregate(route, "psnr"),
                "ssim": aggregate(route, "ssim"),
                "lab_error": aggregate(route, "lab_error"),
                "texture_error": aggregate(route, "texture_error"),
                "halo_score": aggregate(route, "halo_score"),
            }
            for route in ("source", "legacy", "conservative")
        },
        "preservation": {
            route: {
                "mean_psnr": float(np.mean([case["metrics"][route]["psnr"] for case in preservation])),
                "mean_ssim": float(np.mean([case["metrics"][route]["ssim"] for case in preservation])),
            }
            for route in ("legacy", "conservative")
        },
        "limitations": [
            "These generated paired cases test regression behavior, not real historical-photo quality.",
            "No face-identity or OCR-consistency ground truth is available in the repository.",
            "No-reference metrics are descriptive safeguards and never authorize automatic release alone.",
        ],
        "cases": cases,
    }
    report_path = args.report or args.output_dir / "report.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _html_report(args.output_dir, cases)
    print(
        json.dumps(
            {"report": str(report_path), "summary": summary["mean"], "preservation": summary["preservation"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
