"""Render the independent inputs used by the crop reference preflight."""

from __future__ import annotations

from html import escape
from pathlib import Path
import os
from urllib.parse import quote

import cv2
import numpy as np


_MODES = (("full_print", "Ganzer Abzug"), ("image_only", "Nur Bildmotiv"))
_INK = (63, 83, 255)
_MASK = (50, 213, 223)


def _image_file(directory: Path, name: str, image: np.ndarray) -> None:
    suffix = Path(name).suffix
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"Could not encode review image: {name}")
    (directory / name).write_bytes(encoded.tobytes())


def _figure(directory: Path, name: str, image: np.ndarray, caption: str, max_width: int = 560) -> str:
    if image.shape[1] > max_width:
        image = cv2.resize(
            image, (max_width, max(1, round(max_width * image.shape[0] / image.shape[1]))),
            interpolation=cv2.INTER_AREA,
        )
    _image_file(directory, name, image)
    label = escape(caption)
    return (
        f'<figure><a href="{name}"><img loading="lazy" src="{name}" alt="{label}"></a>'
        f"<figcaption>{label}</figcaption></figure>"
    )


def _issues(codes: list[str]) -> str:
    return "".join(f"<li><code>{escape(code)}</code></li>" for code in codes)


def _source_path(directory: Path, path: Path) -> str:
    try:
        href = Path(os.path.relpath(path.resolve(), directory.resolve())).as_posix()
        return quote(href, safe="/")
    except ValueError:  # Windows paths on different volumes.
        return path.resolve().as_uri()


def _source_region(
    image: np.ndarray, reference_mask: np.ndarray, corners: np.ndarray | None,
) -> np.ndarray:
    x, y, width, height = cv2.boundingRect(reference_mask)
    left, top = max(x - 15, 0), max(y - 15, 0)
    right, bottom = min(x + width + 15, image.shape[1]), min(y + height + 15, image.shape[0])
    region = image[top:bottom, left:right].copy()
    contours, _ = cv2.findContours(reference_mask[top:bottom, left:right], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.polylines(region, contours, True, _MASK, max(1, width // 240))
    if corners is not None:
        cv2.polylines(region, [np.rint(corners - [left, top]).astype(np.int32)], True, _INK, max(1, width // 325))
    return region


def _corner_tiles(directory: Path, prefix: str, image: np.ndarray, corners: np.ndarray) -> str:
    tiles = []
    zoom = 3
    for index, point in enumerate(corners):
        left, top = np.maximum(np.rint(point).astype(int) - 26, 0)
        right, bottom = np.minimum(np.rint(point).astype(int) + 27, image.shape[1::-1])
        source = image[top:bottom, left:right]
        enlarged = cv2.resize(source, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        marked = enlarged.copy()
        local = (corners - [left, top] + .5) * zoom - .5
        cv2.polylines(marked, [np.rint(local * 256).astype(np.int32)], True, _INK, 1, shift=8)
        center = tuple(np.rint(local[index] * 256).astype(int))
        cv2.circle(marked, center, 3 * 256, _INK, 1, shift=8)
        caption = f"Ecke {index + 1} · x {point[0]:.2f}, y {point[1]:.2f} px"
        tiles.append(_figure(directory, f"{prefix}-corner-{index}.png", np.hstack((enlarged, marked)), caption))
    return '<div class="corner-grid">' + "".join(tiles) + "</div>"


def _reference_pair(
    directory: Path, prefix: str, image: np.ndarray, reference: np.ndarray, corners: np.ndarray | None,
) -> str:
    name = f"{prefix}-independent.png"
    _image_file(directory, name, reference)
    external = (
        f'<figure><a href="{name}"><img loading="lazy" src="{name}" alt="Separate Referenzaufnahme"></a>'
        "<figcaption>Separate Referenzaufnahme</figcaption></figure>"
    )
    if corners is None:
        return external
    height, width = reference.shape[:2]
    scale = min(1.0, 559 / max(height - 1, width - 1))
    target = (max(2, round((width - 1) * scale) + 1), max(2, round((height - 1) * scale) + 1))
    right, bottom = target[0] - 1, target[1] - 1
    frame = np.float32([[0, 0], [right, 0], [right, bottom], [0, bottom]])
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), frame)
    draft = cv2.warpPerspective(image, matrix, target)
    source = _figure(directory, f"{prefix}-draft.jpg", draft, "Quellansicht nach Referenzecken")
    return '<div class="corner-grid">' + source + external + "</div>"


def render_reference_page(
    directory: Path, index: int, record: dict, image: np.ndarray, expected: list, references: list[dict],
    page_issues: list[str], issues_by_mode: dict[tuple[int, str], list[str]],
) -> str:
    height, width = image.shape[:2]
    scale = np.array([width - 1, height - 1])
    overlay = image.copy()
    for instance, polygon in enumerate(expected):
        points = np.rint(np.asarray(polygon) * scale).astype(np.int32)
        cv2.polylines(overlay, [points], True, _MASK, max(2, width // 410))
        x, y = points.min(axis=0)
        origin = (int(x), max(int(y) - 5, 18))
        font_scale = max(.7, width / 2400)
        cv2.putText(overlay, str(instance), origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(overlay, str(instance), origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, _MASK, 2, cv2.LINE_AA)
    slug = f"page-{index:03d}"
    source = escape(_source_path(directory, record["image"]), quote=True)
    figures = _figure(directory, f"{slug}-overview.jpg", overlay, "Sichtbare Instanzen laut Referenzlabels", 1120)
    blocks = []
    for instance, modes in enumerate(references):
        cards = []
        for key, label in _MODES:
            mode = modes.get(key) or {}
            issue_list = _issues(issues_by_mode.get((instance, key), []))
            if not mode.get("present", True) or not mode:
                cards.append(f"<article><h4>{label}</h4><p>Referenz fehlt.</p><ul>{issue_list}</ul></article>")
                continue
            prefix = f"{slug}-{instance}-{key}"
            corners = mode.get("corners")
            if (
                corners is not None and (
                    corners.shape != (4, 2) or not np.isfinite(corners).all()
                    or np.any((corners < 0) | (corners > scale))
                    or not cv2.isContourConvex(corners.astype(np.float32))
                )
            ):
                corners = None
            evidence = _figure(
                directory, f"{prefix}-source.jpg", _source_region(image, mode["mask"], corners),
                "Quellpixel · gelb sichtbare Fläche · rot Referenzecken",
            )
            reference = mode.get("image")
            if reference is not None:
                evidence += _reference_pair(directory, prefix, image, reference, corners)
            if corners is not None:
                evidence += _corner_tiles(directory, prefix, image, corners)
            aspect = mode.get("expected_aspect_ratio")
            shape = mode.get("shape")
            measurement = ""
            if aspect is not None:
                measurement = f" · gemessenes Seitenverhältnis {aspect:.4f}"
            elif shape is not None:
                measurement = f" · Referenzraster {shape[1]} × {shape[0]} px"
            status = "Bestätigte Referenz" if mode.get("verified") and not issue_list else "Entwurf"
            recoverability = "wiederherstellbar" if mode["recoverable"] else "verdeckt oder unvollständig"
            cards.append(
                f"<article><h4>{label}</h4><p>{status} · {recoverability}{measurement}</p>"
                f'<ul>{issue_list}</ul><div class="evidence">{evidence}</div></article>'
            )
        blocks.append(f'<section><h3>Instanz {instance}</h3><div class="modes">{"".join(cards)}</div></section>')
    identity = escape(str(record["id"]))
    group = escape(str(record.get("source_group") or "Unbekannte Albumgruppe"))
    return (
        f'<section id="{slug}" class="page"><h2>{identity}</h2><p>{group} · {width} × {height} px · '
        f'<a href="{source}">Original öffnen</a></p><ul>{_issues(page_issues)}</ul>{figures}'
        + ("".join(blocks) if blocks else "<p>Die Referenzlabels enthalten keine Fotos.</p>")
        + "</section>"
    )


def write_reference_review(directory: Path, summary: dict, pages: list[tuple[str, str]]) -> Path:
    path = directory / "index.html"
    ready = "Referenzprüfung vollständig" if summary["ready_for_locked_evaluation"] else "Referenzprüfung offen"
    digest = escape(str(summary["dataset_manifest_sha256"]))
    dataset_issues = _issues(summary["reference_issues"]["dataset"])
    contents = "".join(
        f'<li><a href="#page-{index:03d}">{escape(name)}</a></li>' for index, (name, _) in enumerate(pages)
    )
    path.write_text(
        '<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>PhoSca Albumreferenzen</title><style>'
        "body{margin:24px;background:#14171b;color:#eef1f5;font:16px system-ui}"
        "main{max-width:1600px;margin:auto}.page{border-top:1px solid #434a54;margin:24px 0;padding-top:12px}"
        "a{color:#a8dcff}code{color:#f2d898;overflow-wrap:anywhere}figure{margin:0}"
        "img{max-width:100%;height:auto}figcaption{font-size:.9rem;margin-bottom:.6rem}"
        ".modes,.corner-grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr))}"
        "article{background:#20242a;padding:14px}article h4{margin:0;font-size:1.05rem}"
        "nav ul{columns:3 280px}nav li{padding:2px 0}"
        ".evidence{display:grid;gap:12px}ul:empty{display:none}h3{margin-top:24px}"
        "</style><main><h1>Albumreferenzen</h1>"
        f'<p>{ready} · {summary["pages"]} Seiten · {summary["expected_photos"]} Abzüge</p>'
        f"<p>Manifest SHA-256 <code>{digest}</code></p><ul>{dataset_issues}</ul>"
        "<p>Vergleiche die Quellpixel mit den markierten Grenzen und der separaten Aufnahme. "
        "Die Eckbilder zeigen links die Quelle und rechts die Markierung bei dreifacher Vergrößerung. "
        "Die Instanznummern beginnen bei 0 und folgen den YOLO-Labels.</p>"
        f'<nav aria-label="Albumseiten"><ul>{contents}</ul></nav>'
        + "".join(section for _, section in pages)
        + "</main></html>",
        encoding="utf-8",
    )
    return path
