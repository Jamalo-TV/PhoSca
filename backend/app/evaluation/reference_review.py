"""Review the independent inputs used by the crop reference preflight."""

from __future__ import annotations

from base64 import b64encode
from copy import deepcopy
from datetime import datetime
from html import escape
from pathlib import Path
import os
from urllib.parse import quote

import cv2
import numpy as np


_MODES = (("full_print", "Ganzer Abzug"), ("image_only", "Nur Bildmotiv"))
_INK = (63, 83, 255)
_MASK = (50, 213, 223)
_UNVERIFIED = {"PAGE_ANNOTATION_UNVERIFIED", "MODE_ANNOTATION_UNVERIFIED"}
_OUTCOMES = {"verified": "manually_verified", "needs_revision": "reference_review_needs_revision"}


def apply_reference_review(manifest: dict | list, receipt: dict, preflight: dict, *, receipt_sha256: str) -> dict | list:
    """Apply explicit review decisions to a copy of the exact manifest that was checked."""
    digest = preflight["dataset_manifest_sha256"]
    if (not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int
            or receipt["schema_version"] != 1 or receipt.get("manifest_sha256") != digest):
        raise ValueError("Review receipt does not match this reference manifest")
    reviewer, reviewed_at = receipt.get("reviewer"), receipt.get("reviewed_at")
    if not isinstance(reviewer, str) or not 1 <= len(reviewer.strip()) <= 120:
        raise ValueError("Review receipt needs a reviewer")
    if not isinstance(reviewed_at, str) or not reviewed_at.endswith("Z"):
        raise ValueError("Review receipt needs a UTC timestamp")
    datetime.fromisoformat(reviewed_at[:-1] + "+00:00")
    decisions = receipt.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("Review receipt has no decisions")

    reviewed = deepcopy(manifest)
    pages = reviewed if isinstance(reviewed, list) else reviewed["pages"]
    by_id = {str(page["id"]): page for page in pages}
    page_issues = {str(row["page_id"]): set(row["issues"]) for row in preflight["reference_issues"]["pages"]}
    mode_issues = {
        (str(row["page_id"]), row["instance_index"], row["mode"]): set(row["issues"])
        for row in preflight["reference_issues"]["modes"]
    }
    seen = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("Invalid reference review decision")
        page_id, instance = decision.get("page_id"), decision.get("instance_index")
        mode, outcome = decision.get("mode"), decision.get("outcome")
        if (not isinstance(page_id, str) or page_id not in by_id or not isinstance(mode, str)
                or not isinstance(outcome, str) or outcome not in _OUTCOMES):
            raise ValueError("Unknown page, crop mode or outcome in reference review")
        page = by_id[page_id]
        issues = page_issues.get(page_id, set())
        if mode == "page" and instance is None:
            target = page
        else:
            crops = page.get("expected_final_crops") or []
            if (mode not in dict(_MODES) or type(instance) is not int or not 0 <= instance < len(crops)
                    or mode not in crops[instance]):
                raise ValueError("Review decision must address an existing crop reference")
            target = crops[instance][mode]
            issues = issues | mode_issues.get((page_id, instance, mode), set())
        identity = (page_id, instance, mode)
        if identity in seen:
            raise ValueError("Duplicate reference review decision")
        seen.add(identity)
        blockers = issues - _UNVERIFIED
        if outcome == "verified" and blockers:
            raise ValueError(f"Cannot confirm {page_id}/{instance}/{mode}: {', '.join(sorted(blockers))}")
        target["annotation_provenance"] = _OUTCOMES[outcome]
        target["reference_review"] = {
            "reviewer": reviewer.strip(), "reviewed_at": reviewed_at, "manifest_sha256": digest,
            "receipt_sha256": receipt_sha256, "outcome": outcome,
        }
    return reviewed


def _encode_image(suffix: str, image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"Could not encode review image as {suffix}")
    return encoded.tobytes()


def _image_file(directory: Path, name: str, image: np.ndarray) -> None:
    (directory / name).write_bytes(_encode_image(Path(name).suffix, image))


def _embedded_image(mime: str, pixels: bytes, label: str) -> str:
    encoded = b64encode(pixels).decode("ascii")
    return f'<img loading="lazy" src="data:{mime};base64,{encoded}" alt="{escape(label, quote=True)}">'


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


def _review_choice(page_id: str, mode: str, label: str, issues: list[str], instance: int | None = None) -> str:
    instance_attr = "" if instance is None else f' data-instance="{instance}"'
    verified = ('<option value="verified">Bestätigt</option>' if not (set(issues) - _UNVERIFIED) else "")
    return (
        f'<label class="review-choice">{escape(label)} '
        f'<select data-page-id="{escape(page_id, quote=True)}" data-mode="{mode}"{instance_attr}>'
        '<option value="">Keine Änderung</option>' + verified
        + '<option value="needs_revision">Korrektur nötig</option></select></label>'
    )


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
    page_issues: list[str], issues_by_mode: dict[tuple[int, str], list[str]], *, calibration: bool = False,
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
    page_id = str(record["id"])
    identity = escape(page_id)
    group = escape(str(record.get("source_group") or "Unbekannte Albumgruppe"))
    heading = f'<section id="{slug}" class="page"><h2>{identity}</h2><p>{group} · {width} × {height} px'
    if calibration:
        path = record["image"]
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            original = _embedded_image(mime, path.read_bytes(), "Original in voller Auflösung")
        else:
            original = _embedded_image("image/png", _encode_image(".png", image), "Quellbild in voller Auflösung")
        outlines = _embedded_image("image/jpeg", _encode_image(".jpg", overlay), "Quellbild mit markierten Umrissen")
        return (
            f"{heading} · {len(expected)} markierte Abzüge</p><ul>{_issues(page_issues)}</ul>"
            + _review_choice(page_id, "page", "Fotoanzahl, Zuordnung und Umrisse prüfen", page_issues)
            + f'<figure>{outlines}<figcaption>Markierte Umrisse</figcaption></figure>'
            + '<details><summary>Original in voller Auflösung öffnen</summary>'
            + f'<div class="native-image">{original}</div></details></section>'
        )
    source = escape(_source_path(directory, record["image"]), quote=True)
    figures = _figure(directory, f"{slug}-overview.jpg", overlay, "Sichtbare Instanzen laut Referenzlabels", 1120)
    blocks = []
    for instance, modes in enumerate(references):
        cards = []
        for key, label in _MODES:
            mode = modes.get(key) or {}
            codes = issues_by_mode.get((instance, key), [])
            issue_list = _issues(codes)
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
                f'<ul>{issue_list}</ul><div class="evidence">{evidence}</div>'
                + _review_choice(page_id, key, f"{label} prüfen", [*page_issues, *codes], instance)
                + "</article>"
            )
        blocks.append(f'<section><h3>Instanz {instance}</h3><div class="modes">{"".join(cards)}</div></section>')
    return (
        f"{heading} · "
        f'<a href="{source}">Original öffnen</a></p><ul>{_issues(page_issues)}</ul>{figures}'
        + _review_choice(page_id, "page", "Fotoanzahl und Zuordnung prüfen", page_issues)
        + ("".join(blocks) if blocks else "<p>Die Referenzlabels enthalten keine Fotos.</p>")
        + "</section>"
    )


_EXPORT_REVIEW = """<script>
const form = document.getElementById('reference-review');
const controls = Array.from(form.querySelectorAll('select[data-page-id]'));
const exportButton = document.getElementById('export-decisions');
const count = document.getElementById('decision-count');
const receiptText = document.getElementById('receipt-text');
receiptText.addEventListener('focus', () => receiptText.select());
function updateCount() {
    const selected = controls.filter(control => control.value).length;
    exportButton.disabled = selected === 0;
    count.textContent = `${selected} von ${controls.length} Entscheidungen`;
}
form.addEventListener('change', updateCount);
form.addEventListener('submit', event => {
    event.preventDefault();
    const reviewer = form.elements.reviewer.value.trim();
    if (!reviewer) {
        form.elements.reviewer.focus();
        return;
    }
    const decisions = controls.filter(control => control.value).map(control => ({
        page_id: control.dataset.pageId,
        mode: control.dataset.mode,
        instance_index: control.dataset.instance === undefined ? null : Number(control.dataset.instance),
        outcome: control.value,
    }));
    if (!decisions.length) return;
    const receipt = {
        schema_version: 1, manifest_sha256: form.dataset.manifestSha256,
        reviewer, reviewed_at: new Date().toISOString(), decisions,
    };
    receiptText.value = JSON.stringify(receipt, null, 2) + '\\n';
    document.getElementById('receipt-copy').hidden = false;
    const url = URL.createObjectURL(new Blob([receiptText.value], {type: 'application/json'}));
    const link = document.createElement('a');
    link.href = url;
    link.download = form.dataset.receiptFilename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});
updateCount();
</script>"""


def write_reference_review(
    directory: Path, summary: dict, pages: list[tuple[str, str]], *, calibration: bool = False,
) -> Path:
    path = directory / "index.html"
    ready = "Referenzprüfung vollständig" if summary["ready_for_locked_evaluation"] else "Referenzprüfung offen"
    digest = escape(str(summary["dataset_manifest_sha256"]))
    dataset_issues = _issues(summary["reference_issues"]["dataset"])
    contents = "".join(
        f'<li><a href="#page-{index:03d}">{escape(name)}</a></li>' for index, (name, _) in enumerate(pages)
    )
    if calibration:
        title = "Kalibrierung · Seitenprüfung"
        filename = "phosca-kalibrierungspruefung.json"
        instructions = (
            "<p>Prüfe auf jeder Seite Anzahl, Zuordnung und sichtbare Umrisse der Abzüge. "
            "Wähle bei unklaren Rändern <strong>Korrektur nötig</strong>. "
            "Trage deinen Namen ein und speichere nach der letzten Entscheidung den JSON-Beleg.</p>"
        )
        status = f'<p>{summary["pages"]} Ansichten · {summary["expected_photos"]} markierte Abzüge</p>'
    else:
        title = "Albumreferenzen"
        filename = "phosca-referenzpruefung.json"
        instructions = (
            "<p>Vergleiche die Quellpixel mit den markierten Grenzen und der separaten Aufnahme. "
            "Die Eckbilder zeigen links die Quelle und rechts die Markierung bei dreifacher Vergrößerung. "
            "Die Instanznummern beginnen bei 0 und folgen den YOLO-Labels.</p>"
        )
        status = f'<p>{ready} · {summary["pages"]} Seiten · {summary["expected_photos"]} Abzüge</p><ul>{dataset_issues}</ul>'
    path.write_text(
        '<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        f'<title>PhoSca {title}</title><style>'
        "body{margin:24px;background:#14171b;color:#eef1f5;font:16px system-ui}"
        "main{max-width:1600px;margin:auto}.page{border-top:1px solid #434a54;margin:24px 0;padding-top:12px}"
        "a{color:#a8dcff}code{color:#f2d898;overflow-wrap:anywhere}figure{margin:0}"
        "img{max-width:100%;height:auto}figcaption{font-size:.9rem;margin-bottom:.6rem}"
        ".modes,.corner-grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr))}"
        "article{background:#20242a;padding:14px}article h4{margin:0;font-size:1.05rem}"
        "nav ul{columns:3 280px}nav li{padding:2px 0}"
        ".evidence{display:grid;gap:12px}ul:empty{display:none}h3{margin-top:24px}"
        ".review-actions{position:sticky;top:0;z-index:1;background:#20242a;padding:12px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}"
        ".page,article{scroll-margin-top:80px}.review-choice{display:block;margin:10px 0}"
        "input,select,button{font:inherit;padding:.25em .4em;max-width:100%}"
        ".native-image{overflow:auto;max-height:80vh}.native-image img{display:block;max-width:none}"
        "#receipt-text{display:block;width:min(100%,50rem);height:10em}"
        f"</style><main><h1>{title}</h1>{status}"
        f"<p>Manifest SHA-256 <code>{digest}</code></p>{instructions}"
        f'<form id="reference-review" data-manifest-sha256="{digest}" data-receipt-filename="{filename}">'
        '<div class="review-actions"><label>Geprüft von <input name="reviewer" maxlength="120" required></label>'
        '<output id="decision-count" aria-live="polite"></output>'
        '<button id="export-decisions" type="submit" disabled>Prüfentscheidungen speichern</button></div>'
        '<div id="receipt-copy" hidden><label>Falls der Download ausbleibt, kopiere den JSON-Beleg aus diesem Feld.'
        '<textarea id="receipt-text" readonly></textarea></label></div>'
        f'<nav aria-label="Albumseiten"><ul>{contents}</ul></nav>'
        + "".join(section for _, section in pages)
        + "</form></main>" + _EXPORT_REVIEW + "</html>",
        encoding="utf-8",
    )
    return path
