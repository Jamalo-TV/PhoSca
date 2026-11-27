"""Conservative protected-region primitives for crop safety.

This is intentionally a small, inspectable baseline.  It accepts manually or
externally produced regions and adds locally available face proposals.  It is
not a claim that Haar detection is a complete person/object/text safety model;
unknown regions cause no expansion but every known boundary-crossing region is
a hard veto.
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Iterable

import cv2
import numpy as np


PROTECTED_REGION_REVISION = 2
OFF_PRINT_ADVISORY = "off_print_advisory"
PAGE_FACE_SUGGESTIONS = {
    "opencv_haarcascade_frontalface_default": "face",
    "opencv_haarcascade_profileface": "head",
}


class ProtectedRegionRemoval(ValueError):
    def __init__(self, mode: str, regions: list[dict[str, float]]) -> None:
        super().__init__("protected_region_removed")
        self.detail = {"code": str(self), "mode": mode, "regions": regions}


def protected_region_key(region: dict) -> dict:
    return {"type": region.get("type", "unknown"), "source": region.get("source", "unknown"),
            "box": region.get("box", region)}


def is_advisory_page_face(region: dict) -> bool:
    return (PAGE_FACE_SUGGESTIONS.get(region.get("source")) == region.get("type")
            and region.get("source") in PAGE_FACE_SUGGESTIONS
            and not any(region.get(flag) for flag in ("pinned", "verified", "is_verified")))


def protected_region_digest(regions: list[dict]) -> str:
    return hashlib.sha256(json.dumps(regions, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def album_mark_matches_quad(review: dict, quad: list[dict] | None) -> bool:
    return isinstance(quad, list) and len(quad) == 4 and review.get("warp_quad") == quad


def album_mark_correction_matches(review: dict, mask: dict, correction: dict) -> bool:
    quad = mask.get("warp_quad") or mask.get("polygon")
    return bool(review.get("id") and album_mark_matches_quad(review, quad)
                and (correction.get("protected_regions") or {}).get("reviewed_album_mark_id") == review["id"])


def album_mark_source_matches(review: dict, page_sha256: str, page_metadata: dict) -> bool:
    protection = ((page_metadata or {}).get("steps") or {}).get("protected_regions") or {}
    regions = protection.get("regions")
    return (review.get("page_sha256") == page_sha256 and protection.get("enabled", True)
            and isinstance(regions, list) and review.get("protected_regions_sha256") == protected_region_digest(regions)
            and (review.get("source") != OFF_PRINT_ADVISORY
                 or review.get("photo_count_revision") == (page_metadata or {}).get("photo_count_revision")))


def region_box_in_pixels(region: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float] | None:
    box = region.get("box") or region.get("bounding_box") or region
    try:
        x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return None
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


_MARGINS = {"face": 0.18, "head": 0.16, "person": 0.10, "text": 0.08, "object": 0.08, "subject": 0.10}
_cascade_cache = threading.local()


def _iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = left["box"], right["box"]
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - intersection
    return float(intersection / union) if union > 0 else 0.0


def merge_protected_regions(*groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize and de-duplicate detector/manual regions without weakening their union."""
    merged: list[dict[str, Any]] = []
    for region in (item for group in groups for item in group):
        record = dict(region)
        record["type"] = str(record.get("type", "subject"))
        record.setdefault("margin_ratio", _MARGINS.get(record["type"], 0.10))
        advisory = is_advisory_page_face(record)
        duplicate = next((item for item in merged if item["type"] == record["type"] and _iou(item, record) >= 0.85
                          and is_advisory_page_face(item) == advisory), None)
        if duplicate is None:
            merged.append(record)
        elif float(record.get("confidence", 0.0)) > float(duplicate.get("confidence", 0.0)):
            merged[merged.index(duplicate)] = record
    return merged


def _cascade_regions(gray: np.ndarray, filename: str, region_type: str) -> list[dict[str, Any]]:
    # Reuse parsed classifiers on a worker without sharing OpenCV's mutable detector across threads.
    detectors = vars(_cascade_cache)
    detector = detectors.get(filename)
    if detector is None:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
        if detector.empty():
            return []
        detectors[filename] = detector
    boxes = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6, minSize=(24, 24))
    return [
        {
            "type": region_type,
            "source": f"opencv_{filename.removesuffix('.xml')}",
            "confidence": 0.75,
            "box": {"x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h)},
            "margin_ratio": _MARGINS[region_type],
        }
        for x, y, w, h in boxes
    ]


def _person_regions(image: np.ndarray) -> list[dict[str, Any]]:
    """High-threshold built-in person proposals; optional model regions can be supplied separately."""
    height, width = image.shape[:2]
    scale = min(1.0, 960.0 / max(height, width))
    work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else image
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = detector.detectMultiScale(work, winStride=(8, 8), padding=(8, 8), scale=1.08)
    regions = []
    for (x, y, w, h), weight in zip(boxes, weights):
        if float(weight) < 0.8:
            continue
        regions.append({
            "type": "person",
            "source": "opencv_hog_person",
            "confidence": min(0.95, float(weight)),
            "box": {"x1": int(x / scale), "y1": int(y / scale), "x2": int((x + w) / scale), "y2": int((y + h) / scale)},
            "margin_ratio": _MARGINS["person"],
        })
    return regions


def detect_protected_regions(
    image: np.ndarray,
    external_regions: list[dict[str, Any]] | None = None,
    *,
    include_legacy_hog_people: bool = False,
) -> list[dict[str, Any]]:
    """Build a conservative union of local and externally supplied semantic regions."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = _cascade_regions(gray, "haarcascade_frontalface_default.xml", "face")
        profiles = _cascade_regions(gray, "haarcascade_profileface.xml", "head")
        # OpenCV's legacy HOG people detector is both inaccurate for small
        # album subjects and has shown native crashes on Windows. It remains an
        # explicit research opt-in, never part of the automatic safety path.
        people = _person_regions(image) if include_legacy_hog_people else []
    except (cv2.error, AttributeError):
        faces, profiles, people = [], [], []
    return merge_protected_regions(faces, profiles, people, external_regions or [])


def validate_protected_regions(
    quad: np.ndarray,
    regions: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    require_containment: bool = False,
) -> dict[str, Any]:
    """Reject cuts across protected regions, or require all regions inside a crop."""
    contour = quad.reshape(-1, 1, 2).astype(np.float32)
    violations: list[dict[str, Any]] = []
    for region in regions:
        box = region_box_in_pixels(region, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        margin = float(region.get("margin_ratio", 0.08)) * max(x2 - x1, y2 - y1)
        corners = np.array(
            [[x1 - margin, y1 - margin], [x2 + margin, y1 - margin], [x2 + margin, y2 + margin], [x1 - margin, y2 + margin]],
            dtype=np.float32,
        )
        inside = [cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0 for x, y in corners]
        if not all(inside) and (require_containment or any(inside)):
            violations.append(protected_region_key(region))
    return {"passed": not violations, "violations": violations, "regions_checked": len(regions)}


def removed_protected_regions(
    original: np.ndarray, edited: np.ndarray, regions: list[dict], *, width: int, height: int,
    visible: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Locate lost protected content in the old crop's visible footprint."""
    if np.array_equal(original, edited):
        return []
    # A signed fan handles concave visibility polygons without treating gaps
    # between observed pieces as photo content.
    visibility = None
    if visible is not None:
        if cv2.isContourConvex(visible):
            visibility = [(visible, 1)]
        else:
            triangles = [np.float32([visible[0], start, end]) for start, end in zip(visible[1:-1], visible[2:])]
            orientation = 1 if cv2.contourArea(visible, oriented=True) > 0 else -1
            visibility = [(triangle, orientation * np.sign(area)) for triangle in triangles
                          if (area := cv2.contourArea(triangle, oriented=True))]
    removed = []
    for region in regions:
        box = region_box_in_pixels(region, width, height)
        if box is None or not np.isfinite(box).all():
            continue
        left, top, right, bottom = np.clip(box, 0, [width - 1, height - 1] * 2).tolist()
        area = (right - left) * (bottom - top)
        if area <= 1:
            continue
        core = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])
        covered, owned = cv2.intersectConvexConvex(original, core)
        if covered <= 1 or (visibility is None and is_advisory_page_face(region) and area - covered > 1):
            continue
        kept, retained = cv2.intersectConvexConvex(owned, edited)
        if covered - kept <= 1:
            continue
        if visibility is not None:
            loss = sum(sign * (cv2.intersectConvexConvex(piece, owned)[0]
                       - (cv2.intersectConvexConvex(piece, retained)[0] if kept > 0 else 0))
                       for piece, sign in visibility)
            if loss <= 1:
                continue
        removed.append({"x1": left / (width - 1), "y1": top / (height - 1),
                        "x2": right / (width - 1), "y2": bottom / (height - 1)})
    return removed
