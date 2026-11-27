"""Compare current overlapping detections without changing their review state."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from uuid import UUID

import cv2
import numpy as np

from app.models import ExtractedPhoto
from app.pipeline.decisions import photo_is_reviewed_non_print
from app.pipeline.geometry import normalized_polygon_overlap
from app.pipeline.possible_print_regions import POSSIBLE_PRINT_REGIONS
from app.pipeline.shared_motif_review import SHARED_MOTIF_REVIEW, motif_review_id
from app.schemas.pipeline import CompetingPhotoOutline, PhotoRead


@dataclass(frozen=True)
class _Candidate:
    id: UUID
    outline: list[dict]
    vertices: np.ndarray
    valid_geometry: bool
    low: np.ndarray
    high: np.ndarray
    area: float
    motif_evidence: dict | None


def _candidate(photo: ExtractedPhoto | PhotoRead) -> _Candidate | None:
    mask = photo.segmentation_mask or {}
    correction = (photo.enhancement_applied or {}).get("correction") or {}
    modes = correction.get("crop_modes") or {}
    if (photo.segmentation_confidence is None or photo_is_reviewed_non_print(photo)
            or mask.get("source") != "yolo_seg_mask" or mask.get("detector_source") != "yolo_onnx"
            or POSSIBLE_PRINT_REGIONS in mask
            or any(mask.get(key) for key in ("warp_quad", "visible_polygon", "merged_print_pair", "paper_grid_review",
                                             "film_grid_review_seam", "crossing_print_suggestion",
                                             "paper_join_review_traces", "fragment_outline_suggestion", "capture_edge_clipped"))
            or any((modes.get(key) or {}).get("status") != "review_needed" for key in ("full_print", "image_only"))
            or any((modes.get("full_print") or {}).get(key)
                   for key in ("suggested_warp_quad", "outline_preview"))):
        return None
    polygon = mask.get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 4:
        return None
    try:
        coordinates = [(float(point["x"]), float(point["y"])) for point in polygon]
        vertices = np.float32(coordinates)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(vertices).all() or any(not 0 <= value <= 1 for xy in coordinates for value in xy):
        return None
    area = cv2.contourArea(vertices)
    if area <= 0:
        return None
    outline = [{"x": x, "y": y} for x, y in coordinates]
    motif_evidence = mask.get(SHARED_MOTIF_REVIEW)
    return _Candidate(photo.id, outline, vertices, photo.geometry_valid, vertices.min(axis=0), vertices.max(axis=0), area,
                      motif_evidence if isinstance(motif_evidence, dict) else None)


def _overlap_bound(first: _Candidate, second: _Candidate) -> float:
    dimensions = np.minimum(first.high, second.high) - np.maximum(first.low, second.low)
    return float(np.maximum(dimensions, 0).prod())


def _find_comparisons(candidates: list[_Candidate]) -> list[tuple[_Candidate, _Candidate, _Candidate]]:
    framed = [item for item in candidates if item.valid_geometry]
    partial = [item for item in candidates if not item.valid_geometry]
    if not partial:
        return []
    matches = []
    for first, second in combinations(framed, 2):
        wider, narrower = sorted((first, second), key=lambda item: item.area, reverse=True)
        if (not .64 <= narrower.area / wider.area <= .86
                or _overlap_bound(wider, narrower) < .94 * narrower.area):
            continue
        overlap = normalized_polygon_overlap(wider.vertices, narrower.vertices, raster_size=1024)
        if not (.68 <= overlap.first_containment <= .83 and overlap.second_containment >= .96):
            continue
        extensions = []
        for third in partial:
            if _overlap_bound(wider, third) < max(.09 * wider.area, .53 * third.area):
                continue
            shared = normalized_polygon_overlap(wider.vertices, third.vertices, raster_size=1024)
            if not (.10 <= shared.first_containment <= .23 and shared.second_containment >= .55):
                continue
            if _overlap_bound(narrower, third) > 0 and normalized_polygon_overlap(
                narrower.vertices, third.vertices, raster_size=1024,
            ).iou > .005:
                continue
            extensions.append(third)
        if len(extensions) == 1:
            matches.append((wider, narrower, extensions[0]))
    return matches


def photo_outline_comparisons(
    photos: list[ExtractedPhoto | PhotoRead],
) -> dict[UUID, CompetingPhotoOutline]:
    """Build request-local comparisons from the current rows on each page."""
    pages = defaultdict(list)
    for photo in photos:
        if not photo_is_reviewed_non_print(photo):
            pages[photo.page_id].append(photo)
    result = {}
    for current in pages.values():
        candidates = [candidate for photo in current if (candidate := _candidate(photo)) is not None]
        matches = _find_comparisons(candidates)
        legacy_pairs = {(wider.id, narrower.id) for wider, narrower, _ in matches}
        measured = _measured_motif_pairs(candidates, current)
        motifs = {(wider.id, narrower.id): guide for wider, narrower, guide in measured}
        matches = [*matches, *((wider, narrower) for wider, narrower, _ in measured
                               if (wider.id, narrower.id) not in legacy_pairs)]
        counts = Counter(candidate.id for overlap in matches for candidate in overlap)
        for wider, narrower, *extension in matches:
            if any(counts[item.id] > 1 for item in (wider, narrower, *extension)):
                continue
            comparison = CompetingPhotoOutline(
                wider_photo_id=wider.id, narrower_photo_id=narrower.id,
                wider=wider.outline, narrower=narrower.outline,
                motif_guide=motifs.get((wider.id, narrower.id)),
            )
            result[wider.id] = result[narrower.id] = comparison
    return result


def _measured_motif_pairs(
    candidates: list[_Candidate], photos: list[ExtractedPhoto | PhotoRead],
) -> list[tuple[_Candidate, _Candidate, list[dict]]]:
    groups = defaultdict(list)
    for item in candidates:
        evidence = item.motif_evidence or {}
        key = evidence.get("id")
        if item.valid_geometry and isinstance(key, str):
            groups[key].append(item)
    pairs = []
    for key, group in groups.items():
        if len(group) != 2:
            continue
        wider, narrower = sorted(group, key=lambda item: item.area, reverse=True)
        evidence = wider.motif_evidence
        outline = evidence.get("outline")
        if (evidence != narrower.motif_evidence or not isinstance(outline, list) or len(outline) != 4
                or key != motif_review_id(outline, [wider.outline, narrower.outline])):
            continue
        try:
            vertices = [(float(point["x"]), float(point["y"])) for point in outline]
        except (KeyError, TypeError, ValueError):
            continue
        if any(not 0 <= value <= 1 for point in vertices for value in point):
            continue
        other_claims = [claim for photo in photos if photo.id not in (wider.id, narrower.id)
                        for claim in ((photo.segmentation_mask or {}).get(field)
                                      for field in ("polygon", "warp_quad", "visible_polygon")) if claim]
        try:
            conflict = any(normalized_polygon_overlap(vertices, [(point["x"], point["y"]) for point in claim],
                                                      raster_size=1024).first_containment > .001
                           for claim in other_claims)
        except (KeyError, TypeError, ValueError):
            continue
        if conflict:
            continue
        pairs.append((wider, narrower, outline))
    return pairs
