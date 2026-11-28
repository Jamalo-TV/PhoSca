"""Measure crop ownership on two separate prints with a pale album gap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.evaluation.cropping import _source_center_footprint
from app.pipeline.crop_sampling import warp_owned_crop
from app.pipeline.perspective import correct_photo_crop


SIZE = (720, 1000)


def rectangle(left, top, right, bottom):
    return np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])


def source_mask(quad):
    mask = np.zeros(SIZE, np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 1)
    return mask.astype(bool)


def proposal(quad):
    points = quad / np.array([SIZE[1] - 1, SIZE[0] - 1])
    return {
        "source": "yolo_seg_mask",
        "detector_source": "yolo_onnx",
        "polygon": [{"x": float(x), "y": float(y)} for x, y in points],
    }


def paint_motif(image, quad, seed):
    left, top = quad[0].astype(int)
    right, bottom = quad[2].astype(int) + 1
    texture = np.random.default_rng(seed).integers(39, 154, (13, 9, 3), dtype=np.uint8)
    image[top:bottom, left:right] = cv2.resize(texture, (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)


def fixture(gap):
    target = rectangle(82, 93, 439, 626)
    motif = rectangle(101, 112, 420, 607)
    neighbor = rectangle(440 + gap, 82, 907, 639)
    image = np.full((*SIZE, 3), (185, 191, 196), np.uint8)
    image[source_mask(target)] = (188, 194, 199)
    image[source_mask(neighbor)] = (239, 241, 245)
    paint_motif(image, motif, seed=42)
    paint_motif(image, rectangle(463 + gap, 110, 878, 617), seed=47)
    seed = target.copy()
    seed[1:3, 0] = round(439 + gap * .74)
    return image, target, motif, neighbor, seed


def mode_measurement(mode, reference, other):
    result = {"status": mode["status"], "reasons": mode["reason_codes"]}
    if "page_to_output" not in mode:
        return result
    size = mode["shape"]["width"], mode["shape"]["height"]
    matrix = np.asarray(mode["page_to_output"])
    observed, corners = _source_center_footprint(SIZE, matrix, size[::-1])
    paper = source_mask(reference)
    peer = source_mask(other)
    external, supported = warp_owned_crop(np.float32(~paper), matrix, size)
    if not supported:
        raise ValueError("Unable to measure ownership for this transform")
    result.update({
        "warp_quad_native": corners.tolist(),
        "corner_error_native_px": np.linalg.norm(corners - reference, axis=1).tolist(),
        "footprint_iou": float(np.count_nonzero(observed & paper) / np.count_nonzero(observed | paper)),
        "omitted_reference_centres": int(np.count_nonzero(paper & ~observed)),
        "outside_reference_centres": int(np.count_nonzero(observed & ~paper)),
        "neighbor_centres": int(np.count_nonzero(observed & peer)),
        "outputs_with_nonzero_foreign_weight": int(np.count_nonzero(external > 1e-6)),
    })
    return result


def main(output_dir):
    cv2.setNumThreads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for gap in (2, 5, 10, 16):
        image, target, motif, neighbor, seed = fixture(gap)
        crop = correct_photo_crop(
            image, {}, proposal(seed), dual_mode=True, auto_orient=False,
            protected_regions=[], neighbor_masks=[proposal(neighbor)], enable_classical_retry=True,
        )
        modes = crop.metadata["crop_modes"]
        report[gap] = {
            "supplied_target_and_neighbor_separately": True,
            "true_separation_px": gap,
            "full_print": mode_measurement(modes["full_print"], target, neighbor),
            "image_only": mode_measurement(modes["image_only"], motif, neighbor),
        }
        if gap == 16:
            cv2.imwrite(str(output_dir / "authored-source.png"), image)
            cv2.imwrite(str(output_dir / "authored-whole.png"), crop.rectified_untrimmed)
            if crop.image_only is not None:
                cv2.imwrite(str(output_dir / "authored-motif.png"), crop.image_only)
    (output_dir / "neighbor-crop.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args().output_dir)
