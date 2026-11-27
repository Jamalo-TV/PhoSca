import cv2
import numpy as np
import pytest

from app.pipeline.capture_background_review import ALBUM_BOUNDARY_TRACES, CAPTURE_BACKGROUND_COUNT_REVIEW, album_binding_traces
from app.pipeline.near_capture_review import quarter_turn
from app.pipeline.segmentation import SegmentationDetection, SegmentationResult, _flag_capture_background, _normalized_polygon


def _album_capture(*, binding=True, separate_print=False):
    height, width = 900, 800
    rows, columns = np.indices((height, width), dtype=np.float32)
    offset = rows - (174. - .055 * columns)
    rng = np.random.default_rng(42)
    room = np.uint8(np.clip(44 + rng.normal(0, 5, offset.shape), 0, 255))
    image = np.stack((room, room, room + 5), axis=-1)
    image[offset >= 0] = (25, 41, 165)
    image[offset >= 9] = (209, 218, 223)
    if separate_print:
        image[(offset > -17) & (offset <= -12)] = (211, 222, 226)

    proposals = [np.float32([[2, 0], [798, 1], [798, 135], [1, 176]])]
    for left, top, right, bottom, color in (
        (103, 330, 330, 654, (72, 89, 116)), (445, 232, 703, 528, (118, 93, 73)),
    ):
        image[top:bottom + 1, left:right + 1] = color
        cv2.circle(image, ((left + right) // 2, (top + bottom) // 2), 52, (110, 160, 200), -1)
        proposals.append(np.float32([[left, top], [right, top], [right, bottom], [left, bottom]]))
    if binding:
        for index in range(28):
            cv2.circle(image, (760 - index, 158 + 17 * index), 3, (61, 63, 66), -1)
    return image, [points / np.float32([width - 1, height - 1]) for points in proposals]


@pytest.mark.parametrize("turn", range(4))
def test_album_binding_uses_native_spiral_and_signed_material_evidence(turn):
    image, polygons = _album_capture()
    candidate, *peers = [quarter_turn(polygon, turn) for polygon in polygons]
    span, = album_binding_traces(np.rot90(image, turn), candidate, peers)
    native = quarter_turn(span, (-turn) % 4) * [image.shape[1] - 1, image.shape[0] - 1]
    assert np.ptp(native[:, 0]) >= .85 * image.shape[1]
    depth_in_band = native[:, 1] - (174 - .055 * native[:, 0])
    assert np.all((depth_in_band >= 0) & (depth_in_band <= 9))


@pytest.mark.parametrize("alteration", ["no_spiral", "visible_separate_print", "one_neighbor"])
def test_capture_background_abstains_when_context_is_missing_or_a_separate_line_exists(alteration):
    image, (candidate, *peers) = _album_capture(
        binding=alteration != "no_spiral", separate_print=alteration == "visible_separate_print",
    )
    if alteration == "one_neighbor":
        peers.pop()
    assert album_binding_traces(image, candidate, peers) == []


def test_binding_annotation_does_not_change_active_claims_even_if_the_exterior_is_a_print():
    image, polygons = _album_capture()
    detections = [SegmentationDetection(
        {}, {"detector_source": "yolo_onnx", "source": "yolo_seg_mask", "polygon": _normalized_polygon(points, 1, 1),
             "fusion": {"proposal_families": ["model"]}},
        .9, 4.5 if index == 0 else .7, index != 0, ["aspect_ratio_too_wide"] if index == 0 else [],
    ) for index, points in enumerate(polygons)]
    original = SegmentationResult(detections, {"fusion": {"deferred_print_proposals": []}})

    # The pixels could also depict a flush print ending behind the binding.
    # Optical context can annotate the claim, while only a review decision removes it.
    marked = _flag_capture_background(image, original)
    assert marked.metadata is original.metadata
    assert marked.detections[1:] == original.detections[1:]
    candidate = marked.detections[0]
    assert candidate.mask["polygon"] == original.detections[0].mask["polygon"]
    assert candidate.confidence == .9 and not candidate.geometry_valid
    assert candidate.review_reasons == ["aspect_ratio_too_wide", CAPTURE_BACKGROUND_COUNT_REVIEW]
    assert ALBUM_BOUNDARY_TRACES in candidate.mask and ALBUM_BOUNDARY_TRACES not in original.detections[0].mask
