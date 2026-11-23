import cv2
import numpy as np

from app.pipeline.segmentation_filter import paper_and_edge_rejection_reasons


def _rectangle(x1, y1, x2, y2):
    return [{"x": x, "y": y} for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]


def test_rejects_paper_and_border_slivers_while_retaining_bright_prints_and_panoramas():
    page = np.full((600, 800, 3), (173, 205, 225), dtype=np.uint8)
    cv2.putText(page, "Caption", (90, 169), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (68, 80, 83), 1)
    page[140:350, 421:721] = 238
    page[:40, 61:733] = (117, 126, 136)
    polygons = [
        _rectangle(0.05, 0.18, 0.45, 0.40),
        _rectangle(0.53, 0.24, 0.90, 0.58),
        _rectangle(0.08, 0.0, 0.91, 0.065),
        _rectangle(0.06, 0.989, 0.81, 1.0),
        _rectangle(0.0, 0.19, 0.009, 0.80),
    ]

    assert paper_and_edge_rejection_reasons(page, polygons) == {
        0: "paper_colored_region",
        3: "thin_image_edge_fragment",
        4: "thin_image_edge_fragment",
    }


def test_retains_uniform_neutral_subject_when_paper_color_is_ambiguous():
    page = np.full((300, 420, 3), 244, dtype=np.uint8)
    assert paper_and_edge_rejection_reasons(page, [_rectangle(0.15, 0.15, 0.55, 0.55)]) == {}


def test_retains_pale_sepia_print_against_dark_album_backing():
    row, col = np.indices((600, 800))
    fade = ((row + col) % 25).astype(np.uint8)
    page = np.array((37, 35, 107), dtype=np.uint8) + fade[:, :, None]
    page[23:572, 100:472] = (174, 188, 204)
    cv2.ellipse(page, (310, 210), (25, 65), 0, 0, 360, (94, 112, 130), 2)

    assert paper_and_edge_rejection_reasons(page, [_rectangle(0.125, 0.04, 0.59, 0.953)]) == {}


def test_rejects_empty_ink_rectangle_and_retains_faint_or_sparse_prints():
    height, width = 390, 700
    page = np.full((height, width, 3), 195, np.uint8)
    polygons = []
    top, bottom = 100, 251
    for index, left in enumerate((72, 281, 490)):
        right = left + 125
        if index == 1:
            page[top:bottom + 1, left:right + 1] = 198
        if index == 2:
            cv2.circle(page, (left + 62, 176), 4, (133, 132, 130), -1)
        cv2.rectangle(page, (left, top), (right, bottom), (79, 85, 94), 4)
        polygons.append(_rectangle(left / (width - 1), top / (height - 1), right / (width - 1), bottom / (height - 1)))

    # The source still contains the texture and stock transitions that preparation can change.
    prepared = np.full_like(page, 195)
    assert paper_and_edge_rejection_reasons(prepared, polygons, source_image=page) == {0: "empty_drawn_frame"}
