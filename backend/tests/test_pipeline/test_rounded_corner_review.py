import cv2
import numpy as np

from app.pipeline.crop_sampling import _outline_mask
from app.pipeline.rounded_corner_review import measure_rounded_corners


def test_subtract_supported_wedges_and_keep_phase_mismatched_artwork(rounded_corner_page):
    raster, quad, paper, rectangle, backing = rounded_corner_page
    corners = measure_rounded_corners(raster, quad, [])
    assert corners is not None and len(corners.traces) == 4
    owned = _outline_mask(corners.outline, raster.shape[:2]).astype(bool)
    assert np.any(rectangle & ~paper & ~owned)
    assert not np.any(paper & ~owned)

    square = backing.copy()
    square[rectangle] = (135, 144, 123)
    assert measure_rounded_corners(cv2.cvtColor(square, cv2.COLOR_LAB2BGR), quad, []) is None

    # A printed area of the same hue has the shape, but no continuity in the ribs.
    artwork = backing.copy()
    artwork[paper] = (135, 144, 123)
    artwork[rectangle & ~paper] = (180, 128, 144)
    assert measure_rounded_corners(cv2.cvtColor(artwork, cv2.COLOR_LAB2BGR), quad, []) is None

    claim = np.float32([[60, 60], [90, 60], [90, 90], [60, 90]])
    clipped = measure_rounded_corners(raster, quad, [claim])
    assert clipped is not None and len(clipped.traces) == 3
    assert not any(cv2.intersectConvexConvex(cv2.convexHull(wedge), claim)[0] for wedge in clipped.wedges)
