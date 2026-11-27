import cv2
import numpy as np

from app.pipeline.crop_sampling import _outline_mask
from app.pipeline.geometry import pixel_vertices_to_normalized
from app.pipeline.perspective import correct_photo_crop
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


def test_manual_contrast_outline_holds_when_the_measured_edge_is_outside_the_saved_quad(rounded_corner_page):
    _, quad, _, rectangle, backing = rounded_corner_page
    lab = np.full_like(backing, (194, 128, 128))
    texture = np.empty((rectangle.sum(), 3), np.uint8)
    texture[:, 0] = np.random.default_rng(42).integers(72, 123, size=len(texture))
    texture[:, 1:] = (144, 115)
    lab[rectangle] = texture
    source = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def crop(native):
        corners = [{"x": float(x), "y": float(y)}
                   for x, y in pixel_vertices_to_normalized(native, *source.shape[1::-1])]
        return correct_photo_crop(source, {}, {"source": "manual_quad", "polygon": corners, "warp_quad": corners},
                                  dual_mode=True, verify_manual_contrast_outline=True,
                                  protected_regions=[], auto_orient=False)

    inset = quad + np.float32([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    held = crop(inset)
    assert held.metadata["manual_contrast_outline_check"] == 1
    modes = held.metadata["crop_modes"]
    assert all(mode["status"] == "review_needed" and "PRINT_EDGE_CONFIRMATION_REQUIRED" in mode["reason_codes"]
               for mode in modes.values())
    scale = np.float32(source.shape[1::-1]) - 1
    saved, suggested = (np.float32([[p["x"], p["y"]] for p in modes["full_print"][key]]) * scale
                        for key in ("warp_quad", "suggested_warp_quad"))
    np.testing.assert_allclose(saved, inset, atol=.01)
    np.testing.assert_allclose(suggested, quad, atol=.1)
    assert all(mode["status"] == "ready" for mode in crop(suggested).metadata["crop_modes"].values())
