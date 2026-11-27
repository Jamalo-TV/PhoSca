import cv2
import numpy as np

from app.pipeline.contrast_crossing_review import contrast_crossing_traces


def _rectangle(left, top, right, bottom):
    return np.float32([(left, top), (right, top), (right, bottom), (left, bottom)])


def test_contrast_advisory_requires_two_bracketed_areas_with_continuous_detail():
    source = np.full((1000, 1000, 3), 220, dtype=np.uint8)
    upper = _rectangle(445, 115, 899, 419)
    left = _rectangle(110, 467, 479, 848)
    source[115:419, 445:899] = (99, 111, 115)
    source[467:848, 110:479] = (99, 111, 115)
    for x in range(457, 887, 26):
        cv2.line(source, (x, 123), (x, 411), (130, 143, 160), 3)
    for y in range(479, 836, 20):
        cv2.line(source, (118, y), (471, y), (130, 143, 160), 3)
    proposal = _rectangle(300, 270, 770, 780)
    observations = [(0, upper, .88), (3, left, .90)]

    spans = contrast_crossing_traces(source, proposal, observations)
    assert len(spans) == 2
    np.testing.assert_allclose(spans, [[[300, 772], [300, 483]], [[461, 270], [762, 270]]], atol=1.)
    assert contrast_crossing_traces(source, proposal, observations[:1]) == []
    blurred = cv2.GaussianBlur(source, (57, 57), 12)
    assert contrast_crossing_traces(blurred, proposal, observations) == []

    foreground = source.copy()
    cv2.polylines(foreground, [proposal.astype(np.int32)], True, (20, 20, 20), 3)
    requested = []

    def regional_observations():
        requested.append(True)
        yield from observations

    assert contrast_crossing_traces(foreground, proposal, regional_observations()) == []
    assert not requested
