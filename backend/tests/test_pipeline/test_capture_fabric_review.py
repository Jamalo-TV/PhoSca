import cv2
import numpy as np

from app.pipeline.capture_fabric_review import capture_fabric_traces


def test_exposed_texture_suggests_review_and_a_visible_paper_rim_vetoes_it():
    rng = np.random.default_rng(180)
    lightness = rng.integers(30, 220, (820, 750), dtype=np.uint8)
    capture = cv2.cvtColor(lightness, cv2.COLOR_GRAY2BGR)
    capture[155:780, 65:700] = (153, 203, 233)
    outline = np.float32([[55, 0], [620, 0], [620, 108], [55, 108]])

    traces = capture_fabric_traces(capture, [outline])
    assert traces.keys() == {0}
    assert all(trace.shape == (2, 2) for trace in traces[0])

    # Use the same textured scene and leave one visible paper edge.
    capture[:108, 618:620] = 50
    assert capture_fabric_traces(capture, [outline]) == {}
    success, encoded = cv2.imencode(".jpg", capture, [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert success
    assert capture_fabric_traces(cv2.imdecode(encoded, cv2.IMREAD_COLOR), [outline]) == {}
