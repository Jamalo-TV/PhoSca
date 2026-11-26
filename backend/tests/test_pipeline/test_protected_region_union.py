import numpy as np

from app.pipeline.protected_regions import merge_protected_regions, removed_protected_regions


def test_protected_region_union_keeps_semantic_types_and_best_duplicate():
    face = {"type": "face", "source": "a", "confidence": 0.5, "box": {"x1": 1, "y1": 1, "x2": 10, "y2": 10}}
    better_face = {"type": "face", "source": "b", "confidence": 0.9, "box": {"x1": 1, "y1": 1, "x2": 10, "y2": 10}}
    text = {"type": "text", "source": "ocr", "box": {"x1": 2, "y1": 2, "x2": 8, "y2": 8}}
    result = merge_protected_regions([face], [better_face, text])
    assert len(result) == 2
    assert {item["type"] for item in result} == {"face", "text"}
    assert next(item for item in result if item["type"] == "face")["source"] == "b"


def test_page_face_duplicate_cannot_merge_away_a_pinned_record():
    proposal = {"type": "face", "source": "opencv_haarcascade_frontalface_default", "confidence": .75,
                "box": {"x1": 1, "y1": 1, "x2": 10, "y2": 10}}
    pinned = {**proposal, "pinned": True, "confidence": .60}
    assert len(merge_protected_regions([proposal], [pinned])) == 2


def test_removed_protection_follows_the_visible_paper_around_a_concavity():
    prior = np.float32([[20, 20], [90, 20], [90, 90], [20, 90]])
    edited = np.float32([[20, 20], [55, 20], [55, 90], [20, 90]])
    visible = np.float32([[20, 20], [90, 20], [90, 38], [45, 38], [45, 62], [90, 62], [90, 90], [20, 90]])
    exterior = {"type": "text", "source": "manual", "box": {"x1": 58, "y1": 45, "x2": 70, "y2": 55}}
    crossing = {"type": "text", "source": "manual", "box": {"x1": 3, "y1": 30, "x2": 99, "y2": 34}}
    for outline in (visible, visible[::-1].copy()):
        assert removed_protected_regions(prior, edited, [exterior, crossing], width=100, height=100, visible=outline) == [
            {"x1": 3 / 99, "y1": 30 / 99, "x2": 1, "y2": 34 / 99},
        ]
