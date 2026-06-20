from pathlib import Path

import pytest


def test_golden_fixture_annotations_are_locked() -> None:
    fixture_dir = Path("data/golden_fixtures")
    label_dir = fixture_dir / "labels"
    images = sorted(path for path in fixture_dir.glob("*.jpg"))
    labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []

    if len(images) < 10 or len(labels) < 10:
        pytest.skip("Golden fixture regression is locked after 10 images and 10 YOLO segmentation labels exist.")

    assert len(images) == 10
    assert {path.stem for path in images} == {path.stem for path in labels}


@pytest.mark.skipif(not Path("data/golden_fixtures/labels").exists(), reason="Golden labels are required for IoU regression.")
def test_golden_fixture_quality_gate_placeholder() -> None:
    # The real gate is executed by scripts/validate_segmentation.py and scripts/validate_ocr.py
    # against a live database after the 10 golden fixtures are processed.
    assert Path("scripts/validate_segmentation.py").exists()
    assert Path("scripts/validate_ocr.py").exists()

