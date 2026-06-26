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


def test_segmentation_validator_uses_polygon_iou() -> None:
    from scripts.validate_segmentation import detected_polygon, polygon_iou

    expected = [(0.1, 0.1), (0.6, 0.1), (0.6, 0.6), (0.1, 0.6)]
    smaller = [(0.2, 0.2), (0.5, 0.2), (0.5, 0.5), (0.2, 0.5)]
    box = {"x1": 0.0, "y1": 0.0, "x2": 0.9, "y2": 0.9}
    mask = {"polygon": [{"x": x, "y": y} for x, y in smaller]}

    assert polygon_iou(expected, expected) == 1.0
    assert polygon_iou(expected, smaller) < 0.5
    assert detected_polygon(box, mask) == smaller


def test_prepare_yolo_dataset_creates_non_overlapping_splits(tmp_path: Path) -> None:
    from scripts.prepare_yolo_dataset import prepare_dataset
    from scripts.validate_yolo_dataset import validate_dataset

    source_images = tmp_path / "raw"
    source_labels = tmp_path / "labels"
    source_images.mkdir()
    source_labels.mkdir()
    for index in range(1, 7):
        stem = f"page_{index:02d}"
        (source_images / f"{stem}.jpg").write_bytes(b"placeholder")
        (source_labels / f"{stem}.txt").write_text(
            "0 0.1 0.1 0.6 0.1 0.6 0.6 0.1 0.6\n",
            encoding="utf-8",
        )

    manifest = prepare_dataset(
        source_images=source_images,
        source_labels=source_labels,
        output_dataset=tmp_path / "data" / "yolo_dataset",
        golden_dir=tmp_path / "data" / "golden_fixtures",
        seed=7,
        val_count=2,
        golden_count=1,
        require_labels=True,
    )

    assert len(manifest.train) == 3
    assert len(manifest.val) == 2
    assert len(manifest.golden) == 1
    assert set(manifest.train).isdisjoint(manifest.val)
    assert set(manifest.train).isdisjoint(manifest.golden)
    assert validate_dataset(tmp_path / "data" / "data.yaml", golden_dir=tmp_path / "data" / "golden_fixtures").valid is True


def test_label_studio_export_builds_local_file_tasks(tmp_path: Path) -> None:
    from scripts.export_label_studio_tasks import export_tasks

    source_images = tmp_path / "data" / "raw_album_pages"
    source_images.mkdir(parents=True)
    (source_images / "page_01.jpg").write_bytes(b"placeholder")

    tasks = export_tasks(
        source_images=source_images,
        output=tmp_path / "data" / "label_studio_tasks.json",
        document_root=tmp_path / "data",
        url_prefix="/data/local-files/?d=",
        preannotate=False,
        min_confidence=0.45,
    )

    assert tasks == [{"data": {"image": "/data/local-files/?d=raw_album_pages/page_01.jpg"}}]


def test_label_studio_export_adds_classical_preannotations(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    from scripts.export_label_studio_tasks import export_tasks

    source_images = tmp_path / "data" / "raw_album_pages"
    source_images.mkdir(parents=True)
    image_path = source_images / "page_01.jpg"
    image = np.full((420, 560, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (80, 100), (310, 280), (20, 20, 20), 4)
    cv2.rectangle(image, (88, 108), (302, 272), (70, 120, 180), -1)
    cv2.line(image, (180, 108), (210, 272), (255, 255, 255), 2)
    cv2.imwrite(str(image_path), image)

    tasks = export_tasks(
        source_images=source_images,
        output=tmp_path / "data" / "label_studio_tasks.json",
        document_root=tmp_path / "data",
        url_prefix="/data/local-files/?d=",
        preannotate=True,
        min_confidence=0.45,
    )

    prediction = tasks[0]["predictions"][0]
    result = prediction["result"][0]
    assert prediction["model_version"] == "classical_hybrid_quad"
    assert result["type"] == "polygonlabels"
    assert result["value"]["polygonlabels"] == ["photo"]
    assert len(result["value"]["points"]) >= 4

def test_label_studio_json_conversion_writes_yolo_polygons(tmp_path: Path) -> None:
    from scripts.convert_label_studio_to_yolo import convert_tasks

    tasks = [
        {
            "data": {"image": "/data/local-files/?d=raw_album_pages/page_01.jpg"},
            "annotations": [
                {
                    "result": [
                        {
                            "type": "polygonlabels",
                            "value": {
                                "points": [[10, 20], [60, 20], [60, 70], [10, 70]],
                                "polygonlabels": ["photo"],
                            },
                        }
                    ]
                }
            ],
        }
    ]

    report = convert_tasks(tasks, output_dir=tmp_path / "labels")

    assert report.labels_written == 1
    assert report.polygons_written == 1
    assert (tmp_path / "labels" / "page_01.txt").read_text(encoding="utf-8") == (
        "0 0.100000 0.200000 0.600000 0.200000 0.600000 0.700000 0.100000 0.700000\n"
    )


def test_label_studio_json_conversion_can_use_predictions(tmp_path: Path) -> None:
    from scripts.convert_label_studio_to_yolo import convert_tasks

    tasks = [
        {
            "data": {"image": "/data/local-files/?d=raw_album_pages/page_02.jpg"},
            "predictions": [
                {
                    "result": [
                        {
                            "type": "polygonlabels",
                            "value": {
                                "points": [[15, 15], [55, 15], [55, 55], [15, 55]],
                                "polygonlabels": ["photo"],
                            },
                        }
                    ]
                }
            ],
        }
    ]

    without_predictions = convert_tasks(tasks, output_dir=tmp_path / "without", use_predictions=False)
    with_predictions = convert_tasks(tasks, output_dir=tmp_path / "with", use_predictions=True)

    assert without_predictions.missing_annotations == ["page_02"]
    assert with_predictions.labels_written == 1
    assert (tmp_path / "with" / "page_02.txt").exists()


def test_reviewed_yolo_export_groups_masks_by_page(tmp_path: Path) -> None:
    from scripts.export_reviewed_yolo_labels import export_rows

    rows = [
        {
            "original_filename": "page_01.jpg",
            "photo_id": "photo-a",
            "bounding_box": {"x1": 0.0, "y1": 0.0, "x2": 0.9, "y2": 0.9},
            "segmentation_mask": {
                "source": "manual_quad",
                "polygon": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.6, "y": 0.1},
                    {"x": 0.6, "y": 0.5},
                    {"x": 0.1, "y": 0.5},
                ],
            },
        },
        {
            "original_filename": "page_02.jpg",
            "photo_id": None,
            "bounding_box": None,
            "segmentation_mask": None,
        },
    ]

    report = export_rows(rows, album_id="album", output_dir=tmp_path / "labels")

    assert report.pages_seen == 2
    assert report.label_files_written == 1
    assert report.polygons_written == 1
    assert report.missing_pages == ["page_02.jpg"]
    assert (tmp_path / "labels" / "page_01.txt").read_text(encoding="utf-8") == (
        "0 0.100000 0.100000 0.600000 0.100000 0.600000 0.500000 0.100000 0.500000\n"
    )


def test_reviewed_yolo_export_can_require_manual_masks(tmp_path: Path) -> None:
    from scripts.export_reviewed_yolo_labels import export_rows

    rows = [
        {
            "original_filename": "page_01.jpg",
            "photo_id": "auto-photo",
            "bounding_box": {"x1": 0.1, "y1": 0.1, "x2": 0.7, "y2": 0.7},
            "segmentation_mask": {
                "source": "classical_border_quad",
                "polygon": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.7, "y": 0.1},
                    {"x": 0.7, "y": 0.7},
                    {"x": 0.1, "y": 0.7},
                ],
            },
        },
        {
            "original_filename": "page_01.jpg",
            "photo_id": "manual-photo",
            "bounding_box": {"x1": 0.2, "y1": 0.2, "x2": 0.5, "y2": 0.5},
            "segmentation_mask": {
                "source": "manual_quad",
                "polygon": [
                    {"x": 0.2, "y": 0.2},
                    {"x": 0.5, "y": 0.2},
                    {"x": 0.5, "y": 0.5},
                    {"x": 0.2, "y": 0.5},
                ],
            },
        },
    ]

    report = export_rows(rows, album_id="album", output_dir=tmp_path / "labels", manual_only=True)

    assert report.label_files_written == 1
    assert report.polygons_written == 1
    assert report.skipped_photos == ["auto-photo: mask is not manual"]
    assert (tmp_path / "labels" / "page_01.txt").read_text(encoding="utf-8").startswith("0 0.200000")


def test_yolo_dataset_validator_accepts_complete_split(tmp_path: Path) -> None:
    from scripts.validate_yolo_dataset import parse_yolo_segmentation_line, validate_dataset

    root = tmp_path / "dataset"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / f"{split}_page.jpg").write_bytes(b"placeholder")
        (root / "labels" / split / f"{split}_page.txt").write_text(
            "0 0.1 0.1 0.6 0.1 0.6 0.6 0.1 0.6\n",
            encoding="utf-8",
        )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: dataset\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['photo']\n", encoding="utf-8")

    assert len(parse_yolo_segmentation_line("0 0.1 0.1 0.6 0.1 0.6 0.6 0.1 0.6", class_count=1)) == 4
    assert validate_dataset(data_yaml, golden_dir=tmp_path / "golden").valid is True


def test_yolo_dataset_validator_reports_missing_labels(tmp_path: Path) -> None:
    from scripts.validate_yolo_dataset import validate_dataset

    root = tmp_path / "dataset"
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "images" / "train" / "page_a.jpg").write_bytes(b"placeholder")
    (root / "images" / "val" / "page_b.jpg").write_bytes(b"placeholder")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: dataset\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['photo']\n", encoding="utf-8")

    report = validate_dataset(data_yaml, golden_dir=tmp_path / "golden")

    assert report.valid is False
    assert any("missing label file" in issue.message for issue in report.issues)


def test_orientation_training_labels_are_correction_angles() -> None:
    from scripts.train_orientation_model import correction_label, ROTATIONS

    assert ROTATIONS[correction_label(0)] == 0
    assert ROTATIONS[correction_label(90)] == 270
    assert ROTATIONS[correction_label(180)] == 180
    assert ROTATIONS[correction_label(270)] == 90


@pytest.mark.skipif(not Path("data/golden_fixtures/labels").exists(), reason="Golden labels are required for IoU regression.")
def test_golden_fixture_quality_gate_placeholder() -> None:
    # The real gate is executed by scripts/validate_segmentation.py and scripts/validate_ocr.py
    # against a live database after the 10 golden fixtures are processed.
    assert Path("scripts/validate_segmentation.py").exists()
    assert Path("scripts/validate_ocr.py").exists()
