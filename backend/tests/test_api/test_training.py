import json
from urllib.parse import quote

import cv2
import numpy as np


def png_bytes() -> bytes:
    image = np.full((220, 300, 3), 242, dtype=np.uint8)
    cv2.rectangle(image, (45, 45), (245, 170), (250, 250, 250), -1)
    cv2.rectangle(image, (58, 58), (232, 157), (60, 120, 180), -1)
    cv2.rectangle(image, (45, 45), (245, 170), (35, 35, 35), 3)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


async def test_segmentation_training_upload_and_save_labels(client) -> None:
    response = await client.get("/api/v1/training/segmentation")
    assert response.status_code == 200
    assert response.json()["images"] == []

    upload = await client.post(
        "/api/v1/training/segmentation/images",
        files=[("files", ("training-page.png", png_bytes(), "image/png"))],
    )
    assert upload.status_code == 201, upload.text
    image = upload.json()["images"][0]
    assert image["name"] == "training-page.png"
    assert image["width"] == 300
    assert image["height"] == 220

    served = await client.get("/api/v1/training/segmentation/images/training-page.png")
    assert served.status_code == 200
    assert served.headers["content-type"].startswith("image/png")

    labels = await client.put(
        "/api/v1/training/segmentation/images/training-page.png/labels",
        json={
            "polygons": [
                {
                    "points": [
                        {"x": 0.15, "y": 0.20},
                        {"x": 0.82, "y": 0.20},
                        {"x": 0.82, "y": 0.77},
                        {"x": 0.15, "y": 0.77},
                    ]
                }
            ]
        },
    )
    assert labels.status_code == 200, labels.text
    assert len(labels.json()["polygons"]) == 1

    listing = await client.get("/api/v1/training/segmentation")
    assert listing.status_code == 200
    assert listing.json()["images"][0]["labels"] == 1
    assert listing.json()["images"][0]["has_label"] is True


async def test_segmentation_training_preannotate_endpoint(client) -> None:
    upload = await client.post(
        "/api/v1/training/segmentation/images",
        files=[("files", ("detect-page.png", png_bytes(), "image/png"))],
    )
    assert upload.status_code == 201, upload.text

    response = await client.post("/api/v1/training/segmentation/images/detect-page.png/detect")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "metadata" in body
    assert isinstance(body["polygons"], list)


async def test_training_suggestions_are_current_and_yield_to_saved_labels(client, tmp_path) -> None:
    filename = "café #1.png"
    base_url = "/api/v1/training/segmentation/images"
    upload = await client.post(base_url, files=[("files", (filename, png_bytes(), "image/png"))])
    assert upload.status_code == 201, upload.text
    preview = await client.get(upload.json()["images"][0]["url"])
    assert preview.status_code == 200, preview.text
    labels_url = f"{base_url}/{quote(filename)}/labels"
    workspace = tmp_path / "segmentation_training"
    draft_path = workspace / "exports" / "detection_suggestions.json"
    image_stat = (workspace / "images" / filename).stat()
    points = [
        {"x": 0.15, "y": 0.20},
        {"x": 0.82, "y": 0.20},
        {"x": 0.82, "y": 0.77},
        {"x": 0.15, "y": 0.77},
    ]
    suggestion = {"points": points, "source": "yolo", "confidence": 0.9}
    draft = {"mtime_ns": image_stat.st_mtime_ns - 1, "bytes": image_stat.st_size, "polygons": [suggestion]}

    missing = await client.get(labels_url)
    assert missing.status_code == 200, missing.text
    assert "suggestions" not in missing.json()

    draft_path.write_text(json.dumps({filename: draft}), encoding="utf-8")
    stale = await client.get(labels_url)
    assert stale.status_code == 200, stale.text
    assert "suggestions" not in stale.json()

    draft["mtime_ns"] = image_stat.st_mtime_ns
    draft_path.write_text(json.dumps({filename: draft}), encoding="utf-8")
    current = await client.get(labels_url)
    assert current.status_code == 200, current.text
    current_body = current.json()
    assert current_body["polygons"] == []
    assert current_body["image"]["has_label"] is False
    assert current_body["suggestions"] == [suggestion]

    saved = await client.put(labels_url, json={"polygons": [{"points": points}]})
    assert saved.status_code == 200, saved.text
    reviewed = await client.get(labels_url)
    assert reviewed.status_code == 200, reviewed.text
    reviewed_body = reviewed.json()
    assert reviewed_body["polygons"] == [{"points": points, "source": "manual_yolo"}]
    assert reviewed_body["image"]["has_label"] is True
    assert "suggestions" not in reviewed_body
