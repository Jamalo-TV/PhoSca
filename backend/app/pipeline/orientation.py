from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


_ROTATIONS = (0, 90, 180, 270)


@lru_cache(maxsize=2)
def _cached_orientation_session(model_path: str, modified_ns: int):
    import onnxruntime as ort

    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=4)
def _cached_cascade_classifier(cascade_path: str) -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(cascade_path)


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    normalized = angle % 360
    if normalized == 0:
        return image
    if normalized == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported right-angle rotation: {angle}")


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float32) - float(np.max(values))
    exp = np.exp(shifted)
    return exp / max(float(np.sum(exp)), 1e-6)


def _onnx_orientation_scores(image: np.ndarray, model_path: Path | None) -> dict[int, float] | None:
    if model_path is None or not model_path.exists():
        return None

    try:
        session = _cached_orientation_session(str(model_path), model_path.stat().st_mtime_ns)
        input_meta = session.get_inputs()[0]
        shape = input_meta.shape
        target_height = int(shape[2] if isinstance(shape[2], int) else 224)
        target_width = int(shape[3] if isinstance(shape[3], int) else 224)
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        tensor = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        output = np.asarray(session.run(None, {input_meta.name: tensor[np.newaxis, ...]})[0]).reshape(-1)
        if output.size < 4:
            return None
        probabilities = _softmax(output[:4])
        return {angle: float(probabilities[index]) for index, angle in enumerate(_ROTATIONS)}
    except Exception:
        return None


def _cascade_paths() -> list[str]:
    base = getattr(cv2, "data", None)
    if base is None:
        return []
    root = getattr(base, "haarcascades", "")
    return [
        str(Path(root) / "haarcascade_frontalface_default.xml"),
        str(Path(root) / "haarcascade_profileface.xml"),
    ]


def _face_score(image: np.ndarray) -> float:
    height, width = image.shape[:2]
    if height < 48 or width < 48:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_size = max(24, min(height, width) // 10)
    score = 0.0
    for cascade_path in _cascade_paths():
        classifier = _cached_cascade_classifier(cascade_path)
        if classifier.empty():
            continue
        faces = classifier.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(min_size, min_size),
        )
        for x, y, w, h in faces:
            area_ratio = (w * h) / max(1, width * height)
            upper_bias = max(0.0, 0.62 - ((y + h / 2) / height))
            score += min(1.0, area_ratio * 18.0) + upper_bias * 0.35
    return float(score)


def _detail_gravity_score(image: np.ndarray) -> float:
    height, width = image.shape[:2]
    if height < 48 or width < 48:
        return 0.0

    scale = min(1.0, 420.0 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 140).astype(np.float32) / 255.0
    edge_sum = float(edges.sum())
    if edge_sum < max(30.0, width * height * 0.002):
        return 0.0

    y_positions = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, np.newaxis]
    x_positions = np.linspace(-1.0, 1.0, width, dtype=np.float32)[np.newaxis, :]
    top_weight = np.clip(0.72 - y_positions, 0.0, 1.0)
    center_weight = 1.0 - np.clip(np.abs(x_positions), 0.0, 1.0) * 0.35
    bottom_weight = np.clip(y_positions - 0.48, 0.0, 1.0)
    top_detail = float(np.sum(edges * top_weight * center_weight)) / edge_sum
    bottom_detail = float(np.sum(edges * bottom_weight * center_weight)) / edge_sum
    return max(0.0, top_detail - bottom_detail)


def _heuristic_orientation_scores(image: np.ndarray) -> dict[int, dict[str, float]]:
    scores: dict[int, dict[str, float]] = {}
    for angle in _ROTATIONS:
        rotated = rotate_image(image, angle)
        face = _face_score(rotated)
        detail = _detail_gravity_score(rotated)
        scores[angle] = {
            "face": round(face, 4),
            "detail_gravity": round(detail, 4),
            "combined": round(face * 1.8 + detail, 4),
        }
    return scores


def correct_photo_orientation(
    image: np.ndarray,
    *,
    model_path: Path | None = None,
    min_confidence: float = 0.56,
    min_margin: float = 0.12,
) -> tuple[np.ndarray, dict]:
    model_scores = _onnx_orientation_scores(image, model_path)
    if model_scores is not None:
        ranked = sorted(model_scores.items(), key=lambda item: item[1], reverse=True)
        best_angle, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_angle and best_score >= min_confidence and best_score - runner_up >= min_margin:
            return rotate_image(image, best_angle), {
                "rotation_degrees": best_angle,
                "method": "onnx_orientation",
                "confidence": round(best_score, 4),
                "scores": {str(angle): round(score, 4) for angle, score in model_scores.items()},
            }
        return image, {
            "rotation_degrees": 0,
            "method": "onnx_orientation",
            "confidence": round(best_score, 4),
            "reason": "low_confidence_or_margin",
            "scores": {str(angle): round(score, 4) for angle, score in model_scores.items()},
        }

    heuristic_scores = _heuristic_orientation_scores(image)
    ranked = sorted(heuristic_scores.items(), key=lambda item: item[1]["combined"], reverse=True)
    best_angle, best_parts = ranked[0]
    runner_up = ranked[1][1]["combined"] if len(ranked) > 1 else 0.0
    best_score = best_parts["combined"]
    face_present = max(parts["face"] for parts in heuristic_scores.values()) > 0.0
    required_margin = 0.04 if face_present else 0.06
    required_score = 0.05 if face_present else 0.12
    metadata = {
        "rotation_degrees": 0,
        "method": "heuristic_face_detail",
        "confidence": round(best_score, 4),
        "scores": {str(angle): parts for angle, parts in heuristic_scores.items()},
    }
    if best_angle and best_score >= required_score and best_score - runner_up >= required_margin:
        metadata["rotation_degrees"] = best_angle
        return rotate_image(image, best_angle), metadata
    metadata["reason"] = "low_confidence_or_margin"
    return image, metadata
