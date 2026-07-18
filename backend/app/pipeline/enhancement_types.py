"""Typed contracts shared by faithful and model-based enhancement routes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

EnhancementRoute = Literal[
    "noop",
    "conservative_deterministic",
    "restoration_candidate",
    "face_restoration_candidate",
]


@dataclass(frozen=True)
class ImageContract:
    source_dtype: str
    source_shape: tuple[int, ...]
    working_dtype: str = "uint8"
    working_color_space: str = "sRGB"
    channel_order: str = "BGR"
    value_range: tuple[int, int] = (0, 255)
    grayscale_expanded: bool = False
    alpha_composited: bool = False
    range_conversion: str = "none"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DegradationSignal:
    score: float
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnhancementOperation:
    name: str
    params: dict[str, Any]
    reason_codes: tuple[str, ...]
    expected_effect: str
    risk: str = "low"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnhancementPlan:
    route: EnhancementRoute
    operations: tuple[EnhancementOperation, ...] = ()
    reason_codes: tuple[str, ...] = ()
    confidence: float = 0.0

    @property
    def no_op(self) -> bool:
        return self.route == "noop" or not self.operations

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["no_op"] = self.no_op
        return data


def normalize_bgr_uint8(image: np.ndarray) -> tuple[np.ndarray, ImageContract]:
    """Normalize direct-array callers to the pipeline's BGR uint8 contract.

    File decoding already produces BGR uint8.  This boundary also makes unit,
    plugin, and research callers deterministic instead of allowing OpenCV to
    fail later with an opaque channel/depth error.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("image dtype must be numeric")

    source_shape = tuple(int(value) for value in image.shape)
    source_dtype = str(image.dtype)
    working = image
    range_conversion = "none"

    if np.issubdtype(working.dtype, np.floating):
        if not np.isfinite(working).all():
            raise ValueError("image contains non-finite pixels")
        minimum = float(working.min())
        maximum = float(working.max())
        if minimum >= 0.0 and maximum <= 1.0 + 1e-6:
            working = np.rint(np.clip(working, 0.0, 1.0) * 255.0).astype(np.uint8)
            range_conversion = "float_unit_to_uint8"
        elif minimum >= 0.0 and maximum <= 255.0 + 1e-6:
            working = np.rint(np.clip(working, 0.0, 255.0)).astype(np.uint8)
            range_conversion = "float_255_to_uint8"
        else:
            raise ValueError("floating image values must be in [0,1] or [0,255]")
    elif working.dtype == np.uint16:
        working = np.rint(working.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
        range_conversion = "uint16_full_range_to_uint8"
    elif working.dtype != np.uint8:
        info = np.iinfo(working.dtype)
        if info.min < 0:
            raise ValueError("signed integer image arrays are not supported")
        working = np.rint(working.astype(np.float32) * (255.0 / float(info.max))).astype(np.uint8)
        range_conversion = f"{source_dtype}_full_range_to_uint8"

    grayscale_expanded = False
    alpha_composited = False
    if working.ndim == 2:
        working = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
        grayscale_expanded = True
    elif working.ndim == 3 and working.shape[2] == 1:
        working = cv2.cvtColor(working[:, :, 0], cv2.COLOR_GRAY2BGR)
        grayscale_expanded = True
    elif working.ndim == 3 and working.shape[2] == 2:
        gray = working[:, :, 0].astype(np.float32)
        alpha = working[:, :, 1].astype(np.float32) / 255.0
        composited = gray * alpha + 255.0 * (1.0 - alpha)
        working = cv2.cvtColor(np.rint(composited).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        grayscale_expanded = True
        alpha_composited = True
    elif working.ndim == 3 and working.shape[2] == 3:
        working = np.ascontiguousarray(working)
    elif working.ndim == 3 and working.shape[2] == 4:
        color = working[:, :, :3].astype(np.float32)
        alpha = working[:, :, 3:4].astype(np.float32) / 255.0
        working = np.rint(color * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        alpha_composited = True
    else:
        raise ValueError("image must have one, two, three, or four channels")

    return np.ascontiguousarray(working), ImageContract(
        source_dtype=source_dtype,
        source_shape=source_shape,
        grayscale_expanded=grayscale_expanded,
        alpha_composited=alpha_composited,
        range_conversion=range_conversion,
    )
