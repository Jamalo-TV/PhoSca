from __future__ import annotations

import gc
import hashlib
import logging
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from app.pipeline.degradation import DegradationReport, analyze_degradation
from app.pipeline.enhancement_types import normalize_bgr_uint8
from app.pipeline.presets import get_enhancement_config
from app.pipeline.quality_critic import evaluate_candidate
from app.utils.device import get_best_device, get_optimal_tile_size

logger = logging.getLogger(__name__)
_enhancer = None
_face_enhancer = None
_model_cache_key: tuple | None = None


def _path_identity(path: Path | None) -> tuple[str | None, int | None, int | None]:
    if path is None or not path.is_file():
        return None, None, None
    stat = path.stat()
    return str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)


def _array_sha256(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(np.asarray(image.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _blend_with_reference(enhanced: np.ndarray, reference: np.ndarray, reference_weight: float = 0.12) -> np.ndarray:
    if enhanced.shape[:2] != reference.shape[:2]:
        reference = cv2.resize(reference, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(enhanced, 1.0 - reference_weight, reference, reference_weight, 0)


def get_dl_models(
    load_faces: bool = True,
    tile_size: int | None = None,
    tile_pad: int = 10,
    realesrgan_model_path: Path | None = None,
    gfpgan_model_path: Path | None = None,
):
    """Load a cache keyed by artifact identity and inference configuration."""
    global _enhancer, _face_enhancer, _model_cache_key
    if realesrgan_model_path is None or not realesrgan_model_path.is_file():
        raise RuntimeError("Local, verified Real-ESRGAN weights are required; runtime model downloads are disabled.")
    if load_faces and (gfpgan_model_path is None or not gfpgan_model_path.is_file()):
        raise RuntimeError("Local, verified GFPGAN weights are required; runtime model downloads are disabled.")

    device = get_best_device()
    resolved_tile_size = tile_size or get_optimal_tile_size("realesrgan")
    cache_key = (
        _path_identity(realesrgan_model_path),
        _path_identity(gfpgan_model_path) if load_faces else None,
        bool(load_faces),
        int(resolved_tile_size),
        int(tile_pad),
        str(device),
    )
    if _model_cache_key == cache_key and _enhancer is not None and (_face_enhancer is not None or not load_faces):
        return _enhancer, _face_enhancer
    if _model_cache_key is not None and _model_cache_key != cache_key:
        free_dl_memory()

    try:
        import sys

        import torchvision.transforms.functional as functional

        sys.modules["torchvision.transforms.functional_tensor"] = functional
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        logger.info("Initializing Real-ESRGAN with tile=%s, tile_pad=%s", resolved_tile_size, tile_pad)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        _enhancer = RealESRGANer(
            scale=2,
            model_path=str(realesrgan_model_path),
            model=model,
            tile=resolved_tile_size,
            tile_pad=tile_pad,
            pre_pad=0,
            half=device.type == "cuda",
            device=device,
        )
        _face_enhancer = None
        if load_faces:
            from gfpgan import GFPGANer

            _face_enhancer = GFPGANer(
                model_path=str(gfpgan_model_path),
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=_enhancer,
                device=device,
            )
        _model_cache_key = cache_key
    except Exception:
        _enhancer = None
        _face_enhancer = None
        _model_cache_key = None
        logger.exception("Failed to load deep enhancement models")
        raise
    return _enhancer, _face_enhancer


def free_dl_memory() -> None:
    """Free cached model state and accelerator allocations."""
    global _enhancer, _face_enhancer, _model_cache_key
    _enhancer = None
    _face_enhancer = None
    _model_cache_key = None
    try:
        import app.pipeline.inpainting as inpainting

        inpainting._lama = None
    except ImportError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _restoration_recommended(image: np.ndarray, report: DegradationReport) -> tuple[bool, list[str]]:
    minimum_dimension = min(image.shape[:2])
    reasons: list[str] = []
    if minimum_dimension < 900:
        reasons.append("LOW_SOURCE_RESOLUTION")
    blur = report.signals.get("blur")
    if minimum_dimension < 1600 and blur and blur.score >= 0.42 and blur.confidence >= 0.62:
        reasons.append("SUPPORTED_BLUR_AND_RESOLUTION")
    if minimum_dimension < 1600 and report.compression_artifacts >= 0.22:
        reasons.append("COMPRESSION_AND_RESOLUTION")
    return bool(reasons), reasons


def dl_enhance_photo(
    image: np.ndarray,
    preset: str | None = "balanced",
    degradation_report: DegradationReport | None = None,
    model_paths: dict[str, Path] | None = None,
    *,
    allow_model_restoration: bool = False,
    base_image: np.ndarray | None = None,
    model_artifacts: dict[str, dict] | None = None,
) -> tuple[np.ndarray, dict]:
    """Produce an explicitly opted-in restoration candidate.

    This function no longer doubles as the deterministic enhancement route.
    It never marks a model-derived output for automatic release.
    """
    started = perf_counter()
    source, contract = normalize_bgr_uint8(image)
    report = degradation_report or analyze_degradation(source)
    config = get_enhancement_config(preset, recommended_preset=report.recommended_preset)
    if not allow_model_restoration:
        return source.copy(), {
            "method": "no_op",
            "variant_kind": "restoration_candidate",
            "models": [],
            "config": config.as_dict(),
            "degradation_report": report.as_dict(),
            "input_contract": contract.as_dict(),
            "planner": {
                "route": "noop",
                "no_op": True,
                "allowed_steps": [],
                "unsafe_models_enabled": False,
                "reason_codes": ["MODEL_RESTORATION_NOT_EXPLICITLY_ENABLED"],
            },
            "runtime_ms": round((perf_counter() - started) * 1000.0, 3),
            "input_pixel_sha256": _array_sha256(source),
            "output_pixel_sha256": _array_sha256(source),
            "auto_release": False,
        }

    config = replace(config, super_resolution_enabled=True, unsafe_models_enabled=True)
    recommended, route_reasons = _restoration_recommended(source, report)
    if not recommended:
        metadata = {
            "method": "no_op",
            "variant_kind": "restoration_candidate",
            "models": [],
            "config": config.as_dict(),
            "degradation_report": report.as_dict(),
            "input_contract": contract.as_dict(),
            "planner": {
                "route": "noop",
                "no_op": True,
                "allowed_steps": [],
                "unsafe_models_enabled": True,
                "reason_codes": ["NO_SUPPORTED_MODEL_DEGRADATION"],
            },
            "runtime_ms": round((perf_counter() - started) * 1000.0, 3),
            "input_pixel_sha256": _array_sha256(source),
            "output_pixel_sha256": _array_sha256(source),
            "auto_release": False,
        }
        return source.copy(), metadata

    try:
        working = normalize_bgr_uint8(base_image)[0] if base_image is not None else source.copy()
        use_faces = bool(config.face_enhancement_enabled and report.should_enhance_faces)
        tile_size = config.tile_size or get_optimal_tile_size(config.sr_model)
        enhancer, face_enhancer = get_dl_models(
            load_faces=use_faces,
            tile_size=tile_size,
            tile_pad=config.tile_pad,
            realesrgan_model_path=(model_paths or {}).get("realesrgan"),
            gfpgan_model_path=(model_paths or {}).get("gfpgan"),
        )
        if use_faces:
            _, _, output = face_enhancer.enhance(
                working,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=config.face_weight,
            )
            model_chain = ["Real-ESRGAN_x2plus", "GFPGANv1.3"]
            variant_kind = "face_restoration_candidate"
        else:
            output, _ = enhancer.enhance(working, outscale=config.upscale_factor)
            model_chain = ["Real-ESRGAN_x2plus"]
            variant_kind = "restoration_candidate"
        output, _ = normalize_bgr_uint8(output)
        output = _blend_with_reference(output, working, reference_weight=config.reference_blend)
        metadata = {
            "method": "deep_learning_restoration",
            "variant_kind": variant_kind,
            "generative": True,
            "preset": config.preset,
            "config": config.as_dict(),
            "models": model_chain,
            "model_artifacts": dict(model_artifacts or {}),
            "device": str(get_best_device()),
            "tile_size": tile_size,
            "tile_pad": config.tile_pad,
            "reference_blend": config.reference_blend,
            "degradation_report": report.as_dict(),
            "input_contract": contract.as_dict(),
            "planner": {
                "route": variant_kind,
                "no_op": False,
                "allowed_steps": ["super_resolution"] + (["face_restoration"] if use_faces else []),
                "unsafe_models_enabled": True,
                "reason_codes": route_reasons,
            },
            "runtime_ms": round((perf_counter() - started) * 1000.0, 3),
            "input_pixel_sha256": _array_sha256(source),
            "output_pixel_sha256": _array_sha256(output),
            "auto_release": False,
        }
        metadata["gates"] = evaluate_candidate(source, output, metadata)
        return output, metadata
    except Exception as exc:  # noqa: BLE001 - optional restoration must fail closed.
        logger.error("DL enhancement failed: %s. Falling back to source pixels.", exc)
        return source.copy(), {
            "error": str(exc),
            "method": "failed",
            "variant_kind": "restoration_candidate",
            "models": [],
            "model_artifacts": dict(model_artifacts or {}),
            "degradation_report": report.as_dict(),
            "input_contract": contract.as_dict(),
            "planner": {
                "route": "restoration_candidate",
                "no_op": True,
                "allowed_steps": [],
                "unsafe_models_enabled": True,
                "reason_codes": ["MODEL_RESTORATION_FAILED"],
            },
            "runtime_ms": round((perf_counter() - started) * 1000.0, 3),
            "input_pixel_sha256": _array_sha256(source),
            "output_pixel_sha256": _array_sha256(source),
            "auto_release": False,
        }
