import cv2
import numpy as np
import logging
import gc
import torch
from app.pipeline.color_correction import apply_color_corrections
from app.pipeline.degradation import DegradationReport, analyze_degradation
from app.pipeline.presets import EnhancementConfig, get_enhancement_config
from app.pipeline.quality_assessment import QualityAssessor, QualityDelta
from app.utils.device import get_best_device, get_optimal_tile_size

logger = logging.getLogger(__name__)

# Global instances so we don't reload models for every request
_enhancer = None
_face_enhancer = None


def _blend_with_reference(enhanced: np.ndarray, reference: np.ndarray, reference_weight: float = 0.12) -> np.ndarray:
    if enhanced.shape[:2] != reference.shape[:2]:
        reference = cv2.resize(reference, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(enhanced, 1.0 - reference_weight, reference, reference_weight, 0)


def get_dl_models(load_faces: bool = True, tile_size: int | None = None, tile_pad: int = 10):
    """Lazy load models to avoid slowing down backend startup if not used."""
    global _enhancer, _face_enhancer
    if _enhancer is not None and (_face_enhancer is not None or not load_faces):
        return _enhancer, _face_enhancer

    device = get_best_device()

    try:
        import torchvision.transforms.functional as F
        import sys
        sys.modules['torchvision.transforms.functional_tensor'] = F

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        if _enhancer is None:
            resolved_tile_size = tile_size or get_optimal_tile_size("realesrgan")
            logger.info("Initializing Real-ESRGAN with tile=%s, tile_pad=%s...", resolved_tile_size, tile_pad)
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            _enhancer = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=resolved_tile_size,
                tile_pad=tile_pad,
                pre_pad=0,
                half=True if device.type == 'cuda' else False,
                device=device
            )

        if load_faces and _face_enhancer is None:
            from gfpgan import GFPGANer

            logger.info("Initializing GFPGAN...")
            _face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=_enhancer,
                device=device
            )
        logger.info("Deep learning models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load DL models: {e}")
        raise

    return _enhancer, _face_enhancer

def free_dl_memory():
    """Frees up VRAM when not in use."""
    global _enhancer, _face_enhancer
    _enhancer = None
    _face_enhancer = None
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

def _pre_denoise(image: np.ndarray, report: DegradationReport, config: EnhancementConfig) -> tuple[np.ndarray, dict]:
    if not config.denoise_enabled or not report.should_denoise:
        return image, {"denoise": False, "noise_level": report.noise_level, "reason": "not_recommended"}
    strength_multiplier = max(config.denoise_strength, report.denoise_strength_recommended)
    strength = max(1, min(9, int(round(2 + strength_multiplier * 8))))
    denoised = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    return denoised, {
        "denoise": True,
        "method": "fastNlMeansDenoisingColored",
        "order": "before_upscale",
        "strength": strength,
        "noise_level": report.noise_level,
    }


def _quality_gate_step(
    before: np.ndarray,
    after: np.ndarray,
    config: EnhancementConfig,
    step_name: str,
    metadata: dict,
    assessor: QualityAssessor,
) -> tuple[np.ndarray, dict]:
    if not config.quality_gating_enabled:
        return after, metadata

    delta = assessor.compare(before, after)
    gated_metadata = {
        **metadata,
        "quality_delta": delta.as_dict(),
        "quality_gate": "accepted",
    }
    if _should_revert_step(step_name, delta, config):
        gated_metadata["quality_gate"] = "reverted"
        return before, gated_metadata
    return after, gated_metadata


def _should_revert_step(step_name: str, delta: QualityDelta, config: EnhancementConfig) -> bool:
    if step_name == "denoise":
        noise_improved = delta.noise_delta < -0.4
        sharpness_drop_ratio = abs(delta.sharpness_delta) / max(delta.before.sharpness, 1.0)
        if noise_improved and sharpness_drop_ratio <= 0.45 and delta.clipping_delta < 0.04:
            return False
    if step_name == "color":
        range_improved = delta.dynamic_range_delta > 0.015
        if range_improved and delta.clipping_delta < 0.04:
            return False
    if step_name == "scratch" and delta.noise_delta <= 1.0 and delta.clipping_delta < 0.04:
        return False
    return QualityAssessor().should_revert(delta, config.min_quality_delta)


def _run_upscale_or_faces(image: np.ndarray, config: EnhancementConfig) -> tuple[np.ndarray, list[str], dict]:
    tile_size = config.tile_size or get_optimal_tile_size(config.sr_model)
    enhancer, face_enhancer = get_dl_models(
        load_faces=config.face_enhancement_enabled,
        tile_size=tile_size,
        tile_pad=config.tile_pad,
    )
    if config.face_enhancement_enabled:
        _, _, output = face_enhancer.enhance(
            image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=config.face_weight,
        )
        return output, ["Real-ESRGAN_x2plus", "GFPGANv1.3"], {"face_weight": config.face_weight, "tile_size": tile_size}

    output, _ = enhancer.enhance(image, outscale=config.upscale_factor)
    return output, ["Real-ESRGAN_x2plus"], {"face_weight": 0.0, "tile_size": tile_size}


def dl_enhance_photo(
    image: np.ndarray,
    preset: str | None = "balanced",
    degradation_report: DegradationReport | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Enhances a photo using LaMa (inpainting), Real-ESRGAN and GFPGAN.
    Replaces the legacy OpenCV DSP pipeline.
    """
    try:
        from app.pipeline.inpainting import inpaint_scratches

        report = degradation_report or analyze_degradation(image)
        config = get_enhancement_config(preset, recommended_preset=report.recommended_preset)
        working = image.copy()
        assessor = QualityAssessor()

        # Pre-clean source pixels before any 2x model so noise is not amplified.
        candidate, denoise_metadata = _pre_denoise(working, report, config)
        working, denoise_metadata = _quality_gate_step(working, candidate, config, "denoise", denoise_metadata, assessor)

        if report.should_correct_color:
            candidate, color_metadata = apply_color_corrections(working, report, config)
            working, color_metadata = _quality_gate_step(working, candidate, config, "color", color_metadata, assessor)
        else:
            color_metadata = {"color_correction": False, "reason": "not_recommended"}

        if config.scratch_removal_enabled and report.should_inpaint:
            candidate, inpaint_metadata = inpaint_scratches(
                working,
                sensitivity=config.scratch_sensitivity,
                max_mask_ratio=0.03,
            )
            inpainted_image, inpaint_metadata = _quality_gate_step(working, candidate, config, "scratch", inpaint_metadata, assessor)
        else:
            inpainted_image = working
            reason = "disabled" if not config.scratch_removal_enabled else "not_recommended"
            inpaint_metadata = {"scratch_removal": False, "method": reason, "mask_pixels": 0}

        output, model_chain, model_metadata = _run_upscale_or_faces(inpainted_image, config)
        output = _blend_with_reference(output, inpainted_image, reference_weight=config.reference_blend)

        metadata = {
            "method": "deep_learning",
            "preset": config.preset,
            "config": config.as_dict(),
            "models": ["LaMa", *model_chain],
            "device": str(get_best_device()),
            "reference_blend": config.reference_blend,
            "degradation_report": report.as_dict(),
            "steps": {
                "denoise": denoise_metadata,
                "color": color_metadata,
                "scratch": inpaint_metadata,
            },
            "scratch_removal": inpaint_metadata.get("scratch_removal", False),
            "scratch_method": inpaint_metadata.get("method"),
            "scratch_mask_pixels": inpaint_metadata.get("mask_pixels", 0),
            **model_metadata,
        }
        return output, metadata
    except Exception as e:
        logger.error(f"DL enhancement failed: {e}. Falling back to original image.")
        return image, {"error": str(e), "method": "failed"}
