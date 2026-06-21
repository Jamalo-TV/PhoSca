from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class EnhancementConfig:
    preset: str = "balanced"
    denoise_enabled: bool = True
    denoise_strength: float = 0.35
    color_correction_enabled: bool = True
    yellowing_removal_strength: float = 0.55
    fading_recovery_strength: float = 0.35
    auto_white_balance: bool = True
    scratch_removal_enabled: bool = True
    scratch_sensitivity: float = 0.5
    face_enhancement_enabled: bool = True
    face_weight: float = 0.32
    upscale_factor: int = 2
    reference_blend: float = 0.14
    sharpen_amount: float = 0.14
    sr_model: str = "realesrgan"
    face_model: str = "gfpgan"
    denoiser_model: str = "opencv"
    codeformer_fidelity: float = 0.6
    quality_gating_enabled: bool = True
    min_quality_delta: float = -0.03
    tile_size: int = 0
    tile_pad: int = 10

    def as_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, EnhancementConfig] = {
    "light": EnhancementConfig(
        preset="light",
        denoise_strength=0.18,
        yellowing_removal_strength=0.25,
        fading_recovery_strength=0.18,
        scratch_sensitivity=0.35,
        face_weight=0.22,
        reference_blend=0.18,
        sharpen_amount=0.08,
        codeformer_fidelity=0.8,
        min_quality_delta=-0.015,
    ),
    "balanced": EnhancementConfig(),
    "aggressive": EnhancementConfig(
        preset="aggressive",
        denoise_strength=0.58,
        yellowing_removal_strength=0.75,
        fading_recovery_strength=0.55,
        scratch_sensitivity=0.72,
        face_weight=0.42,
        reference_blend=0.10,
        sharpen_amount=0.22,
        codeformer_fidelity=0.4,
        min_quality_delta=-0.045,
    ),
    "archival": EnhancementConfig(
        preset="archival",
        denoise_strength=0.24,
        yellowing_removal_strength=0.28,
        fading_recovery_strength=0.22,
        scratch_sensitivity=0.45,
        face_enhancement_enabled=False,
        face_weight=0.0,
        reference_blend=0.24,
        sharpen_amount=0.06,
        codeformer_fidelity=0.85,
        min_quality_delta=-0.01,
    ),
    "speed": EnhancementConfig(
        preset="speed",
        denoise_enabled=False,
        color_correction_enabled=True,
        scratch_removal_enabled=False,
        face_enhancement_enabled=False,
        reference_blend=0.18,
        sharpen_amount=0.04,
        quality_gating_enabled=False,
        tile_size=256,
    ),
}


PRESET_DESCRIPTIONS: dict[str, str] = {
    "light": "Minimal cleanup with maximum source preservation.",
    "balanced": "Default archival restoration for mixed album photos.",
    "aggressive": "Stronger restoration for noisy, faded, or damaged photos.",
    "archival": "Conservative cleanup and upscale without face restoration.",
    "speed": "Fast upscale path with light color cleanup and no face or scratch model.",
}


def available_presets() -> dict[str, dict]:
    return {
        name: {"description": PRESET_DESCRIPTIONS[name], "config": config.as_dict()}
        for name, config in PRESETS.items()
    }


def get_enhancement_config(preset: str | None, recommended_preset: str = "balanced") -> EnhancementConfig:
    selected = recommended_preset if preset in {None, "", "auto"} else preset
    if selected not in PRESETS:
        selected = "balanced"
    return replace(PRESETS[selected])
