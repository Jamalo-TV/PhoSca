import torch
import logging

logger = logging.getLogger(__name__)


def get_available_vram() -> int:
    """Return currently available CUDA VRAM in MB, or 0 when unavailable."""
    if not torch.cuda.is_available():
        return 0
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        return int(free_bytes / (1024 * 1024))
    except Exception as exc:  # noqa: BLE001 - device probing should never break startup.
        logger.debug("Unable to query available CUDA VRAM: %s", exc)
        return 0


def get_total_vram() -> int:
    """Return total CUDA VRAM in MB, or 0 when unavailable."""
    if not torch.cuda.is_available():
        return 0
    try:
        properties = torch.cuda.get_device_properties(0)
        return int(properties.total_memory / (1024 * 1024))
    except Exception as exc:  # noqa: BLE001 - device probing should never break startup.
        logger.debug("Unable to query total CUDA VRAM: %s", exc)
        return 0


def get_optimal_tile_size(model_type: str = "realesrgan") -> int:
    """Pick a conservative tile size based on free VRAM."""
    available = get_available_vram()
    if available <= 0:
        return 256

    model_type = model_type.lower()
    if model_type in {"hat", "dat", "transformer_sr"}:
        if available >= 10000:
            return 384
        if available >= 6000:
            return 256
        return 160

    if available >= 8000:
        return 512
    if available >= 4000:
        return 400
    return 256


def get_best_device() -> torch.device:
    """Detects and returns the best available PyTorch device."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info("CUDA is available. Using GPU: %s (%s MB VRAM, %s MB free).", device_name, get_total_vram(), get_available_vram())
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple MPS is available. Using Apple Silicon GPU.")
        return torch.device("mps")
    else:
        logger.warning("No GPU found. Falling back to CPU. Inference will be slow.")
        return torch.device("cpu")
