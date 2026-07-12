from __future__ import annotations

import numpy as np
import pytest


def test_deep_model_loader_never_uses_runtime_downloads(tmp_path) -> None:
    from app.pipeline.dl_enhancement import get_dl_models

    with pytest.raises(RuntimeError, match="runtime model downloads are disabled"):
        get_dl_models(realesrgan_model_path=tmp_path / "missing.pth", gfpgan_model_path=tmp_path / "missing-face.pth")


def test_disabled_model_path_is_noop_without_local_weights() -> None:
    from app.pipeline.dl_enhancement import dl_enhance_photo

    image = np.full((24, 24, 3), 120, dtype=np.uint8)
    output, metadata = dl_enhance_photo(image)
    assert np.array_equal(output, image)
    assert metadata["planner"]["unsafe_models_enabled"] is False
