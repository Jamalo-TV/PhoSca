"""Bounded-cardinality product and safety metrics.

No album, tenant, filename, OCR text, or image-derived identifier is exported.
"""
from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram

    SEGMENTATION_SECONDS = Histogram("phosca_segmentation_seconds", "Segmentation wall time", ["model"])
    SEGMENTATION_CONFIDENCE = Histogram(
        "phosca_segmentation_confidence", "Raw segmentation confidence", ["source"], buckets=(0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
    )
    PROTECTED_REGIONS = Histogram("phosca_protected_regions_per_page", "Protected semantic regions per page", buckets=(0, 1, 2, 4, 8, 16, 32))
    RELEASE_DECISIONS = Counter("phosca_release_decisions_total", "Selected output variants", ["variant", "blocked"])
    ENHANCEMENT_ROUTES = Counter("phosca_enhancement_routes_total", "Enhancement planner routes", ["route"])
    RETENTION_DELETIONS = Counter("phosca_retention_deletions_total", "Retention actions", ["kind", "result"])
    CANARY_PERCENTAGE = Gauge("phosca_canary_percentage", "Configured derived-variant canary percentage")
except ImportError:  # pragma: no cover - production requirements install prometheus_client.
    class _Noop:
        def labels(self, *args, **kwargs): return self
        def observe(self, *args, **kwargs): return None
        def inc(self, *args, **kwargs): return None
        def set(self, *args, **kwargs): return None

    SEGMENTATION_SECONDS = SEGMENTATION_CONFIDENCE = PROTECTED_REGIONS = RELEASE_DECISIONS = ENHANCEMENT_ROUTES = RETENTION_DELETIONS = CANARY_PERCENTAGE = _Noop()
