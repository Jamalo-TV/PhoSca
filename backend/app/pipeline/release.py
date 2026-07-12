"""Single, conservative release decision for photo variants.

Generation stages may create as many experimental variants as they need, but
they must never decide what a consumer receives.  This module deliberately has
no model dependencies so the same decision is used by workers and the API.
"""
from __future__ import annotations

from typing import Any


BLOCKING_STATUSES = {"review_needed", "failed"}


def release_reasons(*, photo_status: Any = None, metadata: dict | None = None, variant: dict | None = None) -> list[str]:
    """Return stable reasons why a derived variant cannot be auto-released."""
    reasons: list[str] = []
    status = getattr(photo_status, "value", photo_status)
    if status in BLOCKING_STATUSES:
        reasons.append(f"photo_status_{status}")

    metadata = metadata or {}
    policy = metadata.get("release_policy") or {}
    if policy.get("kill_switch", False):
        reasons.append("policy:kill_switch")
    if policy.get("shadow_mode", False):
        reasons.append("policy:shadow_mode")
    if policy.get("canary_eligible") is False:
        reasons.append("policy:not_in_canary")
    for reason in metadata.get("review_reasons") or []:
        reasons.append(f"review:{reason}")
    correction = metadata.get("correction") or {}
    if correction.get("status") == "rejected":
        reasons.append(f"correction:{correction.get('reason', 'rejected')}")

    record = variant or {}
    for warning in record.get("warnings") or []:
        reasons.append(f"variant:{warning}")
    variant_metadata = record.get("metadata") or {}
    if variant_metadata.get("method") == "failed" or variant_metadata.get("error"):
        reasons.append("variant:stage_failed")
    gates = variant_metadata.get("gates") or record.get("gates") or []
    for gate in gates:
        if isinstance(gate, dict) and gate.get("passed") is False:
            reasons.append(f"gate:{gate.get('name', 'unknown')}")
    return list(dict.fromkeys(reasons))


def select_released_variant(
    variants: dict[str, dict[str, Any]], *, photo_status: Any = None, metadata: dict | None = None
) -> tuple[str, list[str]]:
    """Select an auto-release variant, always retaining ``original`` fallback.

    ``auto_release`` is an explicit opt-in and is insufficient on its own: any
    accumulated review reason, failed stage, warning, or failed gate vetoes the
    candidate.  Experimental variants remain accessible by explicit request.
    """
    original = variants.get("original")
    if not isinstance(original, dict):
        raise ValueError("A release bundle must contain an original variant.")

    global_reasons = release_reasons(photo_status=photo_status, metadata=metadata)
    if global_reasons:
        return "original", global_reasons

    for name in ("conservative", "enhanced", "premium"):
        candidate = variants.get(name)
        if not isinstance(candidate, dict):
            continue
        reasons = release_reasons(variant=candidate)
        if candidate.get("metadata", {}).get("auto_release") is True and not reasons:
            return name, []
    return "original", ["no_derived_variant_passed_release_gate"]
