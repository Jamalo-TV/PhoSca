"""Single, conservative release decision for photo variants.

Generation stages may create as many experimental variants as they need, but
they must never decide what a consumer receives.  This module deliberately has
no model dependencies so the same decision is used by workers and the API.
"""

from __future__ import annotations

from typing import Any

BLOCKING_STATUSES = {"review_needed", "failed"}


def release_reasons(
    *, photo_status: Any = None, metadata: dict | None = None, variant: dict | None = None
) -> list[str]:
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
    variant_kind = variant_metadata.get("variant_kind") or record.get("kind") or record.get("name")
    if variant_kind in {"restoration_candidate", "face_restoration_candidate"} or variant_metadata.get("generative"):
        reasons.append("variant:model_restoration_requires_explicit_review")
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
    source_name = "source_corrected" if isinstance(variants.get("source_corrected"), dict) else "original"
    original = variants.get(source_name)
    if not isinstance(original, dict):
        raise ValueError("A release bundle must contain a source_corrected or original variant.")

    # A completed blind review is the only path that may deliberately select
    # a labelled restoration candidate. Automatic policy and quality scores
    # can never synthesize this override.
    decisions = (metadata or {}).get("review_decisions") or {}
    enhancement_history = decisions.get("enhancement") or []
    latest_enhancement = enhancement_history[-1] if enhancement_history else {}
    if latest_enhancement.get("state") == "selected":
        blind_choice = latest_enhancement.get("blind_choice")
        candidate_name = str(latest_enhancement.get("candidate_name") or "")
        if blind_choice == "candidate" and isinstance(variants.get(candidate_name), dict):
            return candidate_name, []
        if blind_choice in {"original", "equal"}:
            return source_name, []

    global_reasons = release_reasons(photo_status=photo_status, metadata=metadata)
    if global_reasons:
        return source_name, global_reasons

    # Model/generative restoration candidates are intentionally absent.  They
    # can be viewed explicitly, but never become the faithful default here.
    for name in ("conservative_enhanced", "conservative", "enhanced"):
        candidate = variants.get(name)
        if not isinstance(candidate, dict):
            continue
        reasons = release_reasons(variant=candidate)
        if candidate.get("metadata", {}).get("auto_release") is True and not reasons:
            return name, []
    fallback_reasons = ["no_derived_variant_passed_release_gate"]
    for name in ("restoration_candidate", "face_restoration_candidate", "premium"):
        candidate = variants.get(name)
        if isinstance(candidate, dict):
            fallback_reasons.extend(release_reasons(variant=candidate))
    return source_name, list(dict.fromkeys(fallback_reasons))
