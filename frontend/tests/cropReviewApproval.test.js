import assert from "node:assert/strict";
import test from "node:test";
import {
  contactEdgesNeedRefresh, contrastOutlineNeedsConfirmation, heldPrintCorrectionHint,
  measuredOutlineCutoffRequiresReview, measuredOutlineRecheckRequiresReview, paperOutlineRequiresMovement,
  wholePrintApprovalHeld,
} from "../src/cropReviewApproval.js";

test("stacked scene comparisons require an actual paper edit before saving", () => {
  assert.equal(paperOutlineRequiresMovement({ stacked_scene_requires_paper_edit: true }, false), true);
  assert.equal(paperOutlineRequiresMovement({ outline_preview: { source: "disputed_paper_grid_outline_review" } }, false), false);
  assert.equal(paperOutlineRequiresMovement({ motif_comparison_requires_paper_edit: true }, true), true);
  assert.equal(paperOutlineRequiresMovement({ motif_comparison_requires_paper_edit: true }, false), false);
});

test("automatically detected contrast outlines with source boundary review need confirmation", () => {
  const photo = { segmentation_confidence: 0.71, segmentation_mask: { detector_source: "contrast_region" } };
  const full = { status: "review_needed", reason_codes: ["PRINT_BOUNDARY_UNSUPPORTED", "PRINT_ASPECT_UNCERTAIN"] };
  assert.equal(contrastOutlineNeedsConfirmation(photo, full), true);
  assert.equal(contrastOutlineNeedsConfirmation({ ...photo, segmentation_confidence: null, confidence: 0.71 }, full), true);
  assert.equal(contrastOutlineNeedsConfirmation({ ...photo, segmentation_confidence: null }, full), false);
  assert.equal(contrastOutlineNeedsConfirmation({ ...photo, segmentation_mask: { detector_source: "edge_contour" } }, full), false);
  assert.equal(contrastOutlineNeedsConfirmation(photo, { ...full, status: "ready" }), false);
  assert.equal(contrastOutlineNeedsConfirmation(photo, { ...full, reason_codes: [...full.reason_codes, "PRINT_AT_CAPTURE_EDGE"] }), false);
  assert.equal(contrastOutlineNeedsConfirmation(photo, { ...full, reason_codes: ["PRINT_ASPECT_UNCERTAIN"] }), false);
  const contact = { status: "review_needed", reason_codes: ["PRINT_BOUNDARY_AMBIGUOUS"], contact_print_edge_check_required: true };
  assert.equal(contrastOutlineNeedsConfirmation(photo, contact), true);
  assert.equal(contrastOutlineNeedsConfirmation({ ...photo, segmentation_confidence: null }, contact), false);
});

test("a held model contact outline needs the paper confirmation and can refresh its missing edges", () => {
  const photo = { confidence: 0.81, segmentation_mask: { detector_source: "yolo_onnx", source: "yolo_seg_mask",
    fusion: { selected_family: "model", proposal_count: 2 } } };
  const full = { status: "review_needed", reason_codes: ["PRINT_OUTER_RIM_UNCERTAIN", "PRINT_ASPECT_UNCERTAIN"],
    contact_print_edge_check_required: true };
  assert.equal(contrastOutlineNeedsConfirmation(photo, full), true);
  assert.equal(contrastOutlineNeedsConfirmation(photo, { ...full, contact_print_edge_check_required: false,
    reason_codes: ["PRINT_BOUNDARY_UNSUPPORTED"] }), false);
  assert.equal(contrastOutlineNeedsConfirmation(photo, { ...full, reason_codes: ["PRINT_AT_CAPTURE_EDGE"] }), false);
  assert.equal(contrastOutlineNeedsConfirmation({ ...photo, confidence: null }, full), false);
  assert.equal(contactEdgesNeedRefresh(photo, full), true);
  assert.equal(contactEdgesNeedRefresh(photo, { ...full, contact_print_edge_scope: [{ x: .1, y: .2 }] }), false);
  assert.equal(contactEdgesNeedRefresh(photo, { ...full, status: "ready" }), false);
  assert.equal(contactEdgesNeedRefresh(photo, { ...full, contact_print_edge_check_required: undefined }), true);
  assert.equal(contactEdgesNeedRefresh({ ...photo, confidence: null }, full), false);
});

test("opposing outer lines require explicit paper confirmation for a held model proposal", () => {
  const photo = { confidence: .85, segmentation_mask: { detector_source: "yolo_onnx" } };
  const mode = { status: "review_needed", reason_codes: ["PRINT_OUTER_EDGE_CONFLICT_REVIEW"],
    opposing_outer_edges_check_required: true };
  assert.equal(contrastOutlineNeedsConfirmation(photo, mode), true);
  assert.equal(contrastOutlineNeedsConfirmation({ ...photo, confidence: null }, mode), false);
  assert.equal(contrastOutlineNeedsConfirmation(photo, { ...mode, opposing_outer_edges_check_required: false }), false);
  assert.equal(contactEdgesNeedRefresh(photo, mode), false);
});

test("a held model motif comparison requires separate confirmation of the physical paper", () => {
  const mode = { status: "review_needed", reason_codes: ["PRINT_BOUNDARY_AMBIGUOUS"],
    motif_comparison_requires_paper_edit: true };
  const photo = { confidence: .85, segmentation_mask: { detector_source: "yolo_onnx" } };
  assert.equal(contrastOutlineNeedsConfirmation(photo, mode), true);
  assert.equal(contrastOutlineNeedsConfirmation({ ...photo, confidence: null }, mode), false);
  assert.equal(contrastOutlineNeedsConfirmation({ ...photo, segmentation_mask: { detector_source: "contrast_region" } }, mode), false);
  assert.equal(contrastOutlineNeedsConfirmation(photo, { ...mode, motif_comparison_requires_paper_edit: false }), false);
  assert.equal(contrastOutlineNeedsConfirmation(photo, { ...mode, status: "ready" }), false);
  assert.equal(contrastOutlineNeedsConfirmation(photo, { ...mode, reason_codes: ["PRINT_AT_CAPTURE_EDGE"] }), false);
  assert.equal(contactEdgesNeedRefresh(photo, mode), false);
});

test("a held faint paper edge needs explicit confirmation even if its guide is retired", () => {
  const photo = { confidence: .81, segmentation_mask: { detector_source: "yolo_onnx" } };
  const full = { status: "review_needed", reason_codes: ["PRINT_CORNERS_UNCERTAIN"], faint_paper_review_requires_confirmation: true };
  assert.equal(contrastOutlineNeedsConfirmation(photo, full), true);
  assert.equal(contrastOutlineNeedsConfirmation({ ...photo, confidence: null }, full), false);
  assert.equal(contrastOutlineNeedsConfirmation(photo, { ...full, reason_codes: ["PRINT_AT_CAPTURE_EDGE"] }), false);
  assert.equal(contactEdgesNeedRefresh(photo, full), false);

  const unrelated = { ...full, faint_paper_review_requires_confirmation: false, suggested_edge_source: "adjacent_faint_paper_steps" };
  assert.equal(contrastOutlineNeedsConfirmation(photo, unrelated), false);
  assert.equal(contrastOutlineNeedsConfirmation(photo, { ...full, status: "ready" }), false);
  const fallback = { ...photo, segmentation_mask: { detector_source: "contrast_region" } };
  assert.equal(contrastOutlineNeedsConfirmation(fallback, full), false);
});

test("possible print regions require a paper decision on the held automatic model row, even after retirement", () => {
  const mask = { detector_source: "yolo_onnx" };
  const photo = { confidence: .62, segmentation_mask: mask };
  const full = { status: "review_needed", reason_codes: ["POSSIBLE_PRINT_REGIONS_REVIEW"] };
  for (const possible_print_regions of [{ outlines: [] }, null]) {
    const owner = { ...photo, segmentation_mask: { ...mask, possible_print_regions } };
    assert.equal(contrastOutlineNeedsConfirmation(owner, full), true);
    assert.equal(contrastOutlineNeedsConfirmation({ ...owner, confidence: null }, full), false);
    assert.equal(contrastOutlineNeedsConfirmation({ ...owner, segmentation_mask: { ...owner.segmentation_mask,
      detector_source: "edge_contour" } }, full), false);
    assert.equal(contrastOutlineNeedsConfirmation(owner, { ...full, status: "ready" }), false);
  }
  assert.equal(contrastOutlineNeedsConfirmation(photo, full), false);
});

test("the current whole-print URL controls geometry approval even when a mode says ready", () => {
  const urls = { full_print: null, full_print_preview: "/comparison", image_only: "/motif" };
  assert.equal(wholePrintApprovalHeld({ full_print: { status: "review_needed" } }, urls), true);
  assert.equal(wholePrintApprovalHeld({ full_print: { status: "ready" } }, urls), true);
  assert.equal(wholePrintApprovalHeld({ full_print: { status: "ready" } }, { ...urls, full_print: "/print" }), false);
});

test("a measured outline cutoff is a hold for either crop mode", () => {
  const hold = { status: "review_needed", reason_codes: ["PRINT_ASPECT_UNCERTAIN", "PRINT_MEASURED_OUTLINE_CUTOFF"] };
  const modes = { full_print: hold, image_only: hold };
  for (const mode of Object.values(modes)) assert.equal(measuredOutlineCutoffRequiresReview(mode), true);
  assert.equal(measuredOutlineCutoffRequiresReview({ ...hold, status: "ready" }), false);
  assert.equal(measuredOutlineCutoffRequiresReview({ ...hold, reason_codes: ["PRINT_ASPECT_UNCERTAIN"] }), false);
  assert.equal(measuredOutlineCutoffRequiresReview({ status: "review_needed" }), false);
  assert.equal(measuredOutlineCutoffRequiresReview(null), false);
  assert.match(heldPrintCorrectionHint(hold), /gemessenen Papierkontur aus/);
  const stale = { status: "review_needed", reason_codes: ["PRINT_MEASURED_OUTLINE_RECHECK"] };
  assert.equal(measuredOutlineRecheckRequiresReview(stale), true);
  assert.equal(measuredOutlineRecheckRequiresReview({ ...stale, status: "ready" }), false);
  assert.match(heldPrintCorrectionHint(stale), /erneut geprüft/);
});

test("legacy crops and optional motif holds keep their existing geometry approval", () => {
  const urls = { full_print: null, image_only: null };
  assert.equal(wholePrintApprovalHeld({}, urls), false);
  assert.equal(wholePrintApprovalHeld({ full_print: {}, image_only: { status: "review_needed" } }, urls), false);
  assert.equal(wholePrintApprovalHeld({ full_print: { status: "ready" }, image_only: { status: "review_needed" } },
    { ...urls, full_print: "/print" }), false);
});

test("correction guidance prioritizes visible paper around possible occlusion", () => {
  assert.match(heldPrintCorrectionHint({ reason_codes: ["PRINT_ASPECT_UNCERTAIN", "PRINT_POSSIBLY_OCCLUDED"] }),
    /fotografiere den Abzug einzeln/);
  assert.match(heldPrintCorrectionHint({ reason_codes: ["PRINT_ASPECT_UNCERTAIN"] }), /Miss Breite und Höhe/);
  assert.match(heldPrintCorrectionHint({ status: "ready" }), /Prüfe die vier Ecken/);
});
