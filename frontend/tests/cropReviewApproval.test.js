import assert from "node:assert/strict";
import test from "node:test";
import { contrastOutlineNeedsConfirmation, heldPrintCorrectionHint, wholePrintApprovalHeld } from "../src/cropReviewApproval.js";

test("only automatically detected contrast outlines with an unsupported paper boundary need confirmation", () => {
  const photo = { segmentation_confidence: 0.71, segmentation_mask: { detector_source: "contrast_region" } };
  const full = { status: "review_needed", reason_codes: ["PRINT_BOUNDARY_UNSUPPORTED", "PRINT_ASPECT_UNCERTAIN"] };
  assert.equal(contrastOutlineNeedsConfirmation(photo, full), true);
  assert.equal(contrastOutlineNeedsConfirmation({ ...photo, segmentation_confidence: null, confidence: 0.71 }, full), true);
  assert.equal(contrastOutlineNeedsConfirmation({ ...photo, segmentation_confidence: null }, full), false);
  assert.equal(contrastOutlineNeedsConfirmation({ ...photo, segmentation_mask: { detector_source: "edge_contour" } }, full), false);
  assert.equal(contrastOutlineNeedsConfirmation(photo, { ...full, status: "ready" }), false);
  assert.equal(contrastOutlineNeedsConfirmation(photo, { ...full, reason_codes: [...full.reason_codes, "PRINT_AT_CAPTURE_EDGE"] }), false);
  assert.equal(contrastOutlineNeedsConfirmation(photo, { ...full, reason_codes: ["PRINT_ASPECT_UNCERTAIN"] }), false);
});

test("the current whole-print URL controls geometry approval even when a mode says ready", () => {
  const urls = { full_print: null, full_print_preview: "/comparison", image_only: "/motif" };
  assert.equal(wholePrintApprovalHeld({ full_print: { status: "review_needed" } }, urls), true);
  assert.equal(wholePrintApprovalHeld({ full_print: { status: "ready" } }, urls), true);
  assert.equal(wholePrintApprovalHeld({ full_print: { status: "ready" } }, { ...urls, full_print: "/print" }), false);
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
