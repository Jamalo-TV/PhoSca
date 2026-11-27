export const contrastOutlineReviewHint = "Prüfe, ob alle vier Ecken zum selben Papierabzug gehören. Markiert die Kontur keinen eigenen Abzug, wähle «Falsches Objekt».";
export const contrastOutlineConfirmation = "Die vier Ecken liegen am sichtbaren Rand desselben Papierabzugs.";
export const measuredMotifPaperEditHint = "Setze vor dem Speichern alle vier Ecken an den sichtbaren Papierrand des Abzugs.";

export function contrastOutlineNeedsConfirmation(photo, fullPrint) {
  const confidence = photo?.segmentation_confidence ?? photo?.confidence;
  const mask = photo?.segmentation_mask;
  const source = mask?.detector_source;
  const reasons = fullPrint?.reason_codes || [];
  const contact = fullPrint?.contact_print_edge_check_required === true && ["contrast_region", "yolo_onnx"].includes(source);
  const opposingEdges = fullPrint?.opposing_outer_edges_check_required === true && source === "yolo_onnx";
  const motifComparison = fullPrint?.motif_comparison_requires_paper_edit === true && source === "yolo_onnx";
  const faintPaper = fullPrint?.faint_paper_review_requires_confirmation === true && source === "yolo_onnx";
  const possiblePrintRegions = source === "yolo_onnx" && Object.hasOwn(mask, "possible_print_regions");
  const unsupportedContrast = source === "contrast_region" && reasons.includes("PRINT_BOUNDARY_UNSUPPORTED");
  return confidence != null
    && fullPrint?.status === "review_needed"
    && (contact || opposingEdges || motifComparison || faintPaper || possiblePrintRegions || unsupportedContrast)
    && !reasons.includes("PRINT_AT_CAPTURE_EDGE");
}

export function contactEdgesNeedRefresh(photo, fullPrint) {
  if (fullPrint?.status !== "review_needed" || fullPrint.contact_print_edge_scope) return false;
  if (contrastOutlineNeedsConfirmation(photo, fullPrint) && fullPrint.contact_print_edge_check_required === true) return true;
  const mask = photo?.segmentation_mask;
  const reasons = fullPrint.reason_codes || [];
  // Existing crops can lack the marker until their first reprocess.
  return (photo?.segmentation_confidence ?? photo?.confidence) != null && mask?.detector_source === "yolo_onnx"
    && mask.source === "yolo_seg_mask" && mask.fusion?.selected_family === "model" && mask.fusion.proposal_count === 2
    && reasons.length === 2 && reasons.includes("PRINT_OUTER_RIM_UNCERTAIN") && reasons.includes("PRINT_ASPECT_UNCERTAIN");
}

export function wholePrintApprovalHeld(cropModes, variantUrls) {
  const fullPrint = cropModes?.full_print;
  return Boolean(fullPrint && Object.keys(fullPrint).length && variantUrls?.full_print === null);
}

export function heldPrintCorrectionHint(fullPrint) {
  const reasons = fullPrint?.reason_codes || [];
  if (reasons.includes("PRINT_POSSIBLY_OCCLUDED") || reasons.includes("ALBUM_WINDOW_OCCLUDED")) {
    return "Prüfe am Original, ob alle vier Papierkanten sichtbar sind. Ist eine Kante verdeckt, fotografiere den Abzug einzeln mit sichtbarem Rand rundum.";
  }
  if (reasons.includes("PRINT_ASPECT_UNCERTAIN") || reasons.includes("PRINT_ASPECT_REFERENCE_REVIEW")) {
    return "Setze die Ecken an den sichtbaren Papierrand. Miss Breite und Höhe am Abzug und speichere die Masse zusammen mit den Ecken.";
  }
  return "Prüfe die vier Ecken am Original. Setze sie an den sichtbaren Papierrand und speichere den Zuschnitt.";
}
