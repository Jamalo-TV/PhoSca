export const contrastOutlineReviewHint = "Prüfe, ob alle vier Ecken zum selben Papierabzug gehören. Markiert die Kontur keinen eigenen Abzug, wähle «Falsches Objekt».";
export const contrastOutlineConfirmation = "Die vier Ecken liegen am sichtbaren Rand desselben Papierabzugs.";

export function contrastOutlineNeedsConfirmation(photo, fullPrint) {
  const confidence = photo?.segmentation_confidence ?? photo?.confidence;
  return confidence != null && photo?.segmentation_mask?.detector_source === "contrast_region"
    && fullPrint?.status === "review_needed" && fullPrint.reason_codes?.includes("PRINT_BOUNDARY_UNSUPPORTED") === true
    && !fullPrint.reason_codes.includes("PRINT_AT_CAPTURE_EDGE");
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
