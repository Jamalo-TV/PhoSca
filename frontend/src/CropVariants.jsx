import { Download } from "lucide-react";
import { useState } from "react";
import { getPaperGridReview } from "./cropReviewOutlines.js";

export const cropModeLabels = {
  full_print: "Ganzer Abzug",
  image_only: "Nur Bildmotiv",
};

export const clippedOverlapReason = "CLIPPED_PRINT_OVERLAP_UNCERTAIN";
export const clippedOverlapHint = "Prüfe am Original, ob hier ein weiterer Papierabzug aufliegt. Ziehe die Ecken an seine Papierkante oder markiere den Vorschlag als «Falsches Objekt».";
export const scannerBackingReason = "SCANNER_BACKING_UNCERTAIN";
export const scannerBackingHint = "Prüfe am Original, ob die Markierung einen Abzug zeigt. Gehört der Bereich zur Scannerauflage, wähle «Falsches Objekt».";
export const filmGridReason = "FILM_GRID_REVIEW";
export const filmGridHint = "Gehört der Vorschlag zur Negativhülle, wähle «Falsches Objekt». Liegt dort ein Papierabzug, setze unter «Ganzer Abzug» die Ecken an seinen Papierrand.";
export const filmGridCountHint = "Prüfe die Vorschläge beidseits der türkis gestrichelten Linie im Original. Sind dort Papierabzüge oder Teile einer Negativhülle markiert?";
export const paperGridReason = "PAPER_GRID_REVIEW";
export const paperGridCountHint = "Die blau gestrichelte Kontur zeigt einen möglichen Aussenumriss. Prüfe am Original, ob die Ausschnitte an den gepunkteten Linien eigene Papierabzüge sind.";
export const paperGridOuterHint = "Prüfe den möglichen Aussenumriss am Original. Lade ihn für einen ganzen Papierabzug in den Editor oder markiere den Vorschlag als «Falsches Objekt».";
export const paperGridFragmentHint = "Prüfe am Original, ob die Markierung einen eigenen Papierabzug zeigt. Gehört der Ausschnitt zum grösseren Abzug, wähle «Falsches Objekt».";
export const fragmentOutlineReason = "FRAGMENT_OUTLINE_UNCERTAIN";
export const fragmentOutlineHint = "Prüfe am Original, ob die Ausschnitte zu einem Abzug gehören. Falls ja, korrigiere einen Ausschnitt mit dem Aussenumriss und markiere die übrigen darin als «Falsches Objekt».";
export const fragmentCountHint = "Prüfe, ob die markierten Ausschnitte innerhalb der blauen Kontur zu einem Abzug gehören. Achte besonders auf die gepunktete Seite.";
export const scallopedOutlineReviewReason = "PRINT_SCALLOPED_OUTLINE_REVIEW";
export const scallopedOutlineReviewHint = "Die Vorschau folgt der gewellten Papierkante. Prüfe am Original, ob der ganze Abzug sichtbar ist.";
const scallopedOwnershipReason = "PRINT_SCALLOPED_OWNERSHIP_UNCERTAIN";
const scallopedOwnershipHint = "Prüfe am Original, ob im markierten Bereich ein zweiter Abzug aufliegt. Bei getrennten Abzügen korrigiere beide Konturen. Gehört der Teil zum selben Papier, markiere den Teilvorschlag als «Falsches Objekt».";
export const adjacentPrintHint = "Prüfe am Original, ob der gestrichelte Bereich ein eigener Abzug ist und ob der helle Streifen zu seinem Papier gehört.";
export const nestedPrintHint = "Prüfe am Original, ob der gestrichelte Bereich zum selben Papierabzug gehört. Bei einem Abzug bleiben alle Bilder im Motiv. Ergänze den Bereich, wenn dort ein eigener Abzug aufliegt.";
export const deferredPrintHint = "Prüfe die violett gestrichelten Konturen am Original. Bestätige die Anzahl, wenn alle Abzüge bereits erfasst sind.";
export const crossingPrintReason = "CROSSING_PRINT_OUTLINE_REVIEW";
export const crossingPrintHint = "Prüfe am Original, ob die gestrichelte Kontur einen eigenen Abzug zeigt. Falls die durchgezogene Markierung falsch ist, lade die gestrichelte Kontur und setze die Ecken an den Papierrand.";
export const crossingPrintSaveHint = "Korrigiere und speichere zuerst die vier Ecken unter «Ganzer Abzug».";
export const printOutlineSaveHint = "Prüfe und speichere zuerst die Papierkontur unter «Ganzer Abzug».";
export const captureOutlineReviewReason = "PRINT_CAPTURE_OUTLINE_REVIEW";
const captureAtEdgeReason = "PRINT_AT_CAPTURE_EDGE";
const competingPrintEdges = "competing_print_edges";
const captureAtEdgeHint = "Der Abzug reicht über den Aufnahmerand hinaus. Für einen vollständigen Zuschnitt brauchst du eine neue Aufnahme.";
export const captureSplitReviewHint = "Die gestrichelten Linien zeigen einen möglichen gemeinsamen Papierrand. Prüfe am Original, wie viele Abzüge dort liegen.";
export const missingPrintEdgeHint = "Die blauen Abschnitte zeigen sichtbare Kanten. Prüfe am Original, ob dort ein eigener Papierabzug liegt.";
const curvedMotifReviewReason = "IMAGE_BOUNDARY_CURVED";
const curvedMotifReviewHint = "Die Motivkante verläuft unregelmässig. Prüfe am Original, welche Bildfelder und Beschriftungen zum Motiv gehören.";
const markedMotifMarginReason = "IMAGE_MARGIN_MARKS_REVIEW";
const markedMotifMarginHint = "Am Papierrand sind mögliche Beschriftungen sichtbar. Prüfe den ganzen Rand am Original und nimm alle zum Bild gehörenden Beschriftungen in den Zuschnitt auf.";
const printEdgeReviewHints = {
  PRINT_EDGE_CONFIRMATION_REQUIRED: printOutlineSaveHint,
  PRINT_SHADOW_EDGE_REVIEW: "Prüfe die markierte Kante am Original. Achte darauf, wo der Schatten endet und der Abzug beginnt.",
  PRINT_FAINT_OUTER_RIM_REVIEW: "Prüfe am Original, ob der helle Rand zur markierten Papierkante des Abzugs gehört.",
  PRINT_SHARED_RIM_UNCERTAIN: "Prüfe die Kante am benachbarten Abzug. Wenn der Papierrand unklar bleibt, hilft eine Aufnahme mit getrennten Abzügen.",
  PRINT_OUTER_EDGE_CONFLICT_REVIEW: "Aussen sind mehrere Kanten sichtbar. Lege die Ecken im Editor an die tatsächliche Papierkante.",
  PRINT_EXTENDED_OUTLINE_REVIEW: "Prüfe die erweiterte Kontur am Original. Setze die Ecken an den Papierrand, damit der ganze Abzug erhalten bleibt.",
  PRINT_BOUNDARY_UNSUPPORTED: "Prüfe den sichtbaren Papierrand am Original. Korrigiere die markierten Ecken, soweit die Kante erkennbar ist.",
  PRINT_BOUNDARY_CURVED: "Die Abzugskante ist gekrümmt. Prüfe am Original, ob der Abzug für eine neue Aufnahme flach aufgelegt werden kann.",
  PRINT_CURVED_OUTLINE_REVIEW: "Die Papierkante braucht eine weitere Prüfung. Setze die Ecken an die sichtbare Aussenkante. Bei einer Wölbung hilft eine neue Aufnahme mit flach liegendem Abzug.",
  [captureOutlineReviewReason]: "Die vollständige Papierkante ist noch unklar. Vergleiche die Ecken mit der Aussenkante im Original. Bei einer verdeckten oder gewölbten Kante hilft eine neue Aufnahme mit sichtbarem Rand rundum.",
  [captureAtEdgeReason]: captureAtEdgeHint,
  PAPER_GRID_OVERLAP_REVIEW: "Zwei Vorschläge überdecken dieselbe Stelle. Kläre zuerst die überlappenden Vorschläge am Original und berechne den Zuschnitt danach erneut.",
};

export function getSourceEdgeTraces(traces) {
  return Array.isArray(traces) ? traces.filter((trace) => Array.isArray(trace) && trace.length >= 2
    && trace.every((point) => [point?.x, point?.y].every((value) => Number.isFinite(value) && value >= 0 && value <= 1))) : [];
}

export function getDeferredPrintProposals(proposals) {
  return Array.isArray(proposals) ? proposals.filter((proposal) => Number.isInteger(proposal?.id) && proposal.id >= 0
    && proposal.polygon?.length === 4 && getSourceEdgeTraces([proposal.polygon]).length === 1).map((proposal) => {
      const [outline] = getSourceEdgeTraces([proposal.visible_polygon]);
      return { ...proposal, visible_polygon: outline?.length >= 3 ? outline : proposal.polygon };
    }) : [];
}

export function captureRequiresReview(fullPrint) {
  return fullPrint?.reason_codes?.some(reason => reason === captureAtEdgeReason || reason === captureOutlineReviewReason);
}

export function paperOutlineRequiresReview(fullPrint) {
  return fullPrint?.status === "review_needed" && (fullPrint.suggested_edge_source === competingPrintEdges
    || fullPrint.reason_codes?.some(reason => reason === "PRINT_EDGE_CONFIRMATION_REQUIRED" || reason === "PAPER_GRID_OVERLAP_REVIEW"));
}

export function getPrintEdgeReviewHint(fullPrint) {
  if (fullPrint?.reason_codes?.includes(scallopedOwnershipReason)) {
    return scallopedOwnershipHint;
  }
  if (fullPrint?.suggested_warp_quad?.length === 4) {
    return "Vergleiche den vorgeschlagenen Aussenumriss mit dem Original. Setze die Ecken an die sichtbare Papierkante.";
  }
  if (fullPrint?.reason_codes?.includes("PRINT_CURVED_OUTLINE_REVIEW")) {
    return printEdgeReviewHints.PRINT_CURVED_OUTLINE_REVIEW;
  }
  if (getSourceEdgeTraces(fullPrint?.suggested_edge_traces).length) {
    if (fullPrint.suggested_edge_source === competingPrintEdges) {
      return "Die gestrichelten Linien zeigen mögliche Kanten im Original. Prüfe, welche das Fotopapier begrenzt.";
    }
    if (fullPrint.suggested_edge_source === "capture_edge_split") return captureSplitReviewHint;
    if (fullPrint.suggested_edge_source === "capture_edge_boundary") {
      return `Die gestrichelten Linien zeigen die sichtbaren Papierkanten. ${captureAtEdgeHint}`;
    }
    return "Die gestrichelten Abschnitte zeigen die gemessene Papierkante im Original. Vergleiche sie mit den geraden Zuschnittkanten.";
  }
  return fullPrint?.reason_codes?.map((reason) => printEdgeReviewHints[reason]).find(Boolean);
}

export function getMotifBoundaryReviewHint(motif) {
  if (motif?.status !== "review_needed") return null;
  if (motif.reason_codes?.includes(markedMotifMarginReason)) {
    return markedMotifMarginHint;
  }
  if (motif.reason_codes?.includes(curvedMotifReviewReason)) return curvedMotifReviewHint;
  return motif.reason_codes?.includes("IMAGE_RECTANGLE_INCONSISTENT") ? reviewMessages.IMAGE_RECTANGLE_INCONSISTENT : null;
}

export function getMotifMarginMarks(motif) {
  return getSourceEdgeTraces(motif?.margin_mark_regions).filter((region) => region.length === 4);
}

export function getPrintOutlinePreviewHint(fullPrint) {
  const preview = fullPrint?.outline_preview;
  if (!preview?.aspect_ratio_estimation) return null;
  return preview.reason_codes?.includes("PRINT_ASPECT_UNCERTAIN")
    ? "Vorgeschlagener Abzug mit geschätztem Seitenverhältnis."
    : "Vorschau des vorgeschlagenen Aussenumrisses.";
}

export function getCropModes(photo) {
  return photo?.enhancement_applied?.correction?.crop_modes || null;
}

export function canEditMobileMotif(photo, modes) {
  const { segmentation_confidence: confidence = photo?.confidence } = photo || {};
  const quad = photo?.segmentation_mask?.warp_quad;
  return confidence === null && modes?.full_print?.status === "ready" && Boolean(modes.image_only)
    && quad?.length === 4 && getSourceEdgeTraces([quad]).length === 1;
}

export function CropOrientationField({ value, onChange, disabled }) {
  return (
    <label className="crop-orientation-field">
      <span>Ausrichtung ab Albumseite</span>
      <select value={value ?? ""} disabled={disabled} onChange={(event) => {
        const selected = event.target.value;
        onChange(selected === "" ? null : Number(selected));
      }}>
        <option value="">Automatisch</option>
        <option value={0}>0° · beibehalten</option>
        <option value={90}>90° · nach rechts</option>
        <option value={180}>180°</option>
        <option value={270}>90° · nach links</option>
      </select>
    </label>
  );
}

const reviewMessages = {
  ORIENTATION_UNCERTAIN: "Prüfe die Ausrichtung und speichere die passende Drehung im Zuschnitt-Editor.",
  SEGMENTATION_CONFIDENCE_UNCALIBRATED: "Prüfe und speichere die Abzugsecken, um den Download freizugeben.",
  MULTIPANEL_PRINT_REVIEW: "Prüfe die Papierkanten am Original. Auf einem Abzug muss der Motivzuschnitt alle Bilder enthalten. Sind es getrennte Abzüge, korrigierst du die Ecken für den ersten und ergänzt die übrigen im Editor.",
  FRAGMENTED_PRINT_REVIEW: "Prüfe die durchgehende Papierkante am Original. Liegen zwei einzelne Abzüge vor, korrigiere die Ecken des ersten und ergänze den zweiten im Editor.",
  [fragmentOutlineReason]: fragmentOutlineHint,
  [crossingPrintReason]: crossingPrintHint,
  [scannerBackingReason]: scannerBackingHint,
  [filmGridReason]: filmGridHint,
  PAPER_GRID_OVERLAP_REVIEW: printEdgeReviewHints.PAPER_GRID_OVERLAP_REVIEW,
  CROP_VALIDATION_UNAVAILABLE: "Prüfe und speichere den Zuschnitt, um den Download freizugeben.",
  CAMERA_ASPECT_REVIEW: "Die Vorschau nutzt die Kamerakalibrierung. Prüfe und bestätige die Fotoecken.",
  PRINT_AT_CAPTURE_EDGE: "Der Abzug reicht bis an den Aufnahmerand. Prüfe, ob er vollständig zu sehen ist.",
  PRINT_POSSIBLY_OCCLUDED: "Ein Teil des Abzugs könnte verdeckt sein.",
  [scallopedOutlineReviewReason]: scallopedOutlineReviewHint,
  [scallopedOwnershipReason]: scallopedOwnershipHint,
  PRINT_BOUNDARY_BLURRED: "Die Abzugskante ist unscharf. Prüfe den Zuschnitt am Original.",
  ...printEdgeReviewHints,
  ALBUM_WINDOW_OCCLUDED: "Der Abzug steckt in einem Albumfenster. Prüfe, welche Teile sichtbar sind.",
  ALBUM_WINDOW_STOCK_UNCERTAIN: "Prüfe, ob der helle Rand zum Abzug oder zum Albumkarton gehört.",
  PRINT_CORNERS_UNCERTAIN: "Mindestens eine Ecke des Abzugs muss geprüft werden.",
  PRINT_EDGE_CONFIRMATION_REQUIRED: "Prüfe und bestätige die vorgeschlagene Papierkante am Original.",
  PRINT_ASPECT_UNCERTAIN: "Miss Breite und Höhe des Abzugs und trage beides im Zuschnitt-Editor ein.",
  PRINT_ASPECT_REFERENCE_REVIEW: "Die Vorschau nutzt einen vermessenen Abzug derselben Seite. Prüfe die Masse dieses Abzugs.",
  PRINT_BOUNDARY_SAMPLING_UNCERTAIN: "An einer Abzugsecke fehlen eindeutig zuordenbare Bildpixel.",
  IMAGE_BOUNDARY_SAMPLING_UNCERTAIN: "An einer Motivecke fehlen eindeutig zuordenbare Bildpixel.",
  IMAGE_BOUNDARY_UNCERTAIN: "Die Grenze zwischen Rahmen und Bildmotiv muss geprüft werden.",
  [curvedMotifReviewReason]: curvedMotifReviewHint,
  [markedMotifMarginReason]: markedMotifMarginHint,
  WIDE_MOTIF_CONFIRMATION_REQUIRED: "Prüfe den Motivvorschlag im Editor und speichere die Motivecken.",
  IMAGE_VISIBILITY_UNCERTAIN: "Das vollständige Bildmotiv konnte nicht bestätigt werden.",
  IMAGE_BORDER_CLASSIFICATION_DISABLED: "Die automatische Bestimmung der Motivgrenze ist deaktiviert.",
  IMAGE_RECTANGLE_INCONSISTENT: "Die Motivecken passen nicht zum entzerrten Abzug. Prüfe die Motivgrenze.",
  BORDER_PROTECTED_CONTENT_VETO: "An der Motivgrenze könnten Bildinhalte liegen. Prüfe den Zuschnitt.",
};

export function CropVariants({ photo, hidden = false, mobile = false }) {
  const [mode, setMode] = useState(() => !photo.urls?.full_print && photo.urls?.image_only ? "image_only" : "full_print");
  const label = cropModeLabels[mode];
  const result = getCropModes(photo)?.[mode];
  const url = result?.status === "ready" ? photo.urls?.[mode] : null;
  const possibleMultipanel = photo.enhancement_applied?.segmentation_review_reasons?.includes("MULTIPANEL_PRINT_REVIEW");
  const possibleFragmentedPrint = photo.segmentation_confidence != null
    && photo.enhancement_applied?.segmentation_review_reasons?.includes("FRAGMENTED_PRINT_REVIEW");
  const possibleFragmentOutline = photo.segmentation_confidence != null
    && photo.enhancement_applied?.segmentation_review_reasons?.includes(fragmentOutlineReason);
  const possibleFilmGrid = photo.segmentation_confidence != null
    && photo.enhancement_applied?.segmentation_review_reasons?.includes(filmGridReason);
  const paperGrid = photo.segmentation_confidence != null
    && photo.enhancement_applied?.segmentation_review_reasons?.includes(paperGridReason)
    && getPaperGridReview(photo.segmentation_mask);
  const possibleAdjacentPrint = photo.segmentation_confidence != null && photo.segmentation_mask?.adjacent_print_suggestion;
  const possibleNestedPrint = photo.segmentation_confidence != null && photo.segmentation_mask?.nested_print_suggestion;
  const possibleCrossingPrint = photo.segmentation_confidence != null && photo.segmentation_mask?.crossing_print_suggestion;
  const scannerBackingUncertain = photo.segmentation_confidence != null
    && photo.enhancement_applied?.segmentation_review_reasons?.includes(scannerBackingReason);
  const clippedOverlapUncertain = photo.segmentation_confidence != null
    && photo.enhancement_applied?.segmentation_review_reasons?.includes(clippedOverlapReason);
  const captureEdgeClipped = photo.segmentation_mask?.capture_edge_clipped === true;
  const captureSplit = getCropModes(photo)?.full_print?.suggested_edge_source === "capture_edge_split";
  const pendingPaperEdge = getCropModes(photo)?.full_print?.reason_codes?.includes("PRINT_EDGE_CONFIRMATION_REQUIRED");
  const scallopedOutline = mode === "full_print" && result?.reason_codes?.includes(scallopedOutlineReviewReason);
  const printOutline = photo.urls?.full_print_preview;
  const fullPreview = printOutline || photo.urls?.source_corrected_untrimmed || photo.urls?.original;
  const motifPreview = photo.urls?.image_only_preview;
  const preview = mode === "image_only"
    ? motifPreview || ((possibleMultipanel || possibleFragmentedPrint || possibleFragmentOutline || scannerBackingUncertain || clippedOverlapUncertain || possibleFilmGrid
        || paperGrid || captureEdgeClipped || captureSplit || pendingPaperEdge || result?.reason_codes?.includes(markedMotifMarginReason)) && fullPreview)
    : fullPreview;
  const previewAlt = mode === "image_only" && motifPreview
    ? "Konturgetreue Motivvorschau zur Prüfung"
    : preview && preview === printOutline
      ? "Vorschau der Papierkontur zur Prüfung"
      : "Vorschau zur Prüfung";
  const outlinePreviewHint = preview && preview === printOutline
    && getPrintOutlinePreviewHint(getCropModes(photo)?.full_print);
  const reviewMessage = (captureEdgeClipped
    && "Der vorgeschlagene Abzug reicht bis an den Aufnahmerand. Lade für den Zuschnitt eine Aufnahme hoch, die ihn ganz zeigt.")
    || (captureSplit && captureSplitReviewHint)
    || (possibleNestedPrint && nestedPrintHint)
    || (possibleMultipanel && reviewMessages.MULTIPANEL_PRINT_REVIEW)
    || (possibleFragmentedPrint && reviewMessages.FRAGMENTED_PRINT_REVIEW)
    || (possibleFragmentOutline && fragmentOutlineHint)
    || (possibleAdjacentPrint && adjacentPrintHint)
    || (possibleCrossingPrint && crossingPrintHint)
    || (scannerBackingUncertain && scannerBackingHint)
    || (clippedOverlapUncertain && clippedOverlapHint)
    || (possibleFilmGrid && filmGridHint)
    || (paperGrid && (paperGrid.cut ? paperGridFragmentHint : paperGridOuterHint))
    || (scallopedOutline && scallopedOutlineReviewHint)
    || result?.reason_codes?.map((code) => reviewMessages[code]).find(Boolean)
    || (result?.status === "ready" && !url && reviewMessages.CROP_VALIDATION_UNAVAILABLE)
    || (mode === "image_only" ? reviewMessages.IMAGE_BOUNDARY_UNCERTAIN : "Die Aussenkante des Abzugs muss geprüft werden.");
  const orientationNeedsReview = photo.enhancement_applied?.correction?.orientation?.reason_code === "ORIENTATION_UNCERTAIN";

  return (
    <section className="crop-variants" aria-label="Fotoausschnitt" hidden={hidden}>
      <div className="variant-tabs crop-mode-tabs" role="group" aria-label="Ausschnitt wählen">
        {Object.entries(cropModeLabels).map(([name, text]) => (
          <button key={name} type="button" aria-pressed={mode === name} onClick={() => setMode(name)}>
            {text}
          </button>
        ))}
      </div>
      {url ? (
        <>
          {orientationNeedsReview && <p className="crop-mode-status" role="status">{reviewMessages.ORIENTATION_UNCERTAIN}</p>}
          <img key={url} className="crop-variant-image" src={url} alt={label} />
          <a className="crop-download" href={url} download={`photo-${photo.id}-${mode}.jpg`}>
            <Download size={16} /> {label} herunterladen
          </a>
        </>
      ) : (
        <>
          <p className="crop-mode-status" role="status">{reviewMessage}</p>
          {mobile && mode === "image_only" && getMotifBoundaryReviewHint(result) && (
            <p className="crop-mode-status">{canEditMobileMotif(photo, getCropModes(photo))
              ? "Öffne «Cropping prüfen» und wähle beim Abzug «Nur Bildmotiv»."
              : "Die Motivecken kannst du im Desktop-Editor setzen."}</p>
          )}
          {preview && <img className="crop-variant-image" src={preview} alt={previewAlt} />}
          {outlinePreviewHint && <p className="crop-mode-status">{outlinePreviewHint}</p>}
        </>
      )}
    </section>
  );
}
