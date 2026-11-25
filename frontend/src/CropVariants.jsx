import { Download } from "lucide-react";
import { useState } from "react";

export const cropModeLabels = {
  full_print: "Ganzer Abzug",
  image_only: "Nur Bildmotiv",
};

export const clippedOverlapReason = "CLIPPED_PRINT_OVERLAP_UNCERTAIN";
export const clippedOverlapHint = "Prüfe am Original, ob hier ein weiterer Papierabzug aufliegt. Ziehe die Ecken an seine Papierkante oder markiere den Vorschlag als «Falsches Objekt».";
export const fragmentOutlineReason = "FRAGMENT_OUTLINE_UNCERTAIN";
export const fragmentOutlineHint = "Prüfe am Original, ob die Ausschnitte zu einem Abzug gehören. Falls ja, korrigiere einen Ausschnitt mit dem Aussenumriss und markiere die übrigen darin als «Falsches Objekt».";
export const scallopedOutlineReviewReason = "PRINT_SCALLOPED_OUTLINE_REVIEW";
export const scallopedOutlineReviewHint = "Die Vorschau folgt der gewellten Papierkante. Prüfe am Original, ob der ganze Abzug sichtbar ist.";
export const adjacentPrintHint = "Prüfe am Original, ob der gestrichelte Bereich ein eigener Abzug ist und ob der helle Streifen zu seinem Papier gehört.";
export const crossingPrintReason = "CROSSING_PRINT_OUTLINE_REVIEW";
export const crossingPrintHint = "Prüfe am Original, ob die gestrichelte Kontur einen eigenen Abzug zeigt. Falls die durchgezogene Markierung falsch ist, lade die gestrichelte Kontur und setze die Ecken an den Papierrand.";
export const crossingPrintSaveHint = "Korrigiere und speichere zuerst die vier Ecken unter «Ganzer Abzug».";
const printEdgeReviewHints = {
  PRINT_SHADOW_EDGE_REVIEW: "Prüfe die markierte Kante am Original. Achte darauf, wo der Schatten endet und der Abzug beginnt.",
  PRINT_FAINT_OUTER_RIM_REVIEW: "Prüfe am Original, ob der helle Rand zur markierten Papierkante des Abzugs gehört.",
  PRINT_OUTER_EDGE_CONFLICT_REVIEW: "Aussen sind mehrere Kanten sichtbar. Lege die Ecken im Editor an die tatsächliche Papierkante.",
  PRINT_EXTENDED_OUTLINE_REVIEW: "Prüfe die erweiterte Kontur am Original. Setze die Ecken an den Papierrand, damit der ganze Abzug erhalten bleibt.",
  PRINT_BOUNDARY_UNSUPPORTED: "Prüfe den sichtbaren Papierrand am Original. Korrigiere die markierten Ecken, soweit die Kante erkennbar ist.",
};

export function getPrintEdgeReviewHint(fullPrint) {
  return fullPrint?.reason_codes?.map((reason) => printEdgeReviewHints[reason]).find(Boolean);
}

export function getFragmentOutline(mask) {
  const { quad, uncertain_edge: edge } = mask?.fragment_outline_suggestion || {};
  if (!Array.isArray(quad) || quad.length !== 4 || !Number.isInteger(edge) || edge < 0 || edge > 3
      || !quad.every((point) => [point?.x, point?.y].every((value) => Number.isFinite(value) && value >= 0 && value <= 1))) {
    return null;
  }
  return { quad, weakEdge: [quad[edge], quad[(edge + 1) % 4]] };
}

export function getCropModes(photo) {
  return photo?.enhancement_applied?.correction?.crop_modes || null;
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
  SCANNER_BACKING_UNCERTAIN: "Prüfe am Original, ob im markierten Bereich ein weiterer Abzug liegt. Ziehe die Ecken an seine Papierkante, wenn du sie erkennen kannst.",
  CROP_VALIDATION_UNAVAILABLE: "Prüfe und speichere den Zuschnitt, um den Download freizugeben.",
  CAMERA_ASPECT_REVIEW: "Die Vorschau nutzt die Kamerakalibrierung. Prüfe und bestätige die Fotoecken.",
  PRINT_AT_CAPTURE_EDGE: "Der Abzug reicht bis an den Aufnahmerand. Prüfe, ob er vollständig zu sehen ist.",
  PRINT_POSSIBLY_OCCLUDED: "Ein Teil des Abzugs könnte verdeckt sein.",
  PRINT_BOUNDARY_CURVED: "Die Abzugskante ist gekrümmt. Prüfe den Zuschnitt am Original.",
  [scallopedOutlineReviewReason]: scallopedOutlineReviewHint,
  PRINT_SCALLOPED_OWNERSHIP_UNCERTAIN: "Prüfe am Original, ob ein zweiter Abzug auf dem gezackten Papier aufliegt. Passe bei getrennten Abzügen beide Zuschnitte einzeln an.",
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
  IMAGE_BOUNDARY_CURVED: "Die Motivkante ist gewölbt. Prüfe den Zuschnitt am Original.",
  WIDE_MOTIF_CONFIRMATION_REQUIRED: "Prüfe den Motivvorschlag im Desktop-Editor und speichere die Motivecken.",
  IMAGE_VISIBILITY_UNCERTAIN: "Das vollständige Bildmotiv konnte nicht bestätigt werden.",
  IMAGE_BORDER_CLASSIFICATION_DISABLED: "Die automatische Bestimmung der Motivgrenze ist deaktiviert.",
  IMAGE_RECTANGLE_INCONSISTENT: "Die Motivecken passen nicht zum entzerrten Abzug. Prüfe die Motivgrenze.",
  BORDER_PROTECTED_CONTENT_VETO: "An der Motivgrenze könnten Bildinhalte liegen. Prüfe den Zuschnitt.",
};

export function CropVariants({ photo, hidden = false }) {
  const [mode, setMode] = useState(() => !photo.urls?.full_print && photo.urls?.image_only ? "image_only" : "full_print");
  const label = cropModeLabels[mode];
  const result = getCropModes(photo)?.[mode];
  const url = result?.status === "ready" ? photo.urls?.[mode] : null;
  const possibleMultipanel = photo.enhancement_applied?.segmentation_review_reasons?.includes("MULTIPANEL_PRINT_REVIEW");
  const possibleFragmentedPrint = photo.segmentation_confidence != null
    && photo.enhancement_applied?.segmentation_review_reasons?.includes("FRAGMENTED_PRINT_REVIEW");
  const possibleFragmentOutline = photo.segmentation_confidence != null
    && photo.enhancement_applied?.segmentation_review_reasons?.includes(fragmentOutlineReason);
  const possibleAdjacentPrint = photo.segmentation_confidence != null && photo.segmentation_mask?.adjacent_print_suggestion;
  const possibleCrossingPrint = photo.segmentation_confidence != null && photo.segmentation_mask?.crossing_print_suggestion;
  const scannerBackingUncertain = photo.enhancement_applied?.segmentation_review_reasons?.includes("SCANNER_BACKING_UNCERTAIN");
  const clippedOverlapUncertain = photo.segmentation_confidence != null
    && photo.enhancement_applied?.segmentation_review_reasons?.includes(clippedOverlapReason);
  const captureEdgeClipped = photo.segmentation_mask?.capture_edge_clipped === true;
  const pendingPaperEdge = getCropModes(photo)?.full_print?.reason_codes?.includes("PRINT_EDGE_CONFIRMATION_REQUIRED");
  const scallopedOutline = mode === "full_print" && result?.reason_codes?.includes(scallopedOutlineReviewReason);
  const printOutline = photo.urls?.full_print_preview;
  const fullPreview = printOutline || photo.urls?.source_corrected_untrimmed || photo.urls?.original;
  const motifPreview = photo.urls?.image_only_preview;
  const preview = mode === "image_only"
    ? motifPreview || ((possibleMultipanel || possibleFragmentedPrint || possibleFragmentOutline || scannerBackingUncertain || clippedOverlapUncertain
        || captureEdgeClipped || pendingPaperEdge) && fullPreview)
    : fullPreview;
  const previewAlt = mode === "image_only" && motifPreview
    ? "Konturgetreue Motivvorschau zur Prüfung"
    : preview && preview === printOutline
      ? "Vorschau der Papierkontur zur Prüfung"
      : "Vorschau zur Prüfung";
  const reviewMessage = (captureEdgeClipped
    && "Der vorgeschlagene Abzug reicht bis an den Aufnahmerand. Lade für den Zuschnitt eine Aufnahme hoch, die ihn ganz zeigt.")
    || (possibleMultipanel && reviewMessages.MULTIPANEL_PRINT_REVIEW)
    || (possibleFragmentedPrint && reviewMessages.FRAGMENTED_PRINT_REVIEW)
    || (possibleFragmentOutline && fragmentOutlineHint)
    || (possibleAdjacentPrint && adjacentPrintHint)
    || (possibleCrossingPrint && crossingPrintHint)
    || (scannerBackingUncertain && reviewMessages.SCANNER_BACKING_UNCERTAIN)
    || (clippedOverlapUncertain && clippedOverlapHint)
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
          {preview && <img className="crop-variant-image" src={preview} alt={previewAlt} />}
        </>
      )}
    </section>
  );
}
