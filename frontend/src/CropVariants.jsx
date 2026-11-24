import { Download } from "lucide-react";
import { useState } from "react";

export const cropModeLabels = {
  full_print: "Ganzer Abzug",
  image_only: "Nur Bildmotiv",
};

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
  MULTIPANEL_PRINT_REVIEW: "Hier könnten mehrere Bilder auf einem Abzug liegen. Prüfe die gemeinsame Papierkante am Original und achte darauf, dass der Motivzuschnitt alle Bilder enthält.",
  CROP_VALIDATION_UNAVAILABLE: "Prüfe und speichere den Zuschnitt, um den Download freizugeben.",
  CAMERA_ASPECT_REVIEW: "Die Vorschau nutzt die Kamerakalibrierung. Prüfe und bestätige die Fotoecken.",
  PRINT_AT_CAPTURE_EDGE: "Der Abzug reicht bis an den Aufnahmerand. Prüfe, ob er vollständig zu sehen ist.",
  PRINT_POSSIBLY_OCCLUDED: "Ein Teil des Abzugs könnte verdeckt sein.",
  PRINT_BOUNDARY_CURVED: "Die Abzugskante ist gekrümmt. Prüfe den Zuschnitt am Original.",
  PRINT_BOUNDARY_BLURRED: "Die Abzugskante ist unscharf. Prüfe den Zuschnitt am Original.",
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
  const fullPreview = photo.urls?.source_corrected_untrimmed || photo.urls?.original;
  const motifPreview = photo.urls?.image_only_preview;
  const preview = mode === "image_only"
    ? motifPreview || (possibleMultipanel && fullPreview)
    : fullPreview;
  const reviewMessage = (possibleMultipanel && reviewMessages.MULTIPANEL_PRINT_REVIEW)
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
          {preview && <img className="crop-variant-image" src={preview}
            alt={mode === "image_only" && motifPreview
              ? "Konturgetreue Motivvorschau zur Prüfung" : "Vorschau zur Prüfung"} />}
        </>
      )}
    </section>
  );
}
