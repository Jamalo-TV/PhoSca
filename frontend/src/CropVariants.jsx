import { Download } from "lucide-react";
import { useState } from "react";

export const cropModeLabels = {
  full_print: "Ganzer Abzug",
  image_only: "Nur Bildmotiv",
};

export function getCropModes(photo) {
  return photo?.enhancement_applied?.correction?.crop_modes || null;
}

const reviewMessages = {
  SEGMENTATION_CONFIDENCE_UNCALIBRATED: "Prüfe und speichere die Abzugsecken, um den Download freizugeben.",
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
  PRINT_ASPECT_UNCERTAIN: "Das Seitenverhältnis lässt sich aus dieser Aufnahme nicht sicher bestimmen.",
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
  const preview = mode === "image_only"
    ? photo.urls?.image_only_preview
    : photo.urls?.source_corrected_untrimmed || photo.urls?.original;
  const reviewMessage = result?.reason_codes?.map((code) => reviewMessages[code]).find(Boolean)
    || (result?.status === "ready" && !url && reviewMessages.CROP_VALIDATION_UNAVAILABLE)
    || (mode === "image_only" ? reviewMessages.IMAGE_BOUNDARY_UNCERTAIN : "Die Aussenkante des Abzugs muss geprüft werden.");

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
          <img key={url} className="crop-variant-image" src={url} alt={label} />
          <a className="crop-download" href={url} download={`photo-${photo.id}-${mode}.jpg`}>
            <Download size={16} /> {label} herunterladen
          </a>
        </>
      ) : (
        <>
          <p className="crop-mode-status" role="status">{reviewMessage}</p>
          {preview && <img className="crop-variant-image" src={preview}
            alt={mode === "image_only" ? "Konturgetreue Motivvorschau zur Prüfung" : "Vorschau zur Prüfung"} />}
        </>
      )}
    </section>
  );
}
