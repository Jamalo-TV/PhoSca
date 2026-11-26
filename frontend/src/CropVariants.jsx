import { Download } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { filmSourceWindow, getPaperGridReview } from "./cropReviewOutlines.js";

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
export const filmGridRimHint = "Die gepunktete Linie markiert einen möglichen durchsichtigen Saum. Prüfe dort, ob der Untergrund durchscheint. Eine gemeinsame Schutzfolie kann auch mehrere Papierabzüge bedecken.";
export const countGroupReviewError = "Die Vorschläge haben sich geändert. Prüfe die aktualisierte Albumseite.";
export const paperJoinCountHint = "Die gestrichelten Abschnitte zeigen Materialwechsel am äusseren Rand. Prüfe, ob die beiden Vorschläge zu einem Gegenstand gehören und ob er ein Fotoabzug ist.";
export const paperJoinReason = "PAPER_JOIN_REVIEW";
export const paperJoinHint = "Prüfe den ganzen Papierrand am Original. Gehören beide Vorschläge zu einem Fotoabzug, setze die Ecken eines Vorschlags an seinen Aussenrand und prüfe den anderen als möglichen Teilvorschlag.";
export const paperGridReason = "PAPER_GRID_REVIEW";
export const paperGridCountHint = "Die blau gestrichelte Kontur zeigt einen möglichen Aussenumriss. Prüfe am Original, ob die Ausschnitte an den gepunkteten Linien eigene Papierabzüge sind.";
export const paperGridCountSaved = "Als ein Abzug gezählt. Die Papierkontur bleibt in Prüfung.";
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
export const nestedFallbackProposal = "nested_fallback_contour";
export const frameBackingProposal = "possible_frame_backing";
export const borrowedFallbackProposal = "possible_borrowed_print_edge";
export const darkBackingProposal = "possible_dark_album_backing";
export const standalonePrintConfirmation = "Hier liegt ein eigener Papierabzug.";
const deferredPrintReviewHints = {
  [nestedFallbackProposal]: {
    countId: "nested-fallback-count-hint",
    count: "Die markierten Innenkonturen liegen in erfassten Abzügen. Prüfe am Original, ob dort eigene Papierabzüge aufliegen. Sind alle Abzüge erfasst, bestätige die Anzahl.",
    draft: "Erfasse die Innenkontur nur, wenn im Original ein eigener Papierabzug aufliegt. Setze die vier Ecken an seinen Papierrand.",
    evidence: "Türkis markiert zu prüfende Abschnitte innerhalb der Abzüge.",
  },
  [frameBackingProposal]: {
    countId: "frame-backing-count-hint",
    count: "Prüfe am Original, ob der Vorschlag am Aufnahmerand über Albumseite und Untergrund läuft oder ob dort ein eigener Papierabzug liegt.",
    draft: "Erfasse den Vorschlag nur, wenn dort ein eigener Papierabzug liegt. Setze die vier Ecken an seinen sichtbaren Papierrand.",
    evidence: "Türkis markiert den gemessenen Materialwechsel. Prüfe auch, ob zwischen Linie und Aufnahmerand ein schmaler Papierabzug liegt.",
  },
  [borrowedFallbackProposal]: {
    countId: "borrowed-edge-count-hint",
    count: "Der Vorschlag folgt einer Kante eines erfassten Abzugs und reicht bis zum Aufnahmerand. Prüfe am Original, ob dort ein eigener Papierabzug liegt.",
    draft: "Setze die vier Ecken an den sichtbaren Papierrand, wenn du hier einen eigenen Abzug erkennst.",
    evidence: "Orange markiert die gemeinsame Kante, Türkis die Linie im erfassten Abzug und Blau den Abschnitt mit ähnlich hellem Material auf beiden Seiten.",
  },
};
const deferredParentCropHints = [
  {
    reason: "NESTED_FALLBACK_CONTOUR_REVIEW", id: "nested-fallback-crop-hint",
    text: "Prüfe am Original, ob innerhalb dieses Umrisses ein eigener Papierabzug aufliegt. Vergleiche beim Zuschneiden auch den Aussenrand des erfassten Abzugs.",
  },
  {
    reason: "FRAME_BACKING_COUNT_REVIEW", id: "frame-backing-crop-hint",
    text: "Prüfe am Original, wo der Papierabzug endet und der Untergrund beginnt. Achte beim Zuschneiden auf den Bildinhalt am Aufnahmerand.",
  },
];
export const darkBackingReviewErrors = {
  dark_backing_confirmation_required: "Bestätige, dass hier ein eigener Papierabzug liegt.",
  dark_backing_confirmation_stale: "Dieser Vorschlag hat sich geändert. Prüfe die aktualisierte Albumseite.",
  dark_backing_outline_outside_proposal: "Setze die Ecken an die Papierkante des markierten Vorschlags.",
};
const darkBackingEvidence = [
  { key: "dark_backing_traces", label: "Ähnlich dunkles Material beidseits", color: "#0891b2", dash: [9, 4] },
  { key: "borrowed_edge_traces", label: "Bereich am hellen Rand eines Nachbarabzugs", color: "#c2410c", dash: [2, 4] },
];
const nestedFallbackEvidence = [
  { key: "interior_edge_traces", label: "Zu prüfender Abschnitt innerhalb des Abzugs", color: "#0e7490", dash: [6, 4] },
];
const frameBackingEvidence = [
  { key: "surface_edge_traces", label: "Gemessener Materialwechsel am Aufnahmerand", color: "#0e7490", dash: [6, 4] },
];
const borrowedFallbackEvidence = [
  { key: "borrowed_edge_traces", label: "Gemeinsame Kante mit einem erfassten Abzug", color: "#c2410c", dash: [3, 4] },
  { key: "interior_edge_traces", label: "Zu prüfender Abschnitt innerhalb des Abzugs", color: "#0e7490", dash: [6, 4] },
  { key: "exposed_edge_traces", label: "Ähnlich helles Material auf beiden Seiten", color: "#2563eb", dash: [8, 4] },
];
export const sourceFragmentProposal = "possible_source_fragmented_print";
const sourceFragmentGroupReason = "SOURCE_FRAGMENT_GROUP_REVIEW";
export const sourceFragmentCountHint = "Prüfe am Original, ob die beiden Teilvorschläge zu einem Abzug gehören. Prüfe besonders die kurz gestrichelte Papierkante.";
export const sourceFragmentReviewErrors = {
  source_print_fragments_stale: "Die Teilvorschläge haben sich geändert. Prüfe die aktuelle Seite erneut.",
  deferred_print_proposal_stale: "Der Vorschlag ist nicht mehr aktuell. Gehe zurück zur Anzahlprüfung und prüfe die aktualisierte Seite.",
  source_print_fragments_confirmation_required: "Bestätige, dass beide Teilvorschläge zu diesem Abzug gehören.",
  source_print_outline_excludes_fragments: "Prüfe die vier Ecken am Original. Der Umriss muss zu den gemessenen Kanten und den beiden Teilvorschlägen passen.",
};
export const crossingPrintReason = "CROSSING_PRINT_OUTLINE_REVIEW";
export const crossingPrintHint = "Prüfe am Original, ob die gestrichelte Kontur einen eigenen Abzug zeigt. Falls die durchgezogene Markierung falsch ist, lade die gestrichelte Kontur und setze die Ecken an den Papierrand.";
export const crossingPrintSaveHint = "Korrigiere und speichere zuerst die vier Ecken unter «Ganzer Abzug».";
export const printOutlineSaveHint = "Prüfe und speichere zuerst die Papierkontur unter «Ganzer Abzug».";
export const cropProtectionErrors = {
  album_page_mark_review_stale: "Die Schutzmarkierung hat sich geändert. Öffne die aktuelle Prüfung erneut.",
  album_page_mark_choices_required: "Bestätige alle markierten Stellen nur, wenn jede davon ausserhalb des Abzugs auf dem Hintergrund liegt.",
  album_page_mark_outline_changed: "Lass die gemessenen Abzugecken für diese Hintergrundprüfung unverändert.",
  album_page_mark_inside_crop: "Setze die Ecken so, dass die Mitte der markierten Stelle ausserhalb des Abzugs liegt.",
  protected_region_cut: "Der Ausschnitt schneidet einen geschützten Bildbereich ab. Prüfe die sichtbaren Bereiche im Original.",
  protected_region_removed: "Der neue Zuschnitt würde markierten Bildinhalt abschneiden. Verschiebe die Ecken, bis die Markierungen im Ausschnitt liegen.",
  shared_rim_correction_required: "Setze die beiden Ecken an dieser Markierung an den sichtbaren Papierrand. Bleibt die Zuordnung unklar, fotografiere die Abzüge getrennt.",
};
export const captureOutlineReviewReason = "PRINT_CAPTURE_OUTLINE_REVIEW";
const captureAtEdgeReason = "PRINT_AT_CAPTURE_EDGE";
const competingPrintEdges = "competing_print_edges";
const sharedPrintRim = "shared_print_rim";
const sharedPrintRimReason = "PRINT_SHARED_RIM_UNCERTAIN";
const possibleBackedPaperEdge = "possible_backed_paper_edge";
export const possibleOverextendedPrintEdge = "possible_overextended_print_edge";
const possibleInwardPrintEdge = "possible_inward_print_edge";
const possibleInwardPrintCorner = "possible_inward_print_corner";
const possibleBowedPrintEdge = "possible_bowed_print_edge";
const possibleDisplacedPrintEdges = "possible_displaced_print_edges";
const possibleAlbumLayerEdge = "possible_album_layer_edge";
const possibleLocalPrintEdge = "possible_local_print_edge";
const disputedGridPreview = "disputed_paper_grid_outline_review";
const curvedPrintOutlineReviewReason = "PRINT_CURVED_OUTLINE_REVIEW";
const captureAtEdgeHint = "Der Abzug reicht über den Aufnahmerand hinaus. Für einen vollständigen Zuschnitt brauchst du eine neue Aufnahme.";
export const captureSplitReviewHint = "Die gestrichelten Linien zeigen einen möglichen gemeinsamen Papierrand. Prüfe am Original, wie viele Abzüge dort liegen.";
export const missingPrintEdgeHint = "Die blauen Abschnitte zeigen sichtbare Kanten. Prüfe am Original, ob dort ein eigener Papierabzug liegt.";
const curvedMotifReviewReason = "IMAGE_BOUNDARY_CURVED";
const curvedMotifReviewHint = "Die Motivkante verläuft unregelmässig. Prüfe am Original, welche Bildfelder und Beschriftungen zum Motiv gehören.";
const markedMotifMarginReason = "IMAGE_MARGIN_MARKS_REVIEW";
const markedMotifMarginHint = "Am Papierrand sind mögliche Beschriftungen sichtbar. Prüfe den ganzen Rand am Original und nimm alle zum Bild gehörenden Beschriftungen in den Zuschnitt auf.";
export const heldPrintMotifMargin = "held_print_motif_margin";
const manualNeighborOverlap = "MANUAL_NEIGHBOR_OVERLAP";
const manualNeighborContentLoss = "MANUAL_NEIGHBOR_CONTENT_LOSS";
const mountedPrintOutlineReason = "MOUNTED_PRINT_OUTLINE_REVIEW";
const projectivePrintOutlineReason = "PROJECTIVE_PRINT_OUTLINE_REVIEW";
const darkBackingOutlineReason = "DARK_BACKING_OUTLINE_REVIEW";
const printEdgeReviewHints = {
  PRINT_EDGE_CONFIRMATION_REQUIRED: printOutlineSaveHint,
  PRINT_SHADOW_EDGE_REVIEW: "Prüfe die markierte Kante am Original. Achte darauf, wo der Schatten endet und der Abzug beginnt.",
  PRINT_FAINT_OUTER_RIM_REVIEW: "Prüfe am Original, ob der helle Rand zur markierten Papierkante des Abzugs gehört.",
  [sharedPrintRimReason]: "Prüfe die Kante am benachbarten Abzug. Wenn der Papierrand unklar bleibt, hilft eine Aufnahme mit getrennten Abzügen.",
  PRINT_OUTER_EDGE_CONFLICT_REVIEW: "Aussen sind mehrere Kanten sichtbar. Lege die Ecken im Editor an die tatsächliche Papierkante.",
  PRINT_EXTENDED_OUTLINE_REVIEW: "Prüfe die erweiterte Kontur am Original. Setze die Ecken an den Papierrand, damit der ganze Abzug erhalten bleibt.",
  PRINT_BOUNDARY_UNSUPPORTED: "Prüfe den sichtbaren Papierrand am Original. Korrigiere die markierten Ecken, soweit die Kante erkennbar ist.",
  PRINT_BOUNDARY_CURVED: "Die Abzugskante ist gekrümmt. Prüfe am Original, ob der Abzug für eine neue Aufnahme flach aufgelegt werden kann.",
  [curvedPrintOutlineReviewReason]: "Die Papierkante braucht eine weitere Prüfung. Setze die Ecken an die sichtbare Aussenkante. Bei einer Wölbung hilft eine neue Aufnahme mit flach liegendem Abzug.",
  [captureOutlineReviewReason]: "Die vollständige Papierkante ist noch unklar. Vergleiche die Ecken mit der Aussenkante im Original. Bei einer verdeckten oder gewölbten Kante hilft eine neue Aufnahme mit sichtbarem Rand rundum.",
  [captureAtEdgeReason]: captureAtEdgeHint,
  PAPER_GRID_OVERLAP_REVIEW: "Zwei Vorschläge überdecken dieselbe Stelle. Kläre zuerst die überlappenden Vorschläge am Original und berechne den Zuschnitt danach erneut.",
  [manualNeighborOverlap]: "Die violette Kontur zeigt den überlappenden Nachbarabzug. Prüfe beide Papierkanten im Original. Wenn ein Abzug den anderen verdeckt, fotografiere sie getrennt.",
  [manualNeighborContentLoss]: "Die violetten Flächen zeigen sichtbare Teile, die im Zuschnitt fehlen könnten. Vergleiche sie mit dem Original und prüfe die Papierkante. Verdeckt ein Abzug den anderen, fotografiere sie getrennt.",
  [mountedPrintOutlineReason]: "Die Teilvorschläge sind als ein Abzug erfasst. Prüfe die vier Papierkanten am Original und speichere die Ecken. Bei verdecktem Papier hilft eine Aufnahme ohne Fotoecken.",
  [projectivePrintOutlineReason]: "Der Abzug ist erfasst. Prüfe die vier sichtbaren Papierkanten am Original und speichere die Ecken. Bei einer verdeckten Papierkante hilft eine neue Aufnahme.",
  [darkBackingOutlineReason]: "Der Abzug ist erfasst. Prüfe die vier Papierkanten am Original und speichere die Ecken.",
};

export function getSourceEdgeTraces(traces) {
  return Array.isArray(traces) ? traces.filter((trace) => Array.isArray(trace) && trace.length >= 2
    && trace.every((point) => [point?.x, point?.y].every((value) => Number.isFinite(value) && value >= 0 && value <= 1))) : [];
}

export function getDeferredPrintEvidence(proposal) {
  const styles = proposal?.reason === darkBackingProposal ? darkBackingEvidence
    : proposal?.reason === nestedFallbackProposal ? nestedFallbackEvidence
    : proposal?.reason === frameBackingProposal ? frameBackingEvidence
    : proposal?.reason === borrowedFallbackProposal ? borrowedFallbackEvidence : [];
  return styles.map((style) => ({
    ...style, traces: getSourceEdgeTraces(proposal[style.key]).filter((trace) => trace.length === 2),
  })).filter(({ traces }) => traces.length);
}

export function standalonePrintNeedsConfirmation(reason) {
  return reason === darkBackingProposal || reason === nestedFallbackProposal || reason === frameBackingProposal
    || reason === borrowedFallbackProposal;
}

export function deferredPrintCountHintId(reason, mobile = false) {
  const countId = deferredPrintReviewHints[reason]?.countId;
  return countId && `${mobile ? "mobile-" : ""}${countId}`;
}

export function getDeferredParentCropHints(reviewReasons) {
  return deferredParentCropHints.filter(({ reason }) => reviewReasons?.includes(reason));
}

export function DeferredPrintReviewHints({ proposals, draft = false, mobile = false }) {
  return Object.entries(deferredPrintReviewHints).map(([reason, hint]) => {
    const matching = proposals.filter((proposal) => proposal.reason === reason);
    if (!matching.length) return null;
    return <p key={reason} id={draft ? undefined : deferredPrintCountHintId(reason, mobile)}
      className={mobile ? "quad-hint" : "crop-editor-hint"}>
      {draft ? hint.draft : hint.count}
      {matching.some((proposal) => getDeferredPrintEvidence(proposal).length > 0) && ` ${hint.evidence}`}
    </p>;
  });
}

export function DarkBackingEvidenceLegend({ proposals, mobile = false }) {
  const backing = proposals.filter(({ reason }) => reason === darkBackingProposal);
  if (!backing.length) return null;
  const evidence = darkBackingEvidence.filter(({ key }) => backing.some(
    (proposal) => getDeferredPrintEvidence(proposal).some((style) => style.key === key),
  ));
  return <div className={`dark-backing-evidence ${mobile ? "quad-hint" : "crop-editor-hint"}`}>
    <p>Prüfe am Original, ob die dunkle Fläche zur Albumseite gehört oder ein eigener Papierabzug ist.</p>
    {evidence.length > 0 && <ul>
      {evidence.map(({ key, label, color, dash }) => <li key={key}>
        <svg aria-hidden="true" width="32" height="12" viewBox="0 0 32 12">
          <line x1="0" x2="32" y1="6" y2="6" stroke={color} strokeWidth="3" strokeDasharray={dash.join(" ")} />
        </svg>
        <span>{label}</span>
      </li>)}
    </ul>}
  </div>;
}

const groupReviewLabels = {
  film: {
    open: "Negativhülle prüfen",
    confirm: "Drei Vorschläge aus der Abzugsliste nehmen",
    prompt: "Prüfe im Original, ob alle drei türkis markierten Vorschläge zur Negativhülle gehören. Bei einem Papierabzug oder bei Unsicherheit wähle «Abbrechen».",
  },
  paper: {
    open: "Gemeinsame Papierkante prüfen",
    confirm: "Beide Vorschläge aus der Abzugsliste nehmen",
    prompt: "Entferne die beiden türkis markierten Vorschläge gemeinsam, wenn sie zur selben Zeichnung oder zu einem anderen Gegenstand ohne Fotoabzug gehören. Für die Einzelkorrektur eines Fotoabzugs wähle «Abbrechen».",
  },
  crossing: {
    open: "Schrägen Vorschlag prüfen",
    confirm: "Diesen Vorschlag aus der Abzugsliste nehmen",
    prompt: "Orange markiert die gemessene Nachbarkante und die Linie, die ausserhalb des Vorschlags weiterläuft. Türkis markiert Linien in zwei Bildmotiven. Prüfe am Original, ob der hervorgehobene Vorschlag einen eigenen Papierabzug zeigt. Bei einem Abzug oder bei Unsicherheit wähle «Abbrechen».",
  },
  "paper-grid": {
    open: "Zusammengehörigkeit prüfen",
    confirm: "Als einen Abzug zählen",
    prompt: "Prüfe am Original, ob die markierten Teilvorschläge zum selben Papierabzug gehören. Sind es getrennte Abzüge oder bleibt es unklar, wähle «Abbrechen».",
  },
};

function FilmGroupSourceDetails({ sourceUrl, line, rim }) {
  const [source, setSource] = useState(null);
  useEffect(() => {
    if (!sourceUrl) return;
    const image = new Image();
    image.onload = () => setSource({ url: sourceUrl, width: image.naturalWidth, height: image.naturalHeight });
    image.src = sourceUrl;
    return () => { image.onload = null; };
  }, [sourceUrl]);
  if (source?.url !== sourceUrl) return null;
  const { width, height } = source;
  const details = [
    ["Zwischenraum im Raster", filmSourceWindow(line, width, height)],
    ["Möglicher durchsichtiger Saum", filmSourceWindow(rim, width, height)],
  ].filter(([, detail]) => detail);
  if (!details.length) return null;
  return <section className="film-source-details" aria-label="Vergrösserte Originalausschnitte">
    {details.map(([label, { view, ticks }]) => <figure key={label}>
      <svg viewBox={view.join(" ")} role="img" aria-label={label}>
        <image href={sourceUrl} width={width} height={height} />
        <g aria-hidden="true">
          {ticks.map(([[x1, y1], [x2, y2]]) => <g key={`${x1},${y1}`}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#134e4a" strokeWidth="4" vectorEffect="non-scaling-stroke" />
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#99f6e4" strokeWidth="2" vectorEffect="non-scaling-stroke" />
          </g>)}
        </g>
      </svg>
      <figcaption>{label}</figcaption>
    </figure>)}
  </section>;
}

export function CountGroupReview({ groups, selected, busy, sourceUrl, onSelect, onConfirm }) {
  if (!groups.length) return null;
  const labels = selected && groupReviewLabels[selected.kind];
  const exclusions = groups.filter(({ kind }) => kind !== "paper-grid");
  const sharedPaper = groups.filter(({ kind }) => kind === "paper-grid");
  return <div className="count-group-review">
    {selected ? <>
      <p className="crop-editor-hint" role="status">{labels.prompt}</p>
      {selected.kind === "film" && sourceUrl && <FilmGroupSourceDetails sourceUrl={sourceUrl} line={selected.line} rim={selected.rim} />}
      <button type="button" className="primary-button" disabled={busy}
        onClick={() => onConfirm(selected)}>
        {busy ? "Vorschläge werden geprüft …" : labels.confirm}
      </button>
      <button type="button" className="text-button" disabled={busy} onClick={() => onSelect(null)}>Abbrechen</button>
    </> : groups.map((group) => {
      const { key, kind } = group;
      const peers = kind === "paper-grid" ? sharedPaper : exclusions;
      return <button key={key} type="button" className="text-button" disabled={busy} onClick={() => onSelect(key)}>
        {`${groupReviewLabels[kind].open}${peers.length > 1 ? ` (${peers.indexOf(group) + 1})` : ""}`}
      </button>;
    })}
  </div>;
}

export function actionableCountGroups(filmGroups, paperGroups, paperGridGroups = [], crossingGroups = []) {
  return [
    ...filmGroups.map(group => ({ ...group, kind: "film" })),
    ...paperGroups.map(group => ({ ...group, kind: "paper" })),
    ...paperGridGroups.map(group => {
      const [parentId] = group.parents;
      const fragmentIds = [...group.cuts.keys()].sort();
      return { ...group, kind: "paper-grid", key: JSON.stringify(["paper-grid", group.key, parentId, ...fragmentIds]),
        parentId, fragmentIds, photoIds: new Set([parentId, ...fragmentIds]) };
    }),
    ...crossingGroups,
  ];
}

export const crossingFallbackTraceStyles = [
  { color: "#b45309", dash: [9, 5], label: "Gemeinsame Kante mit einem Nachbarabzug" },
  { color: "#b45309", dash: [9, 5], label: "Fortsetzung der Linie ausserhalb des Vorschlags" },
  { color: "#0891b2", dash: [3, 5], label: "Linie im ersten Bildmotiv" },
  { color: "#0891b2", dash: [3, 5], label: "Linie im zweiten Bildmotiv" },
];

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

function hasMeasuredEdgeReview(fullPrint, source) {
  return fullPrint?.suggested_edge_source === source || fullPrint?.outline_preview?.source === source;
}

export function paperOutlineRequiresReview(fullPrint) {
  return fullPrint?.status === "review_needed" && (fullPrint.suggested_warp_quad?.length === 4
    || fullPrint.suggested_edge_source === competingPrintEdges
    || fullPrint.suggested_edge_source === possibleBackedPaperEdge
    || fullPrint.suggested_edge_source === possibleOverextendedPrintEdge
    || fullPrint.suggested_edge_source === possibleInwardPrintEdge
    || hasMeasuredEdgeReview(fullPrint, possibleBowedPrintEdge)
    || hasMeasuredEdgeReview(fullPrint, possibleDisplacedPrintEdges)
    || hasMeasuredEdgeReview(fullPrint, possibleAlbumLayerEdge)
    || fullPrint.reason_codes?.some(reason => reason === "PRINT_EDGE_CONFIRMATION_REQUIRED"
      || reason === curvedPrintOutlineReviewReason
      || reason === "PAPER_GRID_OVERLAP_REVIEW" || reason === manualNeighborOverlap
      || reason === sourceFragmentGroupReason || reason === sharedPrintRimReason
      || reason === manualNeighborContentLoss) || paperOutlineNeedsConfirmation(fullPrint));
}

export function paperOutlineNeedsConfirmation(fullPrint) {
  return Boolean(fullPrint?.reason_codes?.some(reason => reason === mountedPrintOutlineReason
    || reason === projectivePrintOutlineReason || reason === darkBackingOutlineReason));
}

export function getPrintEdgeReviewHint(fullPrint) {
  if (fullPrint?.reason_codes?.includes(darkBackingOutlineReason)) {
    return printEdgeReviewHints[darkBackingOutlineReason];
  }
  if (fullPrint?.reason_codes?.includes(manualNeighborContentLoss)) {
    return printEdgeReviewHints[manualNeighborContentLoss];
  }
  if (fullPrint?.reason_codes?.includes(manualNeighborOverlap)) {
    return printEdgeReviewHints[manualNeighborOverlap];
  }
  if (fullPrint?.reason_codes?.includes(scallopedOwnershipReason)) {
    return scallopedOwnershipHint;
  }
  if (fullPrint?.suggested_outline_source === "adjacent_paper_stock") {
    return "Prüfe am Original, ob der zusätzlich vorgeschlagene Papierstreifen zu diesem Abzug gehört.";
  }
  if (fullPrint?.suggested_outline_source === possibleInwardPrintCorner) {
    return "Prüfe am Original, ob die gemessene Ecke und der ergänzte Bildbereich zum selben Abzug gehören.";
  }
  if (fullPrint?.suggested_warp_quad?.length === 4) {
    return "Vergleiche den vorgeschlagenen Aussenumriss mit dem Original. Setze die Ecken an die sichtbare Papierkante.";
  }
  if (hasMeasuredEdgeReview(fullPrint, possibleDisplacedPrintEdges)) {
    return "Die gestrichelten Abschnitte zeigen zwei Übergänge im Original. Prüfe, wo das Bild weitergeht und ob der helle Streifen zum Fotopapier gehört. Bei verdecktem Rand hilft eine Aufnahme mit getrennten, flach liegenden Abzügen.";
  }
  if (hasMeasuredEdgeReview(fullPrint, possibleAlbumLayerEdge)) {
    return "Die gestrichelte Linie folgt einer möglichen Albumlage bis neben den Abzug. Prüfe am Original, wo das Fotopapier endet und die Albumlage beginnt.";
  }
  if (hasMeasuredEdgeReview(fullPrint, possibleLocalPrintEdge)) {
    return "Die gestrichelte Linie markiert den Übergang am dunklen Streifen. Prüfe am Original, ob er zum Abzug, zu einem benachbarten Abzug oder zu einem Schatten gehört.";
  }
  if (hasMeasuredEdgeReview(fullPrint, possibleBowedPrintEdge)) {
    return "Prüfe am Original, ob sich die Papierkante wölbt, und verfolge den Papierrand rundum. Bleibt er unklar, fotografiere den Abzug flach aufgelegt.";
  }
  if (fullPrint?.reason_codes?.includes(curvedPrintOutlineReviewReason)) {
    return printEdgeReviewHints[curvedPrintOutlineReviewReason];
  }
  if (getSourceEdgeTraces(fullPrint?.exposed_overlap_edge_traces).length) {
    return "Türkis zeigt einen möglichen Papierrand ausserhalb des Zuschnitts. Prüfe am Original, wo das Fotopapier endet und welcher Abzug oben liegt.";
  }
  if (getSourceEdgeTraces(fullPrint?.suggested_edge_traces).length) {
    if (fullPrint.suggested_edge_source === sharedPrintRim) {
      return "Die gestrichelte Markierung zeigt die fragliche Zuschnittkante. Vergleiche sie am Original mit dem Rand des benachbarten Abzugs. Bleibt die Zuordnung unklar, fotografiere die Abzüge getrennt.";
    }
    if (fullPrint.suggested_edge_source === competingPrintEdges) {
      return "Die gestrichelten Linien zeigen mögliche Kanten im Original. Prüfe, welche das Fotopapier begrenzt.";
    }
    if (fullPrint.suggested_edge_source === possibleBackedPaperEdge) {
      return "Die gestrichelten Abschnitte zeigen eine mögliche Papierkante im Original. Prüfe auch den Abschnitt dazwischen.";
    }
    if (fullPrint.suggested_edge_source === possibleOverextendedPrintEdge) {
      return "Die gestrichelten Abschnitte zeigen eine mögliche Papierkante. Prüfe am Original, ob der Streifen daneben zum Abzug gehört.";
    }
    if (fullPrint.suggested_edge_source === possibleInwardPrintEdge) {
      return "Die gestrichelte Linie markiert einen weiter innen sichtbaren Abschnitt. Prüfe am Original, ob dort der Papierrand oder die Motivgrenze verläuft, und verfolge den Papierrand rundum.";
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
  if (motif.margin_mark_source === heldPrintMotifMargin) {
    return "Orange markiert mögliche Beschriftung auf dem Abzug. Prüfe zuerst die Aussenkante. Berücksichtige danach die gesamte zum Bild gehörende Beschriftung beim Motivzuschnitt.";
  }
  if (motif.outline_preview?.source === "paper_stock_motif") {
    return "Die Vorschau folgt der gemessenen Motivgrenze. Prüfe zuerst den Aussenumriss am Original.";
  }
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
  if (preview?.source === disputedGridPreview) {
    const countHint = preview.projected_grid_cuts?.length
      ? "So sieht der Aussenumriss aus, falls die markierten Teile zum selben Abzug gehören. Die gestrichelten Linien zeigen die Kanten der Teilvorschläge. Vergleiche sie mit dem Original."
      : "Vergleiche den vorgeschlagenen Aussenumriss mit den Papierkanten am Original.";
    const aspect = preview.reason_codes?.includes("PRINT_ASPECT_UNCERTAIN") ? " Das Seitenverhältnis ist geschätzt." : "";
    return countHint + aspect;
  }
  if (preview?.source === possibleDisplacedPrintEdges) {
    return "Die Prüfvorschau ergänzt an einer Seite sichtbaren Inhalt und blendet an der anderen einen hellen Streifen teilweise aus. Der Streifen kann zum Abzug gehören. Vergleiche beide Vorschauen mit dem Original.";
  }
  if (preview?.source === possibleAlbumLayerEdge) {
    return "Die Prüfvorschau blendet einen Streifen hinter der beobachteten Linie aus. Der helle Rand direkt am Bild bleibt sichtbar. Prüfe am Original, ob der ausgeblendete Streifen zum Album oder zum Abzug gehört.";
  }
  if (preview?.source === possibleLocalPrintEdge) {
    return "Die Prüfvorschau zeigt den zusätzlichen dunklen Streifen an der Markierung. Vergleiche seine Struktur mit dem Original.";
  }
  if (preview?.source === possibleBowedPrintEdge) {
    return "Die Prüfvorschau ergänzt nur den am Original gemessenen sichtbaren Streifen. Der vollständige Papierrand ist noch unklar.";
  }
  if (preview?.source === possibleInwardPrintCorner) {
    const aspect = preview.reason_codes?.includes("PRINT_ASPECT_UNCERTAIN") ? " Das Seitenverhältnis ist geschätzt." : "";
    return `Die Vorschau ergänzt die am Original gemessene Ecke.${aspect}`;
  }
  if (preview?.source === possibleOverextendedPrintEdge) {
    const aspect = preview.reason_codes?.includes("PRINT_ASPECT_UNCERTAIN") ? "Das Seitenverhältnis ist geschätzt. " : "";
    return aspect + (preview.protected_content_review ? "Prüfe vor einer Korrektur auch den Ausschnitt auf möglichen Inhaltsverlust."
      : "Vergleiche die mögliche Papierkante mit dem Original.");
  }
  if (!preview?.aspect_ratio_estimation) return null;
  return preview.reason_codes?.includes("PRINT_ASPECT_UNCERTAIN")
    ? "Vorgeschlagener Abzug mit geschätztem Seitenverhältnis."
    : "Vorschau des vorgeschlagenen Aussenumrisses.";
}

export function showsPaperEdgeComparison(fullPrint) {
  return [possibleOverextendedPrintEdge, possibleInwardPrintCorner, possibleBowedPrintEdge,
    possibleDisplacedPrintEdges, possibleAlbumLayerEdge, possibleLocalPrintEdge,
    disputedGridPreview].includes(fullPrint?.outline_preview?.source);
}

export function PaperEdgePreviewComparison({ currentUrl, possibleUrl, hint, outlinePreview }) {
  const wholeGrid = outlinePreview?.source === disputedGridPreview;
  const { width, height } = outlinePreview?.shape || {};
  const cuts = wholeGrid && width > 1 && height > 1
    ? getSourceEdgeTraces(outlinePreview.projected_grid_cuts).filter((trace) => trace.length === 2) : [];
  return <div className="paper-edge-comparison">
    <figure>
      <img src={currentUrl} alt="Bisherige Abzugsvorschau" />
      <figcaption>Bisherige Vorschau</figcaption>
    </figure>
    <figure>
      <div className="paper-edge-preview-image">
        <img src={possibleUrl} alt={wholeGrid ? "Prüfvorschau des gesamten möglichen Abzugs" : "Prüfvorschau entlang der möglichen Papierkante"} />
        {cuts.length > 0 && <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Mögliche Trennkanten der Teilvorschläge">
          {cuts.map((trace) => {
            const points = trace.map(({ x, y }) => `${x * (width - 1) + .5},${y * (height - 1) + .5}`).join(" ");
            return <g key={points}>
              <polyline points={points} stroke="#422006" strokeWidth="4" strokeDasharray="5 4" vectorEffect="non-scaling-stroke" />
              <polyline points={points} stroke="#fde047" strokeWidth="2" strokeDasharray="5 4" vectorEffect="non-scaling-stroke" />
            </g>;
          })}
        </svg>}
      </div>
      <figcaption>{wholeGrid ? "Möglicher Aussenumriss" : "Mögliche Papierkante"}</figcaption>
    </figure>
    {hint && <p>{hint}</p>}
  </div>;
}

export const offPrintAdvisorySource = "off_print_advisory";

function albumPageMarkBounds(mark) {
  if (!mark?.id) return [];
  const bounds = mark.source === offPrintAdvisorySource
    ? (Array.isArray(mark.regions) ? mark.regions.map(region => region?.bounds) : [])
    : [mark.bounds];
  return bounds.every(box => box && [box.x1, box.y1, box.x2, box.y2].every(Number.isFinite)
    && box.x1 >= 0 && box.y1 >= 0 && box.x2 <= 1 && box.y2 <= 1
    && box.x1 < box.x2 && box.y1 < box.y2) ? bounds : [];
}

export function useAlbumPageMarkReview(mark, photoId) {
  const bounds = useMemo(() => albumPageMarkBounds(mark), [mark]);
  const key = bounds.length ? JSON.stringify([photoId, mark.id, mark.source, mark.source_size, bounds]) : null;
  const [selection, setSelection] = useState(null);
  useEffect(() => { setSelection(null); }, [key]);
  const checkedIndices = key && selection?.key === key ? selection.indices : [];
  const allChecked = bounds.length > 0 && bounds.every((_, index) => checkedIndices.includes(index));
  const confirmation = allChecked ? {
    id: mark.id,
    indices: mark.source === offPrintAdvisorySource ? bounds.map((_, index) => index) : null,
  } : null;

  function setChecked(index, checked) {
    setSelection(current => {
      const indices = new Set(current?.key === key ? current.indices : []);
      if (checked) indices.add(index);
      else indices.delete(index);
      return { key, indices: [...indices] };
    });
  }

  return { bounds, checkedIndices, setChecked, confirmation, reset: () => setSelection(null) };
}

function AlbumPageMarkDetail({ bounds, sourceSize, sourceUrl, label }) {
  const { width, height } = sourceSize || {};
  if (!(width > 1 && height > 1) || !sourceUrl) return null;
  const { x1, y1, x2, y2 } = bounds;
  const left = x1 * (width - 1), right = x2 * (width - 1);
  const top = y1 * (height - 1), bottom = y2 * (height - 1);
  const pad = .65 * Math.max(right - left, bottom - top);
  const x = Math.max(0, left - pad), y = Math.max(0, top - pad);
  const view = [x, y, Math.min(width, right + pad) - x, Math.min(height, bottom + pad) - y];
  return <svg className="album-page-mark-detail" role="img" aria-label={label} viewBox={view.join(" ")}>
    <image href={sourceUrl} width={width} height={height} />
    <rect x={left} y={top} width={right - left} height={bottom - top} fill="none" stroke="#facc15" strokeWidth="2" />
  </svg>;
}

export function AlbumPageMarkReview({ mark, review, sourceUrl, disabled }) {
  const multiple = mark.source === offPrintAdvisorySource;
  return <fieldset className="print-size-fields album-page-mark-review">
    <legend>Schutzmarkierung prüfen</legend>
    <p>{multiple
      ? "Vergleiche jede gelb umrahmte Stelle mit dem Original. Bestätige sie nur, wenn sie ausserhalb dieses Papierabzugs auf dem Hintergrund liegt. Solange nicht alle Stellen bestätigt sind, bleiben sie beim Speichern geschützt. Lass die gemessenen Ecken für diese Prüfung unverändert."
      : "Vergleiche die gelb umrahmte Stelle mit dem Original. Prüfe danach die vier Ecken des Abzugs."}</p>
    {review.bounds.map((bounds, index) => <div className="album-page-mark-spot" key={index}>
      <AlbumPageMarkDetail bounds={bounds} sourceSize={mark.source_size} sourceUrl={sourceUrl}
        label={multiple ? `Vergrösserung von Stelle ${index + 1} im Original` : "Vergrösserung der Schutzmarkierung im Original"} />
      <label>
        <input type="checkbox" checked={review.checkedIndices.includes(index)}
          onChange={event => review.setChecked(index, event.target.checked)} disabled={disabled} />
        {multiple ? `Stelle ${index + 1} liegt auf dem Hintergrund ausserhalb dieses Abzugs.`
          : "Die markierte Stelle gehört zur Albumseite."}
      </label>
    </div>)}
  </fieldset>;
}

export function NeighborContentReview({ fullPrint, sourceUrl }) {
  const { width, height } = fullPrint?.neighbor_review_source_size || {};
  const outlines = getSourceEdgeTraces(fullPrint?.neighbor_content_loss_outlines);
  if (!(width > 1 && height > 1) || !outlines.length || !sourceUrl) return null;

  return <details className="neighbor-content-review">
    <summary>Markierte Stellen im Original ansehen</summary>
    {Boolean(fullPrint.neighbor_backing_color_match_pixels) &&
      <p>Die markierten Stellen haben eine ähnliche Farbe wie die Albumseite. Vergleiche die feinen Schatten und Papierkanten ringsum.</p>}
    <div className="neighbor-content-details">
      {outlines.map((outline, index) => {
        const points = outline.map(({ x, y }) => [x * (width - 1), y * (height - 1)]);
        const coordinatesX = points.map(([x]) => x), coordinatesY = points.map(([, y]) => y);
        const left = Math.max(0, Math.min(...coordinatesX) - 24), right = Math.min(width, Math.max(...coordinatesX) + 24);
        const top = Math.max(0, Math.min(...coordinatesY) - 24), bottom = Math.min(height, Math.max(...coordinatesY) + 24);
        return <svg key={index} role="img" aria-label="Markierter Bereich im Original"
          viewBox={[left, top, right - left, bottom - top].join(" ")}>
          <image href={sourceUrl} width={width} height={height} />
          <polyline points={points.map(([x, y]) => `${x},${y}`).join(" ")} fill="none"
            stroke="#a21caf" strokeWidth="2" vectorEffect="non-scaling-stroke" />
        </svg>;
      })}
    </div>
  </details>;
}

export const sourceStripColor = "#22d3ee";
export const sourceStripContrastColor = "#164e63";

export function getSourceStripReview(fullPrint) {
  const review = fullPrint?.source_strip_review;
  if (fullPrint?.status !== "review_needed" || !review) return null;
  const { width, height } = review.source_size || {};
  const [outline] = getSourceEdgeTraces([review.source_outline]);
  return Number.isFinite(width) && Number.isFinite(height) && width > 1 && height > 1 && outline?.length === 4
    ? { width, height, outline } : null;
}

export function SourceStripOutline({ points }) {
  return <g aria-label="Möglicher sichtbarer Bildstreifen" fill="none" strokeLinejoin="round" pointerEvents="none">
    <polygon points={points} stroke={sourceStripContrastColor} strokeWidth="4" vectorEffect="non-scaling-stroke" />
    <polygon points={points} stroke={sourceStripColor} strokeWidth="2" strokeDasharray="7 4" vectorEffect="non-scaling-stroke" />
  </g>;
}

export function SourceStripReview({ strip, sourceUrl, hintId }) {
  const { width, height, outline } = strip;
  const points = outline.map(({ x, y }) => [x * (width - 1), y * (height - 1)]);
  const xs = points.map(([x]) => x), ys = points.map(([, y]) => y);
  const stripWidth = Math.max(...xs) - Math.min(...xs);
  const stripHeight = Math.max(...ys) - Math.min(...ys);
  const lateralPad = Math.max(24, .08 * stripWidth);
  const left = Math.max(0, Math.min(...xs) - lateralPad);
  const top = Math.max(0, Math.min(...ys) - Math.max(96, 2 * stripHeight, stripWidth / 3));
  const right = Math.min(width, Math.max(...xs) + lateralPad);
  const bottom = Math.min(height, Math.max(...ys) + Math.max(24, .5 * stripHeight));
  return <section className="neighbor-content-review" aria-label="Möglicher Bildstreifen im Original">
    <p id={hintId}>Prüfe am Original, ob der markierte Streifen zum Abzug gehört. Wenn der Albumkarton Bildinhalt verdeckt, verwende eine unverdeckte Aufnahme.</p>
    <svg className="source-strip-detail" viewBox={[left, top, right - left, bottom - top].join(" ")}
      role="img" aria-label="Markierter Streifen mit Umgebung im Original">
      <image href={sourceUrl} width={width} height={height} />
      <SourceStripOutline points={points.map(([x, y]) => `${x},${y}`).join(" ")} />
    </svg>
  </section>;
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
  [sourceFragmentGroupReason]: "Prüfe in der Anzahlprüfung am Original, ob dieser Teil zu demselben Abzug gehört wie der zweite Vorschlag.",
  [crossingPrintReason]: crossingPrintHint,
  [scannerBackingReason]: scannerBackingHint,
  [filmGridReason]: filmGridHint,
  [paperJoinReason]: paperJoinHint,
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
  const possiblePaperJoin = photo.segmentation_confidence != null
    && photo.enhancement_applied?.segmentation_review_reasons?.includes(paperJoinReason);
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
  const printOutline = showsPaperEdgeComparison(getCropModes(photo)?.full_print)
    ? null : photo.urls?.full_print_preview;
  const fullPreview = printOutline || photo.urls?.source_corrected_untrimmed || photo.urls?.original;
  const motifPreview = photo.urls?.image_only_preview;
  const preview = mode === "image_only"
    ? motifPreview || ((possibleMultipanel || possibleFragmentedPrint || possibleFragmentOutline || scannerBackingUncertain || clippedOverlapUncertain || possibleFilmGrid || possiblePaperJoin
        || paperGrid || captureEdgeClipped || captureSplit || pendingPaperEdge || result?.reason_codes?.includes(markedMotifMarginReason)) && fullPreview)
    : fullPreview;
  const previewAlt = mode === "image_only" && motifPreview
    ? "Motivvorschau zur Prüfung"
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
    || (mode === "image_only" && result?.outline_preview?.source === "paper_stock_motif" && getMotifBoundaryReviewHint(result))
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
