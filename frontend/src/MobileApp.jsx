import { Camera, Check, ChevronLeft, CircleHelp, CloudOff, Images, ListChecks, RefreshCw, Save, Upload, WandSparkles, ZoomIn } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { addManualPhoto, confirmSharedPaperGrid, createAlbum, excludeCountReviewGroup, getPhoto, getReviewProgress, getReviewQueue, hasPhotoCrop, listAlbums, listPages, listPhotos, patchBoundingBox, reprocessPhoto, submitReviewEvent, uploadFileChunked, uploadFiles } from "./api.js";
import { captureCameraImage } from "./cameraCapture.js";
import { motifUsesPaperAxes, projectedCropAspect, resizeMotifInPaperPlane, resamplePerspective, samplePreviewSource } from "./cropPreview.js";
import EdgeMeasurement, { MeasuredEdgeOverlay } from "./EdgeMeasurement.jsx";
import {
  contactEdgesNeedRefresh, contrastOutlineConfirmation, contrastOutlineNeedsConfirmation, contrastOutlineReviewHint,
  heldPrintCorrectionHint, wholePrintApprovalHeld,
} from "./cropReviewApproval.js";
import {
  actionableCountGroups, adjacentPrintHint, albumLeafChildStyle, albumLeafTraceStyle, AlbumPageMarkReview, canEditMobileMotif, captureRequiresReview,
  captureBackgroundReviewAction, captureBackgroundReviewHint, captureBackgroundWrongObjectHint,
  captureSplitReviewHint, CaptureSplitSourceReview, clippedOverlapHint, clippedOverlapReason, CropOrientationField, cropProtectionErrors, CropVariants,
  CompetingPhotoOutlineLegend, CompetingPhotoOutlineOverlay, SharedMotifGuideOverlay,
  cropModeLabels, crossingFallbackTraceStyles, crossingPrintHint,
  countGroupReviewError, CountGroupReview, crossingPrintReason, crossingPrintSaveHint, filmGridCountHint, filmGridHint, filmGridRimHint,
  filmGridReason, fragmentCountHint,
  fragmentOutlineHint, fragmentOutlineReason, getCropModes,
  borrowedFallbackProposal, captureFabricProposal, deferredPrintHint, deferredPrintCountHintId, DeferredPrintReviewHints, frameBackingProposal,
  getDeferredParentCropHints, getDeferredPrintEvidence, getDeferredPrintProposals, getSourceEdgeTraces,
  getMotifBoundaryReviewHint, getMotifMarginMarks, getPrintEdgeReviewHint, getPrintOutlinePreviewHint, heldPrintMotifMargin, mergedPrintAsOneLabel, mergedPrintPairHint, missingPrintEdgeHint,
  nearCapturePrintEdgeHint,
  paperOutlineNeedsConfirmation, NeighborContentReview, nestedPrintHint, PaperEdgePreviewComparison, showsPaperEdgeComparison, printOutlineSaveHint,
  paperGridCountHint, paperGridCountSaved, paperGridFragmentHint, paperGridOuterHint, paperGridReason, paperJoinCountHint, paperJoinHint,
  paperJoinReason, paperOutlineRequiresReview, partialMotifSeams, printedIllustrationFrameStyle,
  isNestedFallbackProposal, requiresStandalonePrintFlag, projectivePrintProposal, standalonePrintReviewErrors, tiledPrintProposal,
  scannerBackingHint, scannerBackingReason,
  sourceFragmentCountHint, sourceFragmentProposal, sourceFragmentReviewErrors,
  standalonePrintConfirmation, standalonePrintNeedsConfirmation, DarkBackingEvidenceLegend, darkBackingProposal, darkBackingReviewErrors,
  embossedAlbumAction, embossedPocketReviewErrors,
  getSourceStripReview, SourceStripOutline, SourceStripReview, offPrintAdvisorySource, useAlbumPageMarkReview,
  savedPrintReviewHint, savedPrintReviewTitle,
} from "./CropVariants.jsx";
import {
  getCompetingPhotoOutline, getCompetingPhotoReviewOutlines,
  captureBackgroundTraceStyle, contrastCrossingTraceHint, contrastCrossingTraceStyle, filmNeighborEdgeColor,
  getCaptureBackgroundReviewTraces, getCompactSceneCrossingTraces,
  getContrastCrossingTraces, getFilmNeighborEdgeIndex,
  getAlbumLeafReviewGroups, getCaptureSplitReviewGroups, getCrossingFallbackReviewGroups, getFilmGridReviewSeams, getFilmGridSeam, getFragmentOutline, getFragmentReviewOutlines,
  getActionableFilmGridReviewGroups, getActionablePaperGridReviewGroups, getPaperGridOutlineParents, getPaperGridReview, getPaperGridReviewGroups,
  getMergedPrintPairs, getPaperJoinReviewGroups, getPrintedIllustrationReviewGroups, getStoredMotifQuad, motifFitsPaper,
  getRibbedPrintReviewOutline, measuredRibbedPrintOutlineSource, ribbedPrintOutlineHint, ribbedPrintOutlineStyle,
  savedFullPrintReviewUrl,
} from "./cropReviewOutlines.js";
import { PrintSizeFields, parsePrintSize, printSizeInputs } from "./PrintSizeFields.jsx";
import { listQueuedUploads, queueUpload, removeQueuedUpload, retryQueuedUploads } from "./offlineQueue.js";

const cropAnswers = {
  segmentation: [
    ["correct", "Alle Prints gefunden"], ["missing_print", "Print fehlt"],
    ["too_many", "Zu viele Erkennungen"], ["unsure", "Unsicher"],
  ],
  geometry: [
    ["correct", "Korrekt & vollständig"], ["too_tight", "Zu eng"],
    ["too_wide", "Zu weit"], ["wrong_object", "Falsches Objekt"],
    ["split_print", "Print geteilt"], ["merged_prints", "Prints zusammen"],
    ["protected_cut", "Motiv abgeschnitten"], ["unsure", "Unsicher"],
  ],
};

const motifDesktopReviewHint = "Setze die Motivecken im Desktop-Editor, bevor du den Zuschnitt als vollständig bestätigst.";
const motifReviewHint = "Prüfe und speichere «Nur Bildmotiv», bevor du den Zuschnitt als vollständig bestätigst.";

function OnlineBadge() {
  const [online, setOnline] = useState(navigator.onLine);
  useEffect(() => {
    const update = () => setOnline(navigator.onLine);
    addEventListener("online", update); addEventListener("offline", update);
    return () => { removeEventListener("online", update); removeEventListener("offline", update); };
  }, []);
  return <span className={`mobile-status ${online ? "online" : "offline"}`}>{online ? "Online · synchronisiert" : "Offline · lokal gesichert"}</span>;
}

function MobileHeader({ title, onBack }) {
  return <header className="mobile-header"><div>{onBack && <button className="mobile-icon" onClick={onBack} aria-label="Zurück"><ChevronLeft /></button>}<span className="mobile-mark">P</span><strong>{title}</strong></div><OnlineBadge /></header>;
}

function Capture({ albums, activeAlbumId, setActiveAlbumId, reload, goHome, uploadQueueScope }) {
  const [queue, setQueue] = useState([]); const [message, setMessage] = useState(""); const [busy, setBusy] = useState(false);
  const [streaming, setStreaming] = useState(false); const [cameraError, setCameraError] = useState(""); const [lastCapture, setLastCapture] = useState("");
  const [startingCamera, setStartingCamera] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [usedVideoFrame, setUsedVideoFrame] = useState(false);
  const videoRef = useRef(null); const streamRef = useRef(null);
  const refreshQueue = useCallback(() => listQueuedUploads(uploadQueueScope).then(setQueue), [uploadQueueScope]);
  useEffect(() => { refreshQueue(); }, [refreshQueue]);
  useEffect(() => () => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
  }, []);
  useEffect(() => () => { if (lastCapture) URL.revokeObjectURL(lastCapture); }, [lastCapture]);

  function stopCameraForNativeCapture() {
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    videoRef.current.srcObject = null;
    setStreaming(false);
  }

  async function startCamera() {
    setCameraError("");
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) return setCameraError("Die Live-Kamera benötigt die HTTPS-Adresse dieser Seite.");
    setStartingCamera(true);
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 4096 }, height: { ideal: 3072 } }, audio: false });
      const video = videoRef.current;
      if (!video) { stream.getTracks().forEach(track => track.stop()); return; }
      streamRef.current?.getTracks().forEach(track => track.stop()); streamRef.current = stream;
      video.srcObject = stream;
      await video.play();
      if (stream === streamRef.current) setStreaming(true);
    } catch (error) {
      if (stream && stream === streamRef.current) stopCameraForNativeCapture();
      if (videoRef.current) setCameraError(error.name === "NotAllowedError" ? "Kamerazugriff wurde nicht erlaubt." : "Die Kamera konnte nicht gestartet werden.");
    } finally {
      if (videoRef.current) setStartingCamera(false);
    }
  }

  async function captureInBrowser() {
    const video = videoRef.current;
    const stream = streamRef.current;
    if (!video || !stream || busy || capturing) return;
    setCapturing(true);
    setCameraError("");
    try {
      const { file, fromVideo } = await captureCameraImage(video, .96);
      if (stream !== streamRef.current) return;
      setUsedVideoFrame(fromVideo);
      setLastCapture(URL.createObjectURL(file));
      navigator.vibrate?.(35);
      await receive([file]);
    } catch {
      if (stream === streamRef.current) setCameraError("Die Aufnahme ist fehlgeschlagen. Versuche es erneut oder verwende die Handy-Kamera.");
    } finally {
      if (stream === streamRef.current) setCapturing(false);
    }
  }

  async function addBatch() {
    const name = prompt("Name des neuen Aufnahme-Batches", `Album ${new Date().toLocaleDateString("de-CH")}`);
    if (!name) return; const album = await createAlbum({ name, description: "Mobiler Capture-Batch" }); setActiveAlbumId(album.id); await reload();
  }

  async function receive(files) {
    if (!activeAlbumId || !files.length) return; setBusy(true); setMessage("");
    try {
      for (const file of files) {
        try {
          if (!navigator.onLine) throw new Error("offline");
          const result = file.size > 20 * 1024 * 1024 ? await uploadFileChunked(activeAlbumId, file) : await uploadFiles(activeAlbumId, [file]);
          if (!result?.pages?.[0]?.verified) throw new Error("Serverbestätigung fehlt");
          setMessage(result.analysis_queued ? "Auf PC gespeichert. Die Albumseite wird analysiert." : "Auf PC gespeichert und geprüft.");
        } catch {
          await queueUpload(activeAlbumId, [file], uploadQueueScope); setMessage("Lokal gesichert – wartet auf Upload.");
        }
      }
      await refreshQueue();
      if (navigator.onLine) await reload().catch(() => {});
    } finally {
      setBusy(false);
    }
  }

  async function retry() {
    setBusy(true);
    try {
      await retryQueuedUploads(setMessage, uploadQueueScope);
      await refreshQueue();
      if (navigator.onLine) await reload().catch(() => {});
    } finally {
      setBusy(false);
    }
  }

  const captureBusy = busy || startingCamera || capturing;
  return <div className="mobile-shell"><MobileHeader title="Seiten aufnehmen" onBack={goHome} /><main className="mobile-content">
    <section className="mobile-card batch-picker"><label>Album / Batch</label><div><select value={activeAlbumId || ""} disabled={captureBusy} onChange={e => setActiveAlbumId(e.target.value)}><option value="" disabled>Batch auswählen</option>{albums.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}</select><button className="mobile-icon light" disabled={captureBusy} onClick={addBatch}>+</button></div></section>
    <section className="capture-stage"><div className={`live-camera ${streaming ? "streaming" : ""}`}><video ref={videoRef} autoPlay playsInline muted />{!streaming && <div className="page-guide"><Camera size={44}/><strong>Kamera direkt im Browser</strong><span>Die Aufnahme landet automatisch auf deinem PC.</span></div>}{streaming && <div className="camera-guide"><span>Ganze Albumseite mit etwas Rand ausrichten</span></div>}{lastCapture && <img className="last-capture" src={lastCapture} alt="Letzte Aufnahme" />}</div>
      {!streaming ? <button className="capture-button" disabled={!activeAlbumId || captureBusy} onClick={startCamera}><Camera/><span>Kamera starten</span></button> : <button className={`shutter-button ${captureBusy ? "busy" : ""}`} disabled={captureBusy || !activeAlbumId} onClick={captureInBrowser} aria-label="Foto jetzt aufnehmen"><span /></button>}
      {cameraError && <p className="camera-error" role="alert">{cameraError}</p>}
      {usedVideoFrame && <p className="camera-note" role="status">Die Browseraufnahme verwendet ein Videobild. Fehlen feine Papierkanten, nimm die Seite mit der Handy-Kamera auf.</p>}
      <label className="secondary-capture"><Camera size={19}/> Mit Handy-Kamera aufnehmen<input type="file" accept="image/jpeg,image/png" capture="environment" disabled={!activeAlbumId || captureBusy} onClick={stopCameraForNativeCapture} onChange={e => receive([...e.target.files])}/></label>
      <label className="secondary-capture"><Images size={19}/> Aus Galerie wählen<input type="file" accept="image/jpeg,image/png" multiple disabled={!activeAlbumId || captureBusy} onChange={e => receive([...e.target.files])}/></label>
    </section>{message && <div className="mobile-message"><Check size={18}/>{message}</div>}
    <section className="queue-card"><div><strong>Upload-Warteschlange</strong><span>{queue.length ? `${queue.length} ausstehend` : "Alles auf dem PC gespeichert"}</span></div>{queue.length > 0 && <button onClick={retry} disabled={captureBusy}><RefreshCw size={18}/> Erneut versuchen</button>}{queue.map(item => <div className="queue-row" key={item.id}><CloudOff/><span><b>{item.file.name}</b><small>{item.status} · Versuch {item.attempts || 0}</small></span><button onClick={() => removeQueuedUpload(item.id).then(refreshQueue)}>×</button></div>)}</section>
  </main></div>;
}

function heldDeferredParentCropHints(fullPrint, reviewReasons) {
  return fullPrint?.status === "review_needed" ? getDeferredParentCropHints(reviewReasons) : [];
}

function DetectionOverlay({ item, editable = false, openPaperOutline, countOutlines, captureSplitGroups, mergedPairs, filmGridGroups,
  selectedCountGroup, paperGridGroups, paperJoinGroups, onSaved, onReprocess, onProtectionBlocked, deferredDraft, onDeferredDraftClose }) {
  const sourceFragmentDraft = deferredDraft?.reason === sourceFragmentProposal;
  const darkBackingDraft = deferredDraft?.reason === darkBackingProposal;
  const standaloneDraft = standalonePrintNeedsConfirmation(deferredDraft?.reason);
  const [measurementPage, setMeasurementPage] = useState(null);
  const [measuredEdge, setMeasuredEdge] = useState(null);
  const [showMissingEdges, setShowMissingEdges] = useState(false);
  useEffect(() => setShowMissingEdges(false), [item.id]);
  useEffect(() => {
    setMeasurementPage(null);
    setMeasuredEdge(null);
  }, [item.page_id]);
  const measuredTraces = measuredEdge?.pageId === item.page_id ? measuredEdge?.traces : null;
  const nearCaptureTraces = getSourceEdgeTraces(item.payload?.near_capture_print_edge_traces);
  const missingTraces = [...getSourceEdgeTraces(item.payload?.missing_print_edge_traces), ...nearCaptureTraces];
  const visibleTraces = showMissingEdges ? [...missingTraces, ...(measuredTraces || [])] : measuredTraces;
  const sourceBox = item.payload?.bounding_box;
  const detections = sourceBox ? [{ bounding_box: sourceBox, photo_id: item.payload?.photo_id }] : (item.payload?.detections || []);
  const visiblePolygon = sanitizeReviewPolygon(item.payload?.visible_polygon || item.payload?.polygon);
  const storedMeasurement = item.payload?.print_size_mm ?? item.payload?.segmentation_mask?.print_size_mm;
  const storedOrientation = item.payload?.segmentation_mask?.orientation_degrees ?? null;
  const fragmentReview = item.payload?.confidence != null && item.payload?.review_reasons?.includes(fragmentOutlineReason);
  const fragmentOutline = fragmentReview && getFragmentOutline(item.payload?.segmentation_mask);
  const reviewedNonPrint = item.payload?.review_reasons?.includes("GEOMETRY_REVIEW_WRONG_OBJECT");
  const filmGrid = editable && !reviewedNonPrint && item.payload?.confidence != null
    && item.payload?.review_reasons?.includes(filmGridReason) && getFilmGridSeam(item.payload?.segmentation_mask);
  const paperGrid = editable && !reviewedNonPrint && item.payload?.confidence != null
    && item.payload?.review_reasons?.includes(paperGridReason) && getPaperGridReview(item.payload?.segmentation_mask);
  const paperJoinReview = editable && !reviewedNonPrint && item.payload?.confidence != null
    && item.payload?.review_reasons?.includes(paperJoinReason);
  const crossingOutline = item.payload?.confidence != null
    && item.payload?.review_reasons?.includes(crossingPrintReason)
    && !reviewedNonPrint
    && cyclicReviewQuad(sanitizeReviewPolygon(item.payload?.segmentation_mask?.crossing_print_suggestion));
  const fullPrintMode = reviewedNonPrint ? null : item.payload?.crop_modes?.full_print;
  const contrastOutlineReview = contrastOutlineNeedsConfirmation(item.payload, item.payload?.crop_modes?.full_print);
  const fullPrintHeld = !reviewedNonPrint && wholePrintApprovalHeld(item.payload?.crop_modes, item.payload?.variant_urls);
  const printEdgeReviewHint = getPrintEdgeReviewHint(fullPrintMode)
    || (fullPrintHeld && heldPrintCorrectionHint(fullPrintMode));
  const motifMode = item.payload?.crop_modes?.image_only;
  const motifReviewPending = !reviewedNonPrint && item.payload?.motif_review_pending;
  const motifBoundaryReviewHint = getMotifBoundaryReviewHint(motifMode);
  const stockMotifPreview = !reviewedNonPrint && motifMode?.status === "review_needed"
    && motifMode?.outline_preview?.source === "paper_stock_motif"
    && item.payload?.variant_urls?.image_only_preview;
  const motifMarginMarks = motifReviewPending ? getMotifMarginMarks(motifMode) : [];
  const preliminaryMotifMarks = motifMode?.margin_mark_source === heldPrintMotifMargin && motifMarginMarks.length > 0;
  const openMotifGuides = !reviewedNonPrint && motifMode?.status === "review_needed"
    && motifMode.suggested_seam_source === partialMotifSeams;
  const motifTraces = motifReviewPending || openMotifGuides ? getSourceEdgeTraces(motifMode?.suggested_seam_traces) : [];
  const outlinePreviewHint = getPrintOutlinePreviewHint(fullPrintMode);
  const printSuggestion = cyclicReviewQuad(sanitizeReviewPolygon(fullPrintMode?.suggested_warp_quad));
  const initialPaperOutline = deferredDraft?.polygon || (openPaperOutline && paperGrid?.cut === null && printSuggestion);
  const printEdgeTraces = getSourceEdgeTraces(fullPrintMode?.suggested_edge_traces);
  const filmEdgeIndex = getFilmNeighborEdgeIndex(fullPrintMode);
  const adjacentOutline = item.payload?.confidence != null
    && cyclicReviewQuad(sanitizeReviewPolygon(item.payload?.segmentation_mask?.adjacent_print_suggestion));
  const nestedOutlines = (sourceBox ? [item.payload] : detections)
    .map(record => cyclicReviewQuad(sanitizeReviewPolygon(record.segmentation_mask?.nested_print_suggestion)))
    .filter(Boolean);
  const deferredProposals = getDeferredPrintProposals(item.payload?.deferred_print_proposals);
  const scallopedAlternate = item.payload?.confidence != null && item.payload?.segmentation_mask?.scalloped_alternate_is_inner
    && cyclicReviewQuad(sanitizeReviewPolygon(item.payload?.segmentation_mask?.scalloped_alternate_polygon));
  const savedQuad = sanitizeReviewPolygon(item.payload?.segmentation_mask?.warp_quad);
  const measuredQuad = storedMeasurement && (savedQuad || sanitizeReviewPolygon(item.payload?.warp_quad));
  const markOffer = item.payload?.correction?.album_page_mark_review;
  const protectedOutline = markOffer?.source === offPrintAdvisorySource && sanitizeReviewPolygon(fullPrintMode?.warp_quad);
  const canEditMotif = editable && !reviewedNonPrint && !contrastOutlineReview && isValidQuad(savedQuad)
    && canEditMobileMotif(item.payload, item.payload?.crop_modes);
  const storedMotif = cyclicReviewQuad(sanitizeReviewPolygon(getStoredMotifQuad(item.payload || {}, motifMode)));
  const suggestedMotif = cyclicReviewQuad(sanitizeReviewPolygon(motifMode?.suggested_warp_quad));
  const initialOuter = () => (isValidQuad(protectedOutline) && protectedOutline) || initialPaperOutline
    || (isValidQuad(measuredQuad) ? measuredQuad : initialQuad(
    item.payload?.confidence === null && savedQuad ? savedQuad : item.payload?.warp_quad,
    visiblePolygon,
    sourceBox,
  ));
  const [quad, setQuad] = useState(initialOuter); const [geometrySaving, setSaving] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);
  const [draftTouched, setDraftTouched] = useState(false);
  const saving = geometrySaving || reprocessing;
  const [confirmDeferredDraft, setConfirmDeferredDraft] = useState(false);
  const [confirmMergedPrintAsOne, setConfirmMergedPrintAsOne] = useState(false);
  const [confirmedContrastOutline, setConfirmedContrastOutline] = useState(null);
  const contrastOutlineConfirmed = quad != null && confirmedContrastOutline === quad;
  const contrastOutlineNeedsCheck = contrastOutlineReview && !contrastOutlineConfirmed;
  const [editMode, setEditMode] = useState("full_print");
  const [motifQuad, setMotifQuad] = useState(storedMotif);
  const editingMotif = canEditMotif && editMode === "image_only";
  const selectedCompetingOutline = editable && getCompetingPhotoOutline(item.payload);
  const competingPhotoOutlines = deferredDraft || editingMotif || selectedCountGroup ? [] : editable
    ? selectedCompetingOutline ? [selectedCompetingOutline] : [] : getCompetingPhotoReviewOutlines(detections);
  const contrastCrossingTraces = deferredDraft || editingMotif || selectedCountGroup ? [] : editable
    ? [
      ...(contrastOutlineReview ? getContrastCrossingTraces(item.payload, initialOuter(), quad) : []),
      ...(!draftTouched ? getCompactSceneCrossingTraces(item.payload) : []),
    ] : detections.flatMap(getCompactSceneCrossingTraces);
  const captureBackgroundTraces = deferredDraft || editingMotif || selectedCountGroup ? [] : editable
    ? getCaptureBackgroundReviewTraces(item.payload) : detections.flatMap(getCaptureBackgroundReviewTraces);
  const selectedMergedPair = !reviewedNonPrint && getMergedPrintPairs([item.payload || {}])[0];
  const sourceStrip = editable && !deferredDraft && !editingMotif ? getSourceStripReview(fullPrintMode) : null;
  const albumMark = editable && !editingMotif && !reviewedNonPrint ? markOffer : null;
  const albumMarkReview = useAlbumPageMarkReview(albumMark, item.payload?.photo_id);
  const displayedQuad = editingMotif ? motifQuad || savedQuad : quad;
  const paperAlignedMotif = editingMotif && motifUsesPaperAxes(savedQuad, displayedQuad);
  const paperPreviewUrl = item.payload?.variant_urls?.full_print_preview;
  const ribbedPreview = fullPrintMode?.outline_preview?.source === measuredRibbedPrintOutlineSource;
  const ribbedReviewOutline = ribbedPreview && editable && !deferredDraft && !editingMotif && item.payload?.untrimmed_url
    && motifMode?.status === "review_needed"
    ? getRibbedPrintReviewOutline(fullPrintMode, paperPreviewUrl, initialOuter(), quad) : null;
  const fullPrintPreview = ribbedPreview && !ribbedReviewOutline ? null : paperPreviewUrl;
  const comparePaperEdge = !editingMotif && showsPaperEdgeComparison(fullPrintMode)
    && item.payload?.untrimmed_url && fullPrintPreview;
  const [usingOutlineSuggestion, setUsingOutlineSuggestion] = useState(Boolean(initialPaperOutline));
  const [editingPreview, setEditingPreview] = useState(Boolean(initialPaperOutline && !outlinePreviewHint));
  const [measurementFields, setMeasurementFields] = useState(() => printSizeInputs(storedMeasurement));
  const [orientation, setOrientation] = useState(storedOrientation);
  const [savedOrientation, setSavedOrientation] = useState(storedOrientation);
  const [savedCropUrl, setSavedCropUrl] = useState(""); const [saveError, setSaveError] = useState("");
  const visibleSavedCropUrl = (editingMotif ? item.payload?.variant_urls?.image_only === null : fullPrintHeld)
    ? "" : savedCropUrl;
  const savedAspectPreview = editable && !editingMotif && !deferredDraft && !reviewedNonPrint && !saving
    ? savedFullPrintReviewUrl(item.payload, initialOuter(), quad, measurementFields, orientation) : null;
  const [blockedCrop, setBlockedCrop] = useState(null);
  const [magnifierEnabled, setMagnifierEnabled] = useState(true);
  const [magnifier, setMagnifier] = useState({ visible: false, x: 0, y: 0 });
  useEffect(() => {
    setQuad(initialOuter());
    setEditMode("full_print");
    setMotifQuad(storedMotif);
    setUsingOutlineSuggestion(Boolean(initialPaperOutline));
    setEditingPreview(Boolean(initialPaperOutline && !outlinePreviewHint));
    setMeasurementFields(printSizeInputs(storedMeasurement));
    setOrientation(storedOrientation);
    setSavedOrientation(storedOrientation);
    setSavedCropUrl("");
    setSaveError("");
    setBlockedCrop(null);
    setDraftTouched(false);
    setConfirmMergedPrintAsOne(false);
    setConfirmedContrastOutline(null);
  }, [item.id]);
  const box = quadToBox(quad);
  const validQuad = !editable || isValidQuad(displayedQuad);
  const motifOutside = editingMotif && !motifFitsPaper(displayedQuad, savedQuad);
  const validMotif = editingMotif && motifQuad && validQuad && !motifOutside;
  const crossingNeedsCorrection = crossingOutline && initialOuter()?.every(
    (point, index) => point.x === quad?.[index]?.x && point.y === quad?.[index]?.y,
  );
  const { size, error: sizeError } = parsePrintSize(measurementFields);
  const refreshContactEdges = contactEdgesNeedRefresh(item.payload, fullPrintMode);
  const canReprocess = onReprocess && !draftTouched && !editingMotif && !usingOutlineSuggestion
    && !measurementPage && !measuredTraces?.length && fullPrintHeld && !markOffer
    && fullPrintMode?.status === "review_needed"
    && (refreshContactEdges || (fullPrintMode.reason_codes?.includes("POSSIBLE_CONTENT_CUTOFF")
      && fullPrintMode.reason_codes.includes("PRINT_ASPECT_UNCERTAIN")));
  function moveCorner(event, cornerIndex) {
    if (!editable || !displayedQuad || saving) return;
    const pointerId = event.pointerId;
    event.preventDefault(); event.currentTarget.setPointerCapture(pointerId);
    const svg = event.currentTarget.ownerSVGElement; const rect = svg.getBoundingClientRect();
    const startX = event.clientX, startY = event.clientY;
    const initial = displayedQuad[cornerIndex];
    let moved = false;
    const move = e => {
      if (e.pointerId !== pointerId) return;
      e.preventDefault();
      const dx = e.clientX - startX, dy = e.clientY - startY;
      if (editingMotif && !moved && Math.hypot(dx, dy) < 2) return;
      moved = true;
      const x = Math.max(0, Math.min(1, editingMotif ? initial.x + dx / rect.width : (e.clientX - rect.left) / rect.width));
      const y = Math.max(0, Math.min(1, editingMotif ? initial.y + dy / rect.height : (e.clientY - rect.top) / rect.height));
      const resizedMotif = paperAlignedMotif && resizeMotifInPaperPlane(savedQuad, displayedQuad, cornerIndex, { x, y });
      if (paperAlignedMotif && !resizedMotif) return;
      const magnifiedCorner = resizedMotif?.[cornerIndex] || { x, y };
      setMagnifier({ visible: magnifierEnabled, ...magnifiedCorner });
      setEditingPreview(true);
      setDraftTouched(true);
      setSavedCropUrl(""); setSaveError("");
      if (editingMotif) {
        setMotifQuad(current => {
          const next = resizedMotif || (current || savedQuad).map((point, index) => index === cornerIndex ? { x, y } : point);
          return !storedMotif && next.every((point, index) =>
            Math.hypot(point.x - savedQuad[index].x, point.y - savedQuad[index].y) < 1e-8) ? null : next;
        });
        return;
      }
      setQuad(current => current.map((point,index) => index === cornerIndex ? { x, y } : point));
    };
    setMagnifier({ visible: magnifierEnabled, x: initial.x, y: initial.y });
    const up = e => {
      if (e.pointerId !== pointerId) return;
      setMagnifier(current => ({ ...current, visible: false }));
      removeEventListener("pointermove", move); removeEventListener("pointerup", up); removeEventListener("pointercancel", up);
    };
    addEventListener("pointermove", move); addEventListener("pointerup", up); addEventListener("pointercancel", up);
  }
  async function save() {
    if (contrastOutlineNeedsCheck) return;
    if (editingMotif ? !validMotif : (!validQuad || sizeError || crossingNeedsCorrection)) return;
    if (deferredDraft && !confirmDeferredDraft) return;
    setSaving(true); setSaveError("");
    try {
      if (deferredDraft) {
        await addManualPhoto(item.page_id, deferredDraft.requestId, quad, size, null, deferredDraft, {
          replaceMountedFragments: sourceFragmentDraft, confirmDarkBacking: darkBackingDraft,
          confirmAlbumPocketPrint: Boolean(embossedAlbumAction(deferredDraft.reason)),
          confirmStandalonePrint: requiresStandalonePrintFlag(deferredDraft.reason),
        });
        onDeferredDraftClose(true);
        return;
      }
      const mask = editingMotif ? {
        polygon: item.payload.segmentation_mask?.polygon || savedQuad,
        visible_polygon: visiblePolygon || savedQuad, warp_quad: savedQuad,
        image_quad: motifQuad, ...(storedMeasurement ? { print_size_mm: storedMeasurement } : {}),
        source: "manual_mobile_motif",
      } : {
        polygon: quad,
        visible_polygon: usingOutlineSuggestion ? quad : (visiblePolygon || quad),
        warp_quad: quad,
        print_size_mm: size,
        ...(orientation !== savedOrientation ? { orientation_degrees: orientation } : {}),
        source: "manual_mobile_quad",
      };
      const mark = albumMarkReview.confirmation;
      const confirmOuter = contrastOutlineReview ? contrastOutlineConfirmed
        : Boolean(printSuggestion) || paperOutlineNeedsConfirmation(fullPrintMode);
      const saved = await patchBoundingBox(item.payload.photo_id, editingMotif ? sourceBox : box, mask,
        !editingMotif && confirmOuter, mark?.id, mark?.indices,
        Boolean(selectedMergedPair && confirmMergedPrintAsOne && !editingMotif));
      albumMarkReview.reset();
      setBlockedCrop(null);
      setConfirmMergedPrintAsOne(false);
      setConfirmedContrastOutline(null);
      const savedDegrees = saved.segmentation_mask?.orientation_degrees ?? null;
      setOrientation(savedDegrees);
      setSavedOrientation(savedDegrees);
      if (!editingMotif && saved.saved_full_print_review_url) {
        setQuad(initialQuad(saved.segmentation_mask.warp_quad, saved.segmentation_mask.visible_polygon, saved.bounding_box));
      }
      const cropModes = getCropModes(saved);
      setMotifQuad(cyclicReviewQuad(sanitizeReviewPolygon(getStoredMotifQuad(saved, cropModes?.image_only))));
      const held = wholePrintApprovalHeld(cropModes, saved.crop_download_urls)
        || captureRequiresReview(cropModes?.full_print) || paperOutlineRequiresReview(cropModes?.full_print)
        || heldDeferredParentCropHints(cropModes?.full_print, saved.enhancement_applied?.review_reasons).length > 0;
      const cropUrl = editingMotif ? saved.crop_download_urls?.image_only
        : !held && (saved.crop_download_urls?.full_print
          || `/api/v1/photos/${item.payload.photo_id}/image?variant=original&v=${Date.now()}`);
      setSavedCropUrl(cropUrl || "");
      onSaved?.(saved);
    } catch (error) {
      if (error.message === "album_page_mark_review_stale") albumMarkReview.reset();
      setSaveError(error.message === "crossing_print_outline_required" ? crossingPrintSaveHint
        : error.message === "print_outline_review_required" ? printOutlineSaveHint
        : error.message === "contrast_print_outline_review_required" ? contrastOutlineReviewHint
        : darkBackingReviewErrors[error.message] || embossedPocketReviewErrors[error.message] || standalonePrintReviewErrors[error.message]
          || sourceFragmentReviewErrors[error.message] || cropProtectionErrors[error.message]
          || error.message || "Ausschnitt konnte nicht gespeichert werden.");
      if (error.message === "protected_region_removed") {
        setBlockedCrop(error.detail);
        onProtectionBlocked?.(item.id);
      }
    }
    finally { setSaving(false); }
  }
  async function reprocess() {
    if (!canReprocess || saving || reprocessing) return;
    setReprocessing(true);
    setSaveError("");
    try {
      await onReprocess(item.payload.photo_id);
    } catch (error) {
      setSaveError(error.message || "Abzug konnte nicht neu berechnet werden.");
    } finally {
      setReprocessing(false);
    }
  }
  const shown = box ? [{ bounding_box: box, photo_id: item.payload?.photo_id }] : detections;
  const count = item.payload?.detection_count;
  const uncertain = nestedOutlines.length + deferredProposals.length;
  const uncertainCount = uncertain ? ` · ${uncertain} unklar` : "";
  const overlayLabel = count == null ? "1 Crop" : `${count} ${count === 1 ? "Print" : "Prints"}${uncertainCount}`;
  const lensPosition = value => ((0.5 - 3.6 * value) / (1 - 3.6)) * 100;
  const lensX=Math.max(20,Math.min(80,magnifier.x*100+(magnifier.x>0.5?-24:24)));
  const lensY=Math.max(20,Math.min(80,magnifier.y*100+(magnifier.y<0.3?25:-25)));
  const uncertainLine = usingOutlineSuggestion && fragmentOutline && (
    <polyline points={fragmentOutline.weakEdge.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")}
      fill="none" stroke="#a21caf" strokeWidth="6" strokeDasharray="16 9" />
  );
  const alternateOutline = (!usingOutlineSuggestion && (crossingOutline || printSuggestion)) || adjacentOutline || scallopedAlternate;
  const alternateOverlay = alternateOutline && (
    <polygon points={alternateOutline.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")}
      fill="none" stroke="#a21caf" strokeWidth="6" strokeDasharray="16 9" />
  );
  const nestedOverlay = nestedOutlines.map((outline, index) => (
    <polygon key={index} points={outline.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")}
      fill="none" stroke="#a21caf" strokeWidth="6" strokeDasharray="16 9" pointerEvents="none" />
  ));
  const deferredOverlay = sourceFragmentDraft ? getSourceEdgeTraces(deferredDraft.fragment_polygons).map((polygon, index) => (
    <polygon key={index} aria-label="Teilvorschlag für den gemeinsamen Abzug"
      points={polygon.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")}
      fill="none" stroke="#a21caf" strokeWidth="5" strokeDasharray="10 6" pointerEvents="none" />
  )) : !deferredDraft && deferredProposals.map(({ id, visible_polygon: polygon, uncertain_edge: uncertainEdge, reason }, index) => (
    <g key={id} aria-label={reason === darkBackingProposal ? `Dunkle Fläche ${index + 1}`
      : isNestedFallbackProposal(reason) ? `Innenkontur ${index + 1}`
        : [frameBackingProposal, borrowedFallbackProposal].includes(reason) ? `Vorschlag ${index + 1} am Aufnahmerand` : "Möglicher Abzug"}
      stroke="#7e22ce" strokeWidth="6" pointerEvents="none">
      {polygon.map(({ x, y }, side) => {
        const next = polygon[(side + 1) % polygon.length];
        return <line key={side} x1={x * 1000} y1={y * 1000} x2={next.x * 1000} y2={next.y * 1000}
          strokeDasharray={side === uncertainEdge ? "4 10" : "18 10"} />;
      })}
      {standalonePrintNeedsConfirmation(reason) && deferredProposals.length > 1 && (
        <text className="dark-backing-proposal-label" x={Math.min(polygon[0].x * 1000 + 12, 945)}
          y={Math.max(polygon[0].y * 1000 - 12, 35)}>{`?${index + 1}`}</text>
      )}
    </g>
  ));
  const deferredEvidenceOffers = deferredDraft ? [deferredDraft] : deferredProposals;
  const deferredEvidenceOverlay = deferredEvidenceOffers.flatMap((proposal) => getDeferredPrintEvidence(proposal).flatMap(
    ({ key, label, color, dash, traces }) => traces.map(([first, last], index) => (
      <line key={`${proposal.id}-${key}-${index}`} aria-label={label}
        x1={first.x * 1000} y1={first.y * 1000} x2={last.x * 1000} y2={last.y * 1000}
        stroke={color} strokeWidth="8" strokeDasharray={dash.map((part) => part * 2).join(" ")} pointerEvents="none" />
    )),
  ));
  const fragmentCountOverlay = countOutlines.map(({ key, supportedSides, weakEdge }) => (
    <g key={key} aria-label="Möglicher gemeinsamer Abzug" fill="none" stroke="#2563eb" strokeWidth="6" pointerEvents="none">
      <polyline points={supportedSides.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} strokeDasharray="18 10" />
      <polyline points={weakEdge.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} strokeDasharray="4 10" />
    </g>
  ));
  const captureSplitOverlay = captureSplitGroups.map(({ key, traces }) => (
    <g key={key} aria-label="Möglicher gemeinsamer Papierrand" fill="none" stroke="#2563eb"
      strokeWidth="6" strokeDasharray="18 10" pointerEvents="none">
      {traces.map((line, index) => <polyline key={index} points={line.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} />)}
    </g>
  ));
  const mergedPairOverlay = mergedPairs.map(({ id, outlines, separator }) => (
    <g key={id} aria-label="Zwei mögliche Umrisse am selben Vorschlag" fill="none" stroke="#2563eb"
      strokeWidth="5" pointerEvents="none">
      {outlines.map((outline, index) => <polygon key={index} strokeDasharray="16 10"
        points={outline.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} />)}
      <polyline strokeDasharray="4 10" points={separator.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} />
    </g>
  ));
  const filmGridOverlay = (filmGrid ? [{ key: "selected-grid", line: filmGrid }] : filmGridGroups).map(({ key, line, rim }) => (
    <g key={key} fill="none" stroke="#0e7490" strokeWidth="7" pointerEvents="none">
      <polyline aria-label="Zwischenraum im Raster" strokeDasharray="14 8"
        points={line.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} />
      {rim && <polyline aria-label="Möglicher durchsichtiger Saum" strokeDasharray="3 10"
        points={rim.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} />}
    </g>
  ));
  const countGroupOverlay = selectedCountGroup && (
    <g aria-label={selectedCountGroup.kind === "film" ? "Vorschläge zur Prüfung der Negativhülle"
      : selectedCountGroup.kind === "album_leaf" ? "Mögliche Albumseite zur Anzahlprüfung"
      : selectedCountGroup.kind === "printed_illustration" ? "Mögliche gemeinsame Abbildung zur Anzahlprüfung"
      : selectedCountGroup.kind === "crossing" ? "Schräger Vorschlag zur Anzahlprüfung" : "Vorschläge zur Prüfung der Papierkante"}
      fill="#0e7490" fillOpacity="0.11"
      stroke="#0e7490" strokeWidth="6" pointerEvents="none">
      {detections.filter(({ photo_id }) => selectedCountGroup.photoIds.has(photo_id)).map(detection => {
        const polygon = sanitizeReviewPolygon(detection.visible_polygon || detection.polygon);
        return polygon && <polygon key={detection.photo_id}
          points={polygon.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} />;
      })}
      {selectedCountGroup.kind === "crossing" && selectedCountGroup.traces.map((line, index) => {
        const { color, dash, label } = crossingFallbackTraceStyles[index];
        return <polyline key={label} aria-label={label} fill="none" stroke={color} strokeWidth="8"
          strokeDasharray={dash.map(value => value * 2).join(" ")}
          points={line.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} />;
      })}
      {selectedCountGroup.kind === "album_leaf" && [
        ...selectedCountGroup.traces.map(points => ({ points, ...albumLeafTraceStyle })),
        { points: selectedCountGroup.child, closed: true, ...albumLeafChildStyle },
      ].map(({ points, closed, color, contrast, dash, label }, index) => {
        const Shape = closed ? "polygon" : "polyline";
        const positions = points.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ");
        return <g key={index} fill="none" aria-label={label}>
          <Shape points={positions} stroke={contrast} strokeWidth="9" />
          <Shape points={positions} stroke={color} strokeWidth="5" strokeDasharray={dash.map(value => value * 2).join(" ")} />
        </g>;
      })}
      {selectedCountGroup.kind === "printed_illustration" && <g fill="none" aria-label={printedIllustrationFrameStyle.label}>
        <polygon points={selectedCountGroup.frame.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")}
          stroke={printedIllustrationFrameStyle.contrast} strokeWidth="9" />
        <polygon points={selectedCountGroup.frame.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")}
          stroke={printedIllustrationFrameStyle.color} strokeWidth="5" strokeDasharray="16 10" />
      </g>}
    </g>
  );
  const paperJoinOverlay = paperJoinGroups.map(({ key, traces }) => (
    <g key={key} aria-label="Gemessene Randabschnitte an der Teilung" fill="none" stroke="#0e7490"
      strokeWidth="7" strokeDasharray="14 8" pointerEvents="none">
      {traces.map((line, index) => <polyline key={index} points={line.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} />)}
    </g>
  ));
  const shownPaperGrids = paperGrid?.cut
    ? [{ key: item.id, outline: paperGrid.outline, cuts: new Map([[item.id, paperGrid.cut]]) }]
    : selectedCountGroup?.kind === "paper-grid" ? [selectedCountGroup] : paperGridGroups;
  const paperGridOverlay = shownPaperGrids.map(({ key, outline, cuts }) => (
    <g key={key} fill="none" stroke="#1d4ed8" strokeWidth="6" pointerEvents="none">
      <polygon aria-label="Möglicher Aussenumriss" strokeDasharray="18 10"
        points={outline.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} />
      {[...cuts].map(([id, cut]) => (
        <polyline key={id} aria-label="Kante eines Teilvorschlags" strokeDasharray="4 10"
          points={cut.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} />
      ))}
    </g>
  ));
  const overlayHints = [countOutlines.length && "mobile-fragment-count-review-hint",
    mergedPairs.length && "mobile-merged-print-pair-hint",
    competingPhotoOutlines.length && "mobile-competing-photo-outline-hint",
    captureSplitGroups.length && "mobile-capture-split-count-hint",
    filmGridGroups.length && "mobile-film-grid-count-hint", paperGridGroups.length && "mobile-paper-grid-count-hint",
    paperJoinGroups.length && "mobile-paper-join-count-hint",
    editable && motifTraces.length && "mobile-motif-seam-hint",
    editable && preliminaryMotifMarks && "mobile-motif-review-hint",
    contrastCrossingTraces.length && "mobile-contrast-crossing-review-hint",
    captureBackgroundTraces.length && "mobile-capture-background-review-hint",
    ribbedReviewOutline && "mobile-ribbed-print-outline-hint",
    sourceStrip && "mobile-source-strip-review-hint",
  ].filter(Boolean).join(" ");
  const paperTraceOverlay = printEdgeTraces.map((trace, index) => (
    <polyline key={index} points={trace.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")}
      fill="none" stroke={index === filmEdgeIndex ? filmNeighborEdgeColor : "#a21caf"}
      strokeWidth="5" strokeDasharray="12 8" pointerEvents="none" />
  ));
  const ribbedOutlinePoints = ribbedReviewOutline?.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ");
  const [neighborOutline] = getSourceEdgeTraces([fullPrintMode?.neighbor_overlap_outline]);
  const missingPaperOutlines = getSourceEdgeTraces(fullPrintMode?.neighbor_content_loss_outlines);
  return <>
    {canEditMotif && <div className="task-switch" role="group" aria-label="Ausschnitt bearbeiten">
      {Object.entries(cropModeLabels).map(([mode, label]) => (
        <button key={mode} type="button" className={editMode === mode ? "active" : ""} aria-pressed={editMode === mode}
          disabled={saving} onClick={() => {
            if (editMode === mode) return;
            setEditMode(mode);
            setEditingPreview(mode === "image_only" && Boolean(motifQuad));
            setSavedCropUrl("");
            setSaveError("");
          }}>{label}</button>
      ))}
    </div>}
    <div className={`review-image overlay-image ${validQuad && !motifOutside ? "" : "invalid-quad"}`}
      onContextMenu={event=>event.preventDefault()}>
    <div className="review-image-frame">
      <img src={item.preview_url} alt="Zu prüfende Albumseite" draggable="false"
        aria-describedby={overlayHints || undefined} />
      <svg viewBox="0 0 1000 1000" preserveAspectRatio="none">
        {competingPhotoOutlines.length > 0 && <CompetingPhotoOutlineOverlay outlines={competingPhotoOutlines} />}
        {editable && displayedQuad ? <g>
          {visiblePolygon && <polygon points={visiblePolygon.map(point => `${point.x * 1000},${point.y * 1000}`).join(" ")}
            fill="none" stroke="#f59e0b" strokeWidth="5" strokeDasharray="14 9" />}
          <polygon className="editable-quad" points={displayedQuad.map(point => `${point.x * 1000},${point.y * 1000}`).join(" ")} />
          {ribbedOutlinePoints && <g aria-hidden="true" fill="none" pointerEvents="none">
            <polygon points={ribbedOutlinePoints} stroke={ribbedPrintOutlineStyle.contrast} strokeWidth="4" vectorEffect="non-scaling-stroke" />
            <polygon points={ribbedOutlinePoints} stroke={ribbedPrintOutlineStyle.color} strokeWidth="2" vectorEffect="non-scaling-stroke"
              strokeDasharray={ribbedPrintOutlineStyle.dash.join(" ")} />
          </g>}
          {!editingMotif && <>{uncertainLine}{alternateOverlay}{paperTraceOverlay}</>}
          {sourceStrip && <SourceStripOutline edgeOnly={sourceStrip.edgeOnly}
            points={sourceStrip.outline.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} />}
          {!editingMotif && getSourceEdgeTraces(fullPrintMode?.exposed_overlap_edge_traces).map((trace, index) => (
            <polyline key={index} aria-label="Möglicher Papierrand ausserhalb des Zuschnitts"
              points={trace.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")}
              fill="none" stroke="#0e7490" strokeWidth="5" strokeDasharray="12 8" pointerEvents="none" />
          ))}
          {!editingMotif && getSourceEdgeTraces(fullPrintMode?.partial_print_extent_traces).map((trace, index) => (
            <polyline key={index} aria-label="Gemessener Übergang ausserhalb der bisherigen Kontur"
              points={trace.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")}
              fill="none" stroke="#0891b2" strokeWidth="1.5" strokeDasharray="8 5" pointerEvents="none" />
          ))}
          {!editingMotif && neighborOutline && <polyline aria-label="Überlappender Nachbarabzug"
            points={neighborOutline.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")}
            fill="none" stroke="#a21caf" strokeWidth="5" strokeDasharray="12 8" pointerEvents="none" />}
          {!editingMotif && missingPaperOutlines.map((outline, index) => (
            <polyline key={index} aria-label="Möglicherweise fehlende Fläche"
              points={outline.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")}
              fill="none" stroke="#a21caf" strokeWidth="5" strokeDasharray="12 8" pointerEvents="none" />
          ))}
          {motifTraces.length > 0 && <g aria-label="Gemessene Abschnitte einer möglichen Motivkante" fill="none"
            stroke="#2563eb" strokeWidth="5" strokeDasharray="12 8" pointerEvents="none">
            {motifTraces.map((trace, index) => <polyline key={index}
              points={trace.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} />)}
          </g>}
          {displayedQuad.map((point, index) => <g key={index}>
            <circle className="drag-handle" cx={point.x * 1000} cy={point.y * 1000} r="25" onPointerDown={event => moveCorner(event, index)} />
            <text className="corner-label" x={point.x * 1000 + 28} y={point.y * 1000 - 22}>{index + 1}</text>
          </g>)}
        </g> : shown.map(({ bounding_box: box, photo_id: photoId }, index) => <g key={photoId || index}>
          {!selectedCountGroup?.photoIds.has(photoId)
            && (selectedCountGroup?.kind !== "album_leaf" || selectedCountGroup.childId !== photoId) && (
            <rect x={box.x1 * 1000} y={box.y1 * 1000} width={(box.x2 - box.x1) * 1000} height={(box.y2 - box.y1) * 1000} />
          )}
          <text x={box.x1 * 1000 + 12} y={box.y1 * 1000 + 38}>{index + 1}</text>
        </g>)}
        {nestedOverlay}{deferredOverlay}{deferredEvidenceOverlay}{fragmentCountOverlay}{captureSplitOverlay}{mergedPairOverlay}{countGroupOverlay}
        {filmGridOverlay}{paperGridOverlay}{paperJoinOverlay}
        {albumMarkReview.bounds.map(({ x1, y1, x2, y2 }, index) => <rect key={index} x={x1 * 1000} y={y1 * 1000}
          width={(x2 - x1) * 1000} height={(y2 - y1) * 1000}
          aria-label={albumMark.source === offPrintAdvisorySource ? `Zu prüfende Schutzmarkierung ${index + 1}` : "Zu prüfende Schutzmarkierung"}
          style={{ fill: "none", stroke: "#facc15", strokeWidth: 6, pointerEvents: "none" }} />)}
        {blockedCrop?.mode === editMode && blockedCrop.regions.map(({ x1, y1, x2, y2 }, index) => (
          <rect key={index} aria-label="Geschützter Bildbereich" x={x1 * 1000} y={y1 * 1000}
            width={(x2 - x1) * 1000} height={(y2 - y1) * 1000}
            style={{ fill: "none", stroke: "#b91c1c", strokeWidth: 7, pointerEvents: "none" }} />
        ))}
        {editable && motifMarginMarks.map((region, index) => (
          <polygon key={index} aria-label="Mögliche Beschriftung am Papierrand" fill="none" stroke="#b45309" strokeWidth="5"
            pointerEvents="none" points={region.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")} />
        ))}
        {Boolean(visibleTraces?.length) && <MeasuredEdgeOverlay traces={visibleTraces} />}
        {[[contrastCrossingTraces, contrastCrossingTraceStyle], [captureBackgroundTraces, captureBackgroundTraceStyle]].map(
          ([traces, { color, contrast, dash }]) => traces.length > 0 && <g key={color} aria-hidden="true" fill="none" pointerEvents="none">
          {traces.map(([start, end], index) => {
            const points = `${start.x * 1000},${start.y * 1000} ${end.x * 1000},${end.y * 1000}`;
            return <g key={index}>
              <polyline points={points} stroke={contrast} strokeWidth="4" vectorEffect="non-scaling-stroke" />
              <polyline points={points} stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" strokeDasharray={dash.join(" ")} />
            </g>;
          })}
        </g>)}
        {competingPhotoOutlines.some(outline => outline.motifGuide) && <SharedMotifGuideOverlay outlines={competingPhotoOutlines} />}
      </svg>
      {magnifier.visible && <div className="crop-magnifier" style={{
        left: `${lensX}%`, top: `${lensY}%`, backgroundImage: `url(${item.preview_url})`, backgroundPosition: `${lensPosition(magnifier.x)}% ${lensPosition(magnifier.y)}%`,
      }}><i /></div>}
    </div>
    <span>{!validQuad ? "Ecken kreuzen sich" : motifOutside ? "Alle Motivecken müssen auf dem Abzug liegen" : overlayLabel}</span>
  </div>
    {item.page_id && (
      <div className="edge-measurement-controls">
        <button className="magnifier-toggle" type="button" onClick={() => setMeasurementPage(item.page_id)}>
          Kantenmessung öffnen
        </button>
        {missingTraces.length > 0 && (
          <button className="magnifier-toggle" type="button" aria-pressed={showMissingEdges}
            onClick={() => setShowMissingEdges(!showMissingEdges)}>Möglichen weiteren Abzug prüfen</button>
        )}
        {Boolean(measuredTraces?.length) && <button className="magnifier-toggle" type="button"
          onClick={() => setMeasuredEdge(null)}>Messspur ausblenden</button>}
      </div>
    )}
    {showMissingEdges && missingTraces.length > 0 && (
      <p className="quad-hint">{nearCaptureTraces.length ? nearCapturePrintEdgeHint : missingPrintEdgeHint}
        {!deferredProposals.length && " Einen eigenen Abzug kannst du im Desktop-Editor ergänzen."}
      </p>
    )}
    {countOutlines.length > 0 && <p className="quad-hint" id="mobile-fragment-count-review-hint">{fragmentCountHint}</p>}
    {mergedPairs.length > 0 && <p className="quad-hint" id="mobile-merged-print-pair-hint">{mergedPrintPairHint}</p>}
    {contrastCrossingTraces.length > 0 && <p className="quad-hint" id="mobile-contrast-crossing-review-hint">{contrastCrossingTraceHint}</p>}
    {captureBackgroundTraces.length > 0 && <p className="quad-hint" id="mobile-capture-background-review-hint">
      {captureBackgroundReviewHint}{editable && ` ${captureBackgroundWrongObjectHint}`}
    </p>}
    {competingPhotoOutlines.length > 0 && <CompetingPhotoOutlineLegend id="mobile-competing-photo-outline-hint" mobile
      showMotifGuide={competingPhotoOutlines.some(outline => outline.motifGuide)} />}
    {captureSplitGroups.length > 0 && <p className="quad-hint" id="mobile-capture-split-count-hint">{captureSplitReviewHint}</p>}
    {!deferredDraft && <CaptureSplitSourceReview groups={captureSplitGroups} photos={detections} sourceUrl={item.preview_url} />}
    {filmGridGroups.length > 0 && <p className="quad-hint" id="mobile-film-grid-count-hint">
      {filmGridCountHint}{filmGridGroups.some(({ rim }) => rim) && ` ${filmGridRimHint}`}
    </p>}
    {paperGridGroups.length > 0 && <p className="quad-hint" id="mobile-paper-grid-count-hint">{paperGridCountHint}</p>}
    {paperJoinGroups.length > 0 && <p className="quad-hint" id="mobile-paper-join-count-hint">{paperJoinCountHint}</p>}
    {editable && motifTraces.length > 0 && <p className="quad-hint" id="mobile-motif-seam-hint">
      Die blau gestrichelten Linien zeigen gemessene Abschnitte einer möglichen Motivkante. {openMotifGuides
        ? motifBoundaryReviewHint : "Prüfe den Inhalt auch ausserhalb der Linien."}
    </p>}
    {nestedOutlines.length > 0 && <p className="quad-hint">{nestedPrintHint} Einen eigenen Abzug kannst du im Desktop-Editor ergänzen.</p>}
    {!darkBackingDraft && deferredProposals.some(({ reason }) => reason === sourceFragmentProposal)
      && <p className="quad-hint">{sourceFragmentCountHint}</p>}
    {!deferredDraft && deferredProposals.some(({ reason }) => reason !== sourceFragmentProposal && !standalonePrintNeedsConfirmation(reason))
      && <p className="quad-hint">{deferredPrintHint} Einen eigenen Abzug kannst du im Desktop-Editor ergänzen.</p>}
    <DarkBackingEvidenceLegend proposals={deferredEvidenceOffers} mobile />
    <DeferredPrintReviewHints proposals={deferredEvidenceOffers} draft={Boolean(deferredDraft)} mobile />
    {measurementPage && measurementPage === item.page_id && <EdgeMeasurement key={item.page_id} pageId={item.page_id}
      onClose={() => setMeasurementPage(null)} onMeasured={(traces) => setMeasuredEdge({ pageId: item.page_id, traces })} />}
    {editable && <>
      {item.payload?.confidence != null && item.payload?.review_reasons?.includes(clippedOverlapReason) && (
        <p className="quad-hint">{clippedOverlapHint}</p>
      )}
      {item.payload?.confidence != null && item.payload?.review_reasons?.includes(scannerBackingReason) && (
        <p className="quad-hint">{scannerBackingHint}</p>
      )}
      {filmGrid && <p className="quad-hint">{filmGridHint}</p>}
      {paperJoinReview && <p className="quad-hint">{paperJoinHint}</p>}
      {paperGrid && (paperGrid.cut || !usingOutlineSuggestion)
        && <p className="quad-hint">{paperGrid.cut ? paperGridFragmentHint : paperGridOuterHint}</p>}
      {fragmentReview && <p className="quad-hint">{fragmentOutlineHint}</p>}
      {crossingOutline && <p className="quad-hint">{crossingPrintHint}</p>}
      {ribbedReviewOutline && <p className="quad-hint" id="mobile-ribbed-print-outline-hint">{ribbedPrintOutlineHint}</p>}
      {contrastOutlineReview && <p className="quad-hint">{contrastOutlineReviewHint}</p>}
      {!reviewedNonPrint && getDeferredParentCropHints(item.payload?.review_reasons).map(({ id, text }) => (
        <p key={id} className="quad-hint" id={`mobile-${id}`}>{text}</p>
      ))}
      {printEdgeReviewHint && <p className="quad-hint" id="mobile-print-edge-review-hint">{printEdgeReviewHint}</p>}
      {sourceStrip && <SourceStripReview strip={sourceStrip} sourceUrl={item.preview_url} hintId="mobile-source-strip-review-hint" />}
      {!editingMotif && <NeighborContentReview fullPrint={fullPrintMode} sourceUrl={item.preview_url} />}
      {albumMarkReview.bounds.length > 0 && <AlbumPageMarkReview mark={albumMark} review={albumMarkReview}
        sourceUrl={item.preview_url} disabled={saving} />}
      {canReprocess && <button className="magnifier-toggle" type="button" disabled={saving || reprocessing} onClick={reprocess}>
        <RefreshCw size={18} />{reprocessing ? "Abzug wird neu berechnet …"
          : refreshContactEdges ? "Papierkanten erneut prüfen" : "Abzug neu berechnen"}
      </button>}
      {motifReviewPending && <p className="quad-hint" id="mobile-motif-review-hint">
        {preliminaryMotifMarks ? motifBoundaryReviewHint : <>
          {motifBoundaryReviewHint && `${motifBoundaryReviewHint} `}{canEditMotif ? motifReviewHint : motifDesktopReviewHint}
        </>}
      </p>}
      {stockMotifPreview && !editingMotif && <details className="quad-hint">
        <summary>Motivvorschau ansehen</summary>
        <p>{motifBoundaryReviewHint}</p>
        <img className="review-crop-preview" src={stockMotifPreview} alt="Motivvorschau zur Prüfung" />
      </details>}
      {adjacentOutline && <p className="quad-hint">{adjacentPrintHint} Einen eigenen Abzug kannst du im Desktop-Editor ergänzen.</p>}
      {scallopedAlternate && <p className="quad-hint">Prüfe den gestrichelten Vorschlag am Original. Wähle «Prints zusammen», falls dort ein eigener Abzug aufliegt.</p>}
      {!editingMotif && (fragmentOutline || crossingOutline || printSuggestion) && (
        <button className="magnifier-toggle" type="button" disabled={saving} onClick={() => {
          const suggestion = crossingOutline || printSuggestion || cyclicReviewQuad(fragmentOutline?.quad);
          if (!suggestion) return;
          setQuad(suggestion);
          setUsingOutlineSuggestion(true);
          setDraftTouched(true);
          if (suggestion !== printSuggestion || !outlinePreviewHint) setEditingPreview(true);
          setSavedCropUrl("");
          setSaveError("");
        }}>{paperGrid && usingOutlineSuggestion ? "Gemessenen Aussenumriss zurücksetzen" : "Aussenumriss in den Editor laden"}</button>
      )}
      {!editingMotif && uncertainLine && <p className="quad-hint">Prüfe besonders die gestrichelte Papierkante.</p>}
      {editingMotif ? <>
        <p className="quad-hint">Setze die Ecken an die Motivgrenze. Lass alle Bildfelder und zugehörigen Beschriftungen im Zuschnitt.
          {paperAlignedMotif && " Beim Ziehen bleiben die Motivkanten parallel zum entzerrten Papier."}
        </p>
        {(suggestedMotif || !storedMotif) && <button className="magnifier-toggle" type="button" disabled={saving} onClick={() => {
          setMotifQuad(suggestedMotif || savedQuad);
          setDraftTouched(true);
          setEditingPreview(true);
          setSavedCropUrl("");
          setSaveError("");
        }}>{suggestedMotif ? "Motivvorschlag in den Editor laden" : "Motiv reicht bis zum Papierrand"}</button>}
      </> : <div className="quad-hint">Vier nummerierte Ecken frei verschieben · danach wird das Foto geradegerichtet</div>}
      <button className={`magnifier-toggle ${magnifierEnabled ? "active" : ""}`} onClick={()=>setMagnifierEnabled(value=>!value)}><ZoomIn size={18}/>{magnifierEnabled ? "Lupe beim Ziehen aktiv" : "Lupe einschalten"}</button>
      {!editingMotif && <><PrintSizeFields
        value={measurementFields}
        onChange={(fields) => {
          setMeasurementFields(fields);
          setDraftTouched(true);
          setEditingPreview(true);
          setSavedCropUrl("");
          setSaveError("");
        }}
        disabled={saving}
        error={sizeError}
      />
      {!deferredDraft && <CropOrientationField value={orientation} disabled={saving} onChange={(value) => {
        setOrientation(value);
        setDraftTouched(true);
        setEditingPreview(true);
        setSavedCropUrl("");
        setSaveError("");
      }} />}</>}
      {deferredDraft && <label className="mounted-print-fragment-review">
        <input type="checkbox" checked={confirmDeferredDraft} disabled={saving}
          onChange={event => setConfirmDeferredDraft(event.target.checked)} />
        {standaloneDraft ? standalonePrintConfirmation : "Beide violett markierten Teilvorschläge gehören zu diesem Abzug."}
      </label>}
      {!deferredDraft && !editingMotif && selectedMergedPair && <>
        {selectedMergedPair.outlines.map((outline, index) => (
          <button key={index} className="magnifier-toggle" type="button" disabled={saving} onClick={() => {
            const measured = cyclicReviewQuad(sanitizeReviewPolygon(outline));
            if (!measured) return;
            setQuad(measured);
            setUsingOutlineSuggestion(true);
            setDraftTouched(true);
            setEditingPreview(true);
            setConfirmMergedPrintAsOne(false);
            setSavedCropUrl("");
            setSaveError("");
          }}>Umriss {index + 1} laden</button>
        ))}
        <label className="mounted-print-fragment-review">
          <input type="checkbox" checked={confirmMergedPrintAsOne} disabled={saving}
            onChange={event => {
              setConfirmMergedPrintAsOne(event.target.checked);
              setSaveError("");
            }} />
          {mergedPrintAsOneLabel}
        </label>
      </>}
      {!deferredDraft && !editingMotif && contrastOutlineReview && (
        <label className="mounted-print-fragment-review">
          <input type="checkbox" checked={contrastOutlineConfirmed} disabled={saving}
            onChange={event => setConfirmedContrastOutline(event.target.checked ? quad : null)} />
          {contrastOutlineConfirmation}
        </label>
      )}
      <button className="save-crop" onClick={save} disabled={saving
        || (deferredDraft && !confirmDeferredDraft)
        || contrastOutlineNeedsCheck
        || (editingMotif ? !validMotif : (!validQuad || Boolean(sizeError) || crossingNeedsCorrection))}>
        <Save size={18}/>{geometrySaving ? "Richtet das Bild …" : standaloneDraft ? "Als Abzug erfassen" : sourceFragmentDraft ? "Als einen Abzug speichern"
          : editingMotif ? "Motivzuschnitt speichern" : "Viereck speichern & geraderichten"}
      </button>
      {deferredDraft && <button type="button" className="magnifier-toggle" disabled={saving}
        onClick={() => onDeferredDraftClose(false)}>Zurück zur Printanzahl</button>}
      {saveError && <p className="crop-save-error" role="alert">{saveError}</p>}
      {displayedQuad && !motifOutside && (comparePaperEdge && !editingPreview && !visibleSavedCropUrl
        ? <PaperEdgePreviewComparison currentUrl={item.payload.untrimmed_url} possibleUrl={comparePaperEdge}
            hint={outlinePreviewHint} outlinePreview={fullPrintMode?.outline_preview} />
        : <LiveCropPreview src={item.preview_url} quad={displayedQuad} savedCropUrl={visibleSavedCropUrl}
            outlinePreviewUrl={editingPreview ? null : editingMotif ? item.payload?.variant_urls?.image_only_preview : fullPrintPreview}
            savedAspectPreviewUrl={savedAspectPreview}
            outlinePreviewHint={editingMotif ? motifBoundaryReviewHint : outlinePreviewHint}
            measuredAspect={!editingMotif && size ? size.width / size.height : null}
            referenceMatrix={editingMotif ? fullPrintMode.page_to_output : null}
            orientation={editingMotif ? item.payload?.orientation?.rotation_degrees : orientation} />)}
    </>}
  </>;
}

function sanitizeReviewPolygon(polygon) {
  if (!Array.isArray(polygon) || polygon.length < 4) return null;
  const points=polygon.map(point=>({x:Number(point?.x),y:Number(point?.y)}));
  return points.every(point=>Number.isFinite(point.x)&&Number.isFinite(point.y)) ? points : null;
}

function cyclicReviewQuad(points) {
  if (!Array.isArray(points) || points.length !== 4) return null;
  const center=points.reduce((sum,point)=>({x:sum.x+point.x/4,y:sum.y+point.y/4}),{x:0,y:0});
  const ordered=[...points].sort((a,b)=>Math.atan2(a.y-center.y,a.x-center.x)-Math.atan2(b.y-center.y,b.x-center.x));
  const start=ordered.reduce((best,point,index)=>point.x+point.y<ordered[best].x+ordered[best].y?index:best,0);
  const candidate=[...ordered.slice(start),...ordered.slice(0,start)];
  return isValidQuad(candidate) ? candidate : null;
}

function reviewHull(points) {
  const sorted=[...points].sort((a,b)=>a.x-b.x||a.y-b.y); const cross=(o,a,b)=>(a.x-o.x)*(b.y-o.y)-(a.y-o.y)*(b.x-o.x);
  const lower=[]; for(const point of sorted){while(lower.length>=2&&cross(lower.at(-2),lower.at(-1),point)<=0)lower.pop();lower.push(point);}
  const upper=[]; for(const point of [...sorted].reverse()){while(upper.length>=2&&cross(upper.at(-2),upper.at(-1),point)<=0)upper.pop();upper.push(point);}
  return [...lower.slice(0,-1),...upper.slice(0,-1)];
}

function minimumAreaReviewQuad(points) {
  const hull=reviewHull(points); if(hull.length===4)return cyclicReviewQuad(hull); if(hull.length<2)return null; let best=null;
  hull.forEach((point,index)=>{const next=hull[(index+1)%hull.length],angle=Math.atan2(next.y-point.y,next.x-point.x),cos=Math.cos(angle),sin=Math.sin(angle);
    const rotated=hull.map(candidate=>({x:candidate.x*cos+candidate.y*sin,y:-candidate.x*sin+candidate.y*cos}));
    const xs=rotated.map(candidate=>candidate.x),ys=rotated.map(candidate=>candidate.y),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys),area=(maxX-minX)*(maxY-minY);
    if(best&&best.area<=area)return; const inverse=candidate=>({x:candidate.x*cos-candidate.y*sin,y:candidate.x*sin+candidate.y*cos});
    best={area,points:[{x:minX,y:minY},{x:maxX,y:minY},{x:maxX,y:maxY},{x:minX,y:maxY}].map(inverse)};
  });
  return cyclicReviewQuad(best?.points||[]);
}

function initialQuad(serverQuad, visiblePolygon, box) {
  const server=sanitizeReviewPolygon(serverQuad);
  if(server?.length===4){const candidate=cyclicReviewQuad(server);if(candidate)return candidate;}
  if(visiblePolygon){const candidate=visiblePolygon.length===4?cyclicReviewQuad(visiblePolygon):minimumAreaReviewQuad(visiblePolygon);if(candidate)return candidate;}
  if(!box) return null;
  return [{x:box.x1,y:box.y1},{x:box.x2,y:box.y1},{x:box.x2,y:box.y2},{x:box.x1,y:box.y2}];
}

function quadToBox(quad) {
  if(!quad) return null; const xs=quad.map(point=>point.x), ys=quad.map(point=>point.y);
  return {x1:Math.min(...xs),y1:Math.min(...ys),x2:Math.max(...xs),y2:Math.max(...ys)};
}

function isValidQuad(quad) {
  if(!quad || quad.length!==4) return false;
  const crosses=quad.map((point,index)=>{const next=quad[(index+1)%4],after=quad[(index+2)%4];return (next.x-point.x)*(after.y-next.y)-(next.y-point.y)*(after.x-next.x);});
  const sameDirection=crosses.every(value=>value>0)||crosses.every(value=>value<0);
  const area=Math.abs(quad.reduce((sum,point,index)=>sum+point.x*quad[(index+1)%4].y-quad[(index+1)%4].x*point.y,0)/2);
  return sameDirection && area>0.001;
}

function drawQuadPreview(canvas, { image, pixels }, quad, measuredAspect, orientation, referenceMatrix) {
  const [first, second, third] = quad;
  const clockwise = (second.x - first.x) * (third.y - first.y) - (second.y - first.y) * (third.x - first.x) > 0;
  const ordered = clockwise ? quad : [quad[1], quad[0], quad[3], quad[2]];
  const source = ordered.map(({ x, y }) => ({
    x: Math.fround(Math.fround(x) * (image.naturalWidth - 1)),
    y: Math.fround(Math.fround(y) * (image.naturalHeight - 1)),
  }));
  if (measuredAspect == null) {
    // Use the export's source-pixel corner order. Measured prints retain their width edge.
    const sum = ({ x, y }) => Math.fround(x + y);
    const minimum = Math.min(...source.map(sum));
    const [firstPixel] = source.filter((point) => sum(point) - minimum <= 0.0001)
      .sort((a, b) => a.y - b.y || a.x - b.x);
    source.push(...source.splice(0, source.indexOf(firstPixel)));
  }
  const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
  const rawWidth=Math.max(distance(source[0],source[1]),distance(source[3],source[2]),1);
  const rawHeight=Math.max(distance(source[0],source[3]),distance(source[1],source[2]),1);
  const aspect = measuredAspect || (referenceMatrix && projectedCropAspect(source, referenceMatrix)) || rawWidth / rawHeight;
  const limit = 720;
  const width = Math.max(1, Math.min(limit, Math.round(limit * aspect), Math.max(240, Math.round(rawWidth))));
  const height = Math.min(limit, Math.max(1, Math.round(width / aspect)));
  const scaled = source.map(({ x, y }) => ({
    x: x * (pixels.width - 1) / Math.max(1, image.naturalWidth - 1),
    y: y * (pixels.height - 1) / Math.max(1, image.naturalHeight - 1),
  }));
  const result = resamplePerspective(pixels, scaled, width, height, orientation);
  if (!result) return;
  if (canvas.width !== result.width) canvas.width = result.width;
  if (canvas.height !== result.height) canvas.height = result.height;
  canvas.getContext("2d", { alpha: false }).putImageData(new ImageData(result.data, result.width, result.height), 0, 0);
}

function LiveCropPreview({ src, quad, savedCropUrl, outlinePreviewUrl, outlinePreviewHint, savedAspectPreviewUrl,
  measuredAspect, orientation, referenceMatrix }) {
  const canvasRef = useRef(null);
  const [sample, setSample] = useState(null);
  const [failedAspectUrl, setFailedAspectUrl] = useState(null);
  const aspectUrl = savedAspectPreviewUrl !== failedAspectUrl ? savedAspectPreviewUrl : null;
  const previewUrl = savedCropUrl || outlinePreviewUrl || aspectUrl;
  const showingAspect = Boolean(aspectUrl && previewUrl === aspectUrl);
  useEffect(() => {
    if (!src) return;
    const image = new Image();
    image.onload = () => setSample({ src, image, pixels: samplePreviewSource(image) });
    image.src = src;
    return () => { image.onload = null; };
  }, [src]);
  useEffect(() => {
    if (previewUrl || !sample || sample.src !== src || !isValidQuad(quad)) return;
    const frame = requestAnimationFrame(() => {
      if (canvasRef.current) drawQuadPreview(canvasRef.current, sample, quad, measuredAspect, orientation, referenceMatrix);
    });
    return () => cancelAnimationFrame(frame);
  }, [sample, src, quad, measuredAspect, orientation, previewUrl, referenceMatrix]);
  const title = savedCropUrl ? "Gerichteter Ausschnitt" : outlinePreviewUrl ? "Abzugsvorschau zur Prüfung"
    : showingAspect ? savedPrintReviewTitle : "Live-Ausschnitt";
  const detail = savedCropUrl ? "Perspektive wurde aus den vier Ecken korrigiert" : outlinePreviewUrl
    ? outlinePreviewHint || "Vorschau folgt der gezackten Papierkante"
    : showingAspect ? savedPrintReviewHint : "Vorschau folgt allen vier Ecken live";
  return <section className={`live-crop-preview ${savedCropUrl || showingAspect ? "rectified" : ""}`}>
    <header><ZoomIn size={17}/><div><strong>{title}</strong><span>{detail}</span></div></header>
    {previewUrl ? <img src={previewUrl} alt={title} onError={showingAspect ? () => setFailedAspectUrl(aspectUrl) : undefined} />
      : <canvas ref={canvasRef}/>}
  </section>;
}

function CropReview({ albumId, goHome }) {
  const [type, setType] = useState("segmentation"); const [index, setIndex] = useState(0); const [message, setMessage] = useState("");
  const [queue, setQueue] = useState(null);
  const [focusedPhotoId, setFocusedPhotoId] = useState(null);
  const [blockedCropTaskId, setBlockedCropTaskId] = useState(null);
  const [answerError, setAnswerError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [groupSelection, setGroupSelection] = useState(null);
  const [confirmedPaperGrid, setConfirmedPaperGrid] = useState(null);
  const [deferredDraft, setDeferredDraft] = useState(null);
  const [savedDeferredDraft, setSavedDeferredDraft] = useState(null);
  const returnCountId = useRef(null);
  const leavingRef=useRef(false); const goHomeRef=useRef(goHome);
  const queueKey = JSON.stringify([albumId, type, focusedPhotoId]);
  const load = useCallback(async (signal) => {
    const rows = await getReviewQueue(albumId || undefined, type, { includeRejected: false, photoId: focusedPhotoId, signal });
    if (signal?.aborted) return;
    const target = focusedPhotoId && rows[0]?.payload;
    const validTarget = !focusedPhotoId || getCaptureBackgroundReviewTraces(target).length > 0
      || (getPaperGridReview(target?.segmentation_mask)?.cut === null
        && cyclicReviewQuad(sanitizeReviewPolygon(target?.crop_modes?.full_print?.suggested_warp_quad)));
    const countIndex = type === "segmentation" ? rows.findIndex(row => row.id === returnCountId.current) : -1;
    if (type === "segmentation") returnCountId.current = null;
    setQueue({ key: queueKey, rows: validTarget ? rows : [] });
    setIndex(Math.max(0, countIndex));
  }, [albumId, type, focusedPhotoId, queueKey]);
  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal).catch(error => { if (!controller.signal.aborted) setMessage(error.message); });
    return () => controller.abort();
  }, [load]);
  const loading = queue?.key !== queueKey;
  const items = loading ? [] : queue.rows;
  const item = items[index];
  const fullPrint = item?.payload?.crop_modes?.full_print;
  const fullPrintHeld = type === "geometry" && wholePrintApprovalHeld(item?.payload?.crop_modes, item?.payload?.variant_urls);
  const captureReviewPending = type === "geometry" && captureRequiresReview(fullPrint);
  const paperOutlinePending = type === "geometry" && paperOutlineRequiresReview(fullPrint);
  const deferredParentHints = type === "geometry" ? heldDeferredParentCropHints(fullPrint, item?.payload?.review_reasons) : [];
  const deferredParentReviewPending = deferredParentHints.length > 0;
  const motifReviewPending = type === "geometry" && item?.payload?.motif_review_pending;
  const rejectedDraftPending = type === "geometry" && item?.id === blockedCropTaskId;
  const approvalHint = fullPrintHeld || captureReviewPending || paperOutlinePending ? "mobile-print-edge-review-hint"
    : deferredParentReviewPending ? deferredParentHints.map(({ id }) => `mobile-${id}`).join(" ")
      : motifReviewPending ? "mobile-motif-review-hint" : undefined;
  const deferredProposals = getDeferredPrintProposals(item?.payload?.deferred_print_proposals);
  const activeDeferredDraft = deferredDraft?.pageId === item?.id ? deferredDraft : null;
  const countOutlines = type === "segmentation" ? getFragmentReviewOutlines(item?.payload?.detections || []) : [];
  const mergedPairs = type === "segmentation" ? getMergedPrintPairs(item?.payload?.detections || []) : [];
  const captureSplitGroups = type === "segmentation" ? getCaptureSplitReviewGroups(item?.payload?.detections || []) : [];
  const filmGridGroups = type === "segmentation" ? getFilmGridReviewSeams(item?.payload?.detections || []) : [];
  const paperJoinGroups = type === "segmentation" ? getPaperJoinReviewGroups(item?.payload?.detections || []) : [];
  const countDetections = type === "segmentation" ? item?.payload?.detections || [] : [];
  const captureBackgroundReviews = countDetections.filter(photo => getCaptureBackgroundReviewTraces(photo).length > 0);
  const paperGridGroups = getPaperGridReviewGroups(countDetections);
  const paperGridOutlineParents = getPaperGridOutlineParents(countDetections).filter(parent =>
    cyclicReviewQuad(sanitizeReviewPolygon(parent.crop_modes?.full_print?.suggested_warp_quad)));
  const albumLeafGroups = getAlbumLeafReviewGroups(countDetections);
  const countGroups = actionableCountGroups(getActionableFilmGridReviewGroups(filmGridGroups, countDetections), paperJoinGroups,
    getActionablePaperGridReviewGroups(paperGridGroups, countDetections), [...getCrossingFallbackReviewGroups(countDetections), ...albumLeafGroups,
      ...(!deferredProposals.length ? getPrintedIllustrationReviewGroups(countDetections) : [])]);
  const selectedCountGroup = groupSelection?.pageId === item?.id
    ? countGroups.find(({ key }) => key === groupSelection?.key) : null;
  const approvalBlocked = fullPrintHeld || captureReviewPending || paperOutlinePending || deferredParentReviewPending
    || motifReviewPending || rejectedDraftPending || selectedCountGroup;
  const paperGridConfirmed = type === "segmentation" && confirmedPaperGrid?.pageId === item?.id
    && confirmedPaperGrid?.eventId != null && confirmedPaperGrid.eventId === item?.payload?.photo_count_revision;
  useEffect(()=>{ goHomeRef.current=goHome; },[goHome]);
  useEffect(() => {
    if(!history.state?.phoscaCropGuard){
      const guardState={...(history.state || {}),phoscaCropGuard:true}; history.pushState(guardState,"",location.href);
    }
    const keepReviewOpen=event=>{
      if(event.state?.phoscaCropGuard) return;
      if(leavingRef.current){ leavingRef.current=false; goHomeRef.current(); return; }
      history.forward();
    };
    addEventListener("popstate",keepReviewOpen);
    return ()=>removeEventListener("popstate",keepReviewOpen);
  },[]);
  function leaveReview(){ leavingRef.current=true; history.back(); }
  function switchTask(nextType, photoId = null) {
    if (photoId) returnCountId.current = item.id;
    setFocusedPhotoId(photoId);
    setType(nextType);
    setBlockedCropTaskId(null);
    setGroupSelection(null);
    setConfirmedPaperGrid(null);
    setSavedDeferredDraft(null);
    setAnswerError("");
    setMessage("");
  }
  function withoutPageOutlineComparison(task) {
    if (task.page_id !== item.page_id || !task.payload.competing_photo_outline) return task;
    return { ...task, payload: { ...task.payload, competing_photo_outline: null } };
  }
  function updateSavedCropModes(saved) {
    setBlockedCropTaskId(null);
    const cropModes = getCropModes(saved);
    if (cropModes) {
      const mask = saved.segmentation_mask;
      setQueue(current => current?.key === queueKey ? { ...current, rows: current.rows.map(withoutPageOutlineComparison).map(task => {
        if (task.id !== item.id) return task;
        const variantUrls = { ...task.payload.variant_urls, ...saved.crop_download_urls };
        if (!cropModes.full_print?.outline_preview) delete variantUrls.full_print_preview;
        if (!cropModes.image_only?.outline_preview) delete variantUrls.image_only_preview;
        return { ...task, payload: { ...task.payload, crop_modes: cropModes, segmentation_mask: mask,
          correction: saved.enhancement_applied?.correction,
          bounding_box: saved.bounding_box, confidence: saved.segmentation_confidence,
          polygon: mask?.polygon, visible_polygon: mask?.visible_polygon || mask?.polygon,
          warp_quad: cropModes.full_print?.warp_quad || mask?.warp_quad,
          image_quad: mask?.image_quad, print_size_mm: mask?.print_size_mm,
          saved_full_print_review_url: saved.saved_full_print_review_url,
          orientation: saved.enhancement_applied?.correction?.orientation,
          review_reasons: saved.enhancement_applied?.review_reasons,
          motif_review_pending: saved.motif_review_pending, variant_urls: variantUrls } };
      }) } : current);
    }
    setMessage(wholePrintApprovalHeld(cropModes, saved.crop_download_urls) || captureRequiresReview(cropModes?.full_print)
      || paperOutlineRequiresReview(cropModes?.full_print)
      || heldDeferredParentCropHints(cropModes?.full_print, saved.enhancement_applied?.review_reasons).length > 0
      ? "Vier Ecken gespeichert. Der ganze Abzug bleibt in Prüfung."
      : saved.motif_review_pending ? "Vier Ecken gespeichert. Der Motivzuschnitt bleibt in Prüfung."
        : "Vier Ecken gespeichert und Perspektive korrigiert");
  }
  async function reprocessHeldPhoto(photoId) {
    if (type !== "geometry" || !photoId || item?.payload.photo_id !== photoId || submitting) return;
    setSubmitting(true);
    setAnswerError("");
    setMessage("");
    try {
      await reprocessPhoto(photoId);
      const reviews = await getReviewQueue(albumId || undefined, "geometry", { includeRejected: false, photoId });
      const currentReview = reviews.find(review => review.payload.photo_id === photoId);
      const refreshed = currentReview && { ...currentReview, reprocessKey: crypto.randomUUID() };
      setQueue(current => current?.key === queueKey ? { ...current, rows: current.rows.flatMap(review => (
        review.id === item.id ? (refreshed ? [refreshed] : []) : [withoutPageOutlineComparison(review)]
      )) } : current);
      setBlockedCropTaskId(null);
      setMessage("Abzug neu berechnet.");
    } finally {
      setSubmitting(false);
    }
  }
  async function record(value) {
    if (!item || submitting || (value === "correct" && approvalBlocked)) return;
    setSubmitting(true);
    setAnswerError("");
    const metadata = { page_id: item.page_id, ui: "mobile-v2" };
    if (type === "segmentation") {
      metadata.photo_count_revision = item.payload.photo_count_revision;
      if (deferredProposals.length && value === "correct") {
        metadata.deferred_proposal_ids = deferredProposals.map(({ id }) => id);
      }
    }
    try {
      await submitReviewEvent({
        task_type: type, task_id: item.id, answer: value,
        reason_code: value === "correct" ? null : value,
        idempotency_key: crypto.randomUUID(), metadata,
      });
      if (focusedPhotoId) switchTask("segmentation");
      else {
        if (type === "geometry" && value === "wrong_object") {
          setQueue(current => current?.key === queueKey
            ? { ...current, rows: current.rows.map(withoutPageOutlineComparison) } : current);
        }
        setMessage("Antwort gespeichert");
        setIndex((current) => current + 1);
      }
    } catch (error) {
      const motifHint = canEditMobileMotif(item.payload, item.payload?.crop_modes) ? motifReviewHint : motifDesktopReviewHint;
      const recalculate = error.message === "Adjust the print corners or recapture the page before approving this crop.";
      setAnswerError(recalculate ? heldPrintCorrectionHint(fullPrint)
        : error.message === "motif_geometry_review_required" ? motifHint : cropProtectionErrors[error.message] || error.message);
      await load().catch(() => {});
    } finally {
      setSubmitting(false);
    }
  }
  async function resolveCountGroup(group) {
    if (!item || submitting) return;
    const sharedPaper = group.kind === "paper-grid";
    setSubmitting(true);
    setAnswerError("");
    returnCountId.current = item.id;
    try {
      const event = sharedPaper ? await confirmSharedPaperGrid(item.id, groupSelection?.revision, group, "mobile-v2")
        : await excludeCountReviewGroup(item.id, groupSelection?.revision, group, "mobile-v2");
      await load();
      if (sharedPaper) setConfirmedPaperGrid({ pageId: item.id, eventId: event.id });
    } catch (error) {
      setAnswerError(error.message.endsWith("_review_group_stale") ? countGroupReviewError : error.message);
      await load().catch(() => {});
    } finally {
      setGroupSelection(null);
      setSubmitting(false);
    }
  }
  function closeDeferredDraft(saved) {
    setDeferredDraft(null);
    returnCountId.current = item.id;
    if (saved) {
      setSavedDeferredDraft({ pageId: item.id, reason: activeDeferredDraft.reason });
      setQueue(null);
    }
    load().catch(error => setAnswerError(error.message));
  }
  return <div className="mobile-shell crop-review-shell"><MobileHeader title="Cropping prüfen" onBack={leaveReview}/><main className="mobile-content"><div className="review-top"><div><span>Kurze Sitzung</span><strong>{Math.min(index + 1, items.length || 5)} / {Math.min(items.length, 5) || 5}</strong></div><div className="review-progress"><i style={{width:`${Math.min(100, index / Math.max(1, Math.min(items.length,5)) * 100)}%`}}/></div></div>
    <div className="task-switch"><button className={type === "segmentation" ? "active" : ""} onClick={() => switchTask("segmentation")} disabled={submitting || Boolean(activeDeferredDraft)}>Printanzahl</button><button className={type === "geometry" ? "active" : ""} onClick={() => switchTask("geometry")} disabled={submitting || Boolean(activeDeferredDraft)}>Crop-Qualität</button></div>
    {answerError && <p role="alert">{answerError}</p>}
    {paperGridConfirmed && !activeDeferredDraft && <p role="status" id="mobile-paper-grid-count-result">{paperGridCountSaved}</p>}
    {savedDeferredDraft && savedDeferredDraft.pageId === item?.id && !activeDeferredDraft && <p role="status">
      {savedDeferredDraft.reason === sourceFragmentProposal ? "Ein gemeinsamer Abzug gespeichert." : "Abzug erfasst."}
      {" Die Papierkanten bleiben in Prüfung."}
    </p>}
    {item ? <>
      {item.reprocessKey && <p role="status">Abzug neu berechnet. Prüfe die aktualisierte Ansicht.</p>}
      <DetectionOverlay key={activeDeferredDraft?.requestId || item.reprocessKey || "review"} item={item}
        editable={type === "geometry" || Boolean(activeDeferredDraft)} openPaperOutline={Boolean(focusedPhotoId)}
        countOutlines={countOutlines} captureSplitGroups={captureSplitGroups} mergedPairs={activeDeferredDraft ? [] : mergedPairs}
        filmGridGroups={filmGridGroups} selectedCountGroup={selectedCountGroup} paperGridGroups={paperGridGroups}
        paperJoinGroups={paperJoinGroups} onSaved={updateSavedCropModes}
        onReprocess={type === "geometry" ? reprocessHeldPhoto : undefined}
        deferredDraft={activeDeferredDraft} onDeferredDraftClose={closeDeferredDraft}
        onProtectionBlocked={setBlockedCropTaskId} />
      {!activeDeferredDraft && <>
      {!selectedCountGroup && captureBackgroundReviews.map(({ photo_id: photoId }, index) => (
        <button key={photoId} className="magnifier-toggle" type="button" disabled={submitting}
          aria-describedby="mobile-capture-background-review-hint" onClick={() => switchTask("geometry", photoId)}>
          {captureBackgroundReviewAction}{captureBackgroundReviews.length > 1 ? ` (${index + 1})` : ""}
        </button>
      ))}
      {deferredProposals.map((proposal, index) => {
        if (proposal.reason !== sourceFragmentProposal && !standalonePrintNeedsConfirmation(proposal.reason)) return null;
        return <button key={proposal.id} className="magnifier-toggle" type="button"
          aria-describedby={deferredPrintCountHintId(proposal.reason, true)}
          disabled={submitting} onClick={() => {
            const polygon = cyclicReviewQuad(sanitizeReviewPolygon(proposal.polygon));
            if (polygon) setDeferredDraft({ ...proposal, polygon, revision: item.payload.photo_count_revision,
              pageId: item.id, requestId: crypto.randomUUID() });
          }}>
          {proposal.reason === darkBackingProposal ? `Dunkle Fläche prüfen${deferredProposals.length > 1 ? ` (${index + 1})` : ""}`
            : isNestedFallbackProposal(proposal.reason) ? `Innenkontur ${index + 1} prüfen`
              : proposal.reason === captureFabricProposal ? `Stoff am Aufnahmerand ${index + 1} prüfen`
              : [projectivePrintProposal, tiledPrintProposal].includes(proposal.reason) ? `Vorgeschlagenen Abzug ${index + 1} prüfen`
              : embossedAlbumAction(proposal.reason) ?? ([frameBackingProposal, borrowedFallbackProposal].includes(proposal.reason) ? `Vorschlag ${index + 1} am Aufnahmerand prüfen`
                : "Möglichen gemeinsamen Abzug prüfen")}
        </button>;
      })}
      <CountGroupReview groups={countGroups} selected={selectedCountGroup} busy={submitting} sourceUrl={item.preview_url}
        onSelect={key => {
          setGroupSelection(key == null ? null : { pageId: item.id, key, revision: item.payload.photo_count_revision });
          setConfirmedPaperGrid(null);
          setAnswerError("");
        }}
        onConfirm={resolveCountGroup} />
      {paperGridOutlineParents.map(({ photo_id: parentId }, index) => (
        <button key={parentId} className="magnifier-toggle" type="button"
          disabled={submitting || selectedCountGroup?.kind === "paper-grid"}
          aria-describedby={paperGridGroups.length > 0 ? "mobile-paper-grid-count-hint"
            : paperGridConfirmed ? "mobile-paper-grid-count-result" : undefined}
          onClick={() => switchTask("geometry", parentId)}>
          {`Aussenumriss prüfen${paperGridOutlineParents.length > 1 ? ` (${index + 1})` : ""}`}
        </button>
      ))}
      <section className="answer-grid">
        {cropAnswers[type].map(([value,label]) => (
          <button key={value} className={value === "correct" ? "good" : ""}
            disabled={submitting || (value === "correct" && approvalBlocked)}
            aria-describedby={value === "correct" ? approvalHint : undefined}
            onClick={() => record(value)}>
            {value === "correct" && type === "segmentation" && (item.payload?.detection_count === 0
              || deferredProposals.length || item.payload?.missing_print_edge_traces?.length
              || item.payload?.near_capture_print_edge_traces?.length
              || countOutlines.length || captureSplitGroups.length || filmGridGroups.length || paperGridGroups.length || paperJoinGroups.length
              || albumLeafGroups.length || mergedPairs.length)
              ? "Anzahl bestätigen" : label}
          </button>
        ))}
      </section>
      </>}
    </> : <section className="review-done"><Check size={48}/><h2>{message || (loading ? "Prüfaufgaben werden geladen …"
      : focusedPhotoId ? "Dieser Abzug ist nicht mehr in der Prüfung" : "Keine Aufgaben offen")}</h2>
      {focusedPhotoId ? <button onClick={() => switchTask("segmentation")}>Zurück zur Printanzahl</button>
        : <button onClick={() => load().catch(error => setMessage(error.message))}><RefreshCw/> Neu laden</button>}
    </section>}{item && !activeDeferredDraft && <button className="skip-task" disabled={submitting}
      onClick={() => focusedPhotoId ? switchTask("segmentation") : setIndex(i => i + 1)}>
      {focusedPhotoId ? "Zurück zur Printanzahl" : "Überspringen"}
    </button>}
  </main></div>;
}

function EnhancementReview({ albumId, goHome }) {
  const [items, setItems] = useState([]); const [index, setIndex] = useState(0); const [message, setMessage] = useState(""); const [reason, setReason] = useState("");
  const load = useCallback(async () => { setItems(await getReviewQueue(albumId || undefined, "enhancement")); setIndex(0); }, [albumId]);
  useEffect(() => { load().catch(e => setMessage(e.message)); }, [load]); const item = items[index];
  const candidateLeft = useMemo(() => item ? [...item.id].reduce((n,c)=>n+c.charCodeAt(0),0)%2===0 : false, [item]);
  async function answer(side) {
    const semantic = side === "equal" || side === "both_bad" || side === "unsure" ? side : ((side === "left") === candidateLeft ? "candidate" : "original");
    await submitReviewEvent({ task_type: "enhancement", task_id: item.id, answer: side, reason_code: reason || null, idempotency_key: crypto.randomUUID(), metadata: { ui: "mobile-v2", blind_choice: semantic, candidate_name: item.payload.candidate_name } }); setReason(""); setMessage("Vergleich gespeichert"); setIndex(i=>i+1);
  }
  const left = item && (candidateLeft ? item.payload.candidate_url : item.payload.original_url); const right = item && (candidateLeft ? item.payload.original_url : item.payload.candidate_url);
  return <div className="mobile-shell"><MobileHeader title="Enhancement vergleichen" onBack={goHome}/><main className="mobile-content"><div className="review-top"><div><span>Verblindeter Vergleich</span><strong>{Math.min(index+1,items.length||5)} / {Math.min(items.length,5)||5}</strong></div></div>
    {item ? <><div className="blind-compare"><figure><figcaption>A</figcaption><img src={left} alt="Variante A"/></figure><figure><figcaption>B</figcaption><img src={right} alt="Variante B"/></figure></div><div className="reason-chips">{[["","Kein Schaden"],["unnatural","Unnatürlich"],["too_smooth","Zu glatt"],["halo","Halo/überschärft"],["color","Falsche Farbe"],["identity","Gesicht verändert"],["text","Text verändert"]].map(([v,l])=><button className={reason===v?"active":""} key={l} onClick={()=>setReason(v)}>{l}</button>)}</div><section className="answer-grid enhancement-answers"><button onClick={()=>answer("left")}>A besser</button><button onClick={()=>answer("right")}>B besser</button><button className="good" onClick={()=>answer("equal")}>Gleich gut / No-op</button><button onClick={()=>answer("both_bad")}>Beide schlecht</button><button onClick={()=>answer("unsure")}>Unsicher</button></section></> : <section className="review-done"><Check size={48}/><h2>{message || "Noch keine Vergleichspaare"}</h2><p>Sie werden aus bestätigten Crops automatisch vorbereitet.</p><button onClick={load}><RefreshCw/> Neu laden</button></section>}
  </main></div>;
}

function MobilePhotoDetail({ photo, goBack }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    getPhoto(photo.id).then((result) => {
      if (active) setDetail(result);
    }).catch((failure) => {
      if (active) setError(failure.message || "Das Foto konnte nicht geladen werden.");
    });
    return () => { active = false; };
  }, [photo.id]);

  const legacyUrl = detail?.urls?.original || photo.image_url;
  return (
    <div className="mobile-shell">
      <MobileHeader title="Foto" onBack={goBack} />
      <main className="mobile-content">
        {error && <p role="alert">{error}</p>}
        {!detail && !error && <p role="status">Foto wird geladen …</p>}
        {detail && (getCropModes(detail) ? <CropVariants photo={detail} mobile /> : (
          <section className="crop-variants">
            <img className="crop-variant-image" src={legacyUrl} alt="Ausgeschnittenes Foto" />
            <a className="crop-download" href={legacyUrl} download={`photo-${photo.id}.jpg`}>Foto herunterladen</a>
          </section>
        ))}
      </main>
    </div>
  );
}

function BatchGallery({ albums, goHome }) {
  const [batches, setBatches] = useState([]); const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activePhoto, setActivePhoto] = useState(null);
  useEffect(() => {
    let active = true;
    Promise.all(albums.map(async (album) => {
      const [pages, photos] = await Promise.all([listPages(album.id), listPhotos(album.id)]);
      return { album, pages, crops: photos.filter(hasPhotoCrop) };
    })).then((nextBatches) => {
      if (active) {
        setBatches(nextBatches);
        setError("");
      }
    }).catch(() => {
      if (active) setError("Bilder konnten nicht geladen werden.");
    }).finally(() => {
      if (active) setLoading(false);
    });
    return () => { active = false; };
  }, [albums]);

  if (activePhoto) return <MobilePhotoDetail key={activePhoto.id} photo={activePhoto} goBack={() => setActivePhoto(null)} />;

  return <div className="mobile-shell">
    <MobileHeader title="Bilder nach Batch" onBack={goHome} />
    <main className="mobile-content">
      <div className="gallery-heading"><h2>Albumseiten und Fotos</h2></div>
      {loading && <div className="gallery-empty">Bilder werden geladen …</div>}
      {error && <p role="alert">{error}</p>}
      <div className="batch-gallery">
        {batches.map(({ album, pages, crops }) => <section className="batch-section" key={album.id}>
          <header><div><strong>{album.name}</strong><span>{pages.length} Seiten · {crops.length} Fotos</span></div></header>
          {crops.length > 0 && <div className="batch-images batch-crops">
            {crops.map((photo, index) => {
              const url = photo.image_url;
              return <button type="button" key={photo.id} onClick={() => setActivePhoto(photo)}>
                <img src={url} loading="lazy" alt={`Ausgeschnittenes Foto ${index + 1}`} />
              </button>;
            })}
          </div>}
          {album.status === "processing" && <p>Fotos werden ausgeschnitten …</p>}
          {pages.length ? <div className="batch-images">
            {pages.map((page, index) => <a href={`/api/v1/pages/${page.id}/image`} target="_blank" rel="noreferrer" key={page.id}>
              <img src={`/api/v1/pages/${page.id}/image`} loading="lazy" alt={`${album.name}, Seite ${index + 1}`} />
              <span>{index + 1}</span>
            </a>)}
          </div> : <p>Noch keine Aufnahmen.</p>}
        </section>)}
      </div>
    </main>
  </div>;
}

function Progress({ albums, goHome }) {
  const [progress, setProgress] = useState({completed:0,by_type:{}}); const [pages, setPages] = useState(0); const [photos, setPhotos] = useState(0);
  useEffect(() => { Promise.all([getReviewProgress(), ...albums.flatMap(a => [listPages(a.id), listPhotos(a.id)])]).then(([p,...lists]) => { setProgress(p); setPages(lists.filter((_,i)=>i%2===0).reduce((n,x)=>n+x.length,0)); setPhotos(lists.filter((_,i)=>i%2===1).reduce((n,x)=>n+x.length,0)); }); }, [albums]);
  return <div className="mobile-shell"><MobileHeader title="Fortschritt" onBack={goHome}/><main className="mobile-content"><h2 className="mobile-title">Dein lokaler Fortschritt</h2><div className="metric-grid"><div><strong>{pages}</strong><span>Seiten gespeichert</span></div><div><strong>{photos}</strong><span>Prints erkannt</span></div><div><strong>{progress.completed}</strong><span>Reviews erledigt</span></div><div><strong>{progress.by_type?.enhancement || 0}</strong><span>A/B-Vergleiche</span></div></div><section className="mobile-card"><strong>Daten bleiben lokal</strong><p>Originale bleiben unverändert. Feedback wird versioniert; neue Modelle werden erst nach Tests aktiviert.</p></section></main></div>;
}

export default function MobileApp({ uploadQueueScope }) {
  const route = location.pathname; const [view,setView] = useState(route.includes("/capture") ? "capture" : route.includes("/batches") ? "batches" : route.includes("/progress") ? "progress" : route.includes("/enhancement") ? "enhancement" : route.includes("/review") ? "review" : "home");
  const [albums,setAlbums] = useState([]); const [albumId,setAlbumId] = useState(localStorage.getItem("phosca-batch") || "");
  const reload = useCallback(async () => { const next = await listAlbums(); setAlbums(next); if (!albumId && next[0]) setAlbumId(next[0].id); }, [albumId]);
  useEffect(() => { reload(); }, [reload]); useEffect(() => { if(albumId) localStorage.setItem("phosca-batch",albumId); },[albumId]);
  const processing = albums.some((album) => album.status === "processing");
  useEffect(() => {
    if (view !== "batches" || !processing) return;
    const timer = window.setInterval(() => reload().catch(() => {}), 2500);
    return () => window.clearInterval(timer);
  }, [processing, reload, view]);
  const nav = v => { setView(v); history.replaceState({},"", v === "home" ? "/mobile" : `/mobile/${v}`); };
  if(view === "capture") return <Capture albums={albums} activeAlbumId={albumId} setActiveAlbumId={setAlbumId} reload={reload} goHome={()=>nav("home")} uploadQueueScope={uploadQueueScope}/>;
  if(view === "batches") return <BatchGallery albums={albums} goHome={()=>nav("home")}/>;
  if(view === "review") return <CropReview albumId={albumId} goHome={()=>nav("home")}/>;
  if(view === "enhancement") return <EnhancementReview albumId={albumId} goHome={()=>nav("home")}/>;
  if(view === "progress") return <Progress albums={albums} goHome={()=>nav("home")}/>;
  return <div className="mobile-shell"><MobileHeader title="PhoSca"/><main className="mobile-content home"><p className="eyebrow">FÜNF MINUTEN FREI?</p><h1>Was möchtest du tun?</h1><p>Seiten aufnehmen oder mit kurzen Aufgaben die Erkennung verbessern.</p>
    <button className="home-action capture" onClick={()=>nav("capture")}><span><Camera/></span><div><strong>Albumseiten aufnehmen</strong><small>Direkt, geprüft und automatisch analysiert</small></div></button>
    <button className="home-action review" onClick={()=>nav("review")}><span><ListChecks/></span><div><strong>Cropping prüfen</strong><small>Printanzahl und Crop-Grenzen</small></div></button>
    <button className="home-action enhancement" onClick={()=>nav("enhancement")}><span><WandSparkles/></span><div><strong>Enhancement vergleichen</strong><small>Verblindete A/B-Aufgaben und No-op</small></div></button>
    <button className="home-action batches" onClick={()=>nav("batches")}><span><Images/></span><div><strong>Bilder nach Batch</strong><small>Alle Aufnahmen geordnet ansehen</small></div></button>
    <button className="home-action progress" onClick={()=>nav("progress")}><span><Upload/></span><div><strong>Fortschritt ansehen</strong><small>Daten, Reviews und Abdeckung</small></div></button>
    <div className="privacy-note"><CircleHelp/><span><strong>Alles bleibt auf deinem PC.</strong><small>Keine Cloud, keine automatische Modellfreigabe.</small></span></div></main></div>;
}
