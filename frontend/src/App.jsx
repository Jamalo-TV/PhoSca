import {
  Aperture,
  Camera,
  Check,
  Download,
  FileImage,
  FolderPlus,
  GalleryHorizontal,
  Loader2,
  Pencil,
  Play,
  RefreshCw,
  Save,
  Search,
  Upload,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  addManualPhoto,
  analyzeAlbumNow,
  confirmSharedPaperGrid,
  createAlbum,
  deleteSegmentationTrainingImage,
  detectSegmentationTrainingLabels,
  excludeCountReviewGroup,
  getPageOcr,
  getPhoto,
  getReviewQueue,
  getSegmentationTrainingLabels,
  hasPhotoCrop,
  listAlbums,
  listPages,
  listPhotos,
  listSegmentationTrainingImages,
  patchBoundingBox,
  patchOcr,
  prepareSegmentationTrainingDataset,
  reprocessPhoto,
  saveSegmentationTrainingLabels,
  search,
  submitReviewEvent,
  uploadSegmentationTrainingImages,
  uploadFileChunked,
  uploadFiles,
} from "./api.js";
import { captureCameraImage } from "./cameraCapture.js";
import { motifUsesPaperAxes, resizeMotifInPaperPlane } from "./cropPreview.js";
import {
  contactEdgesNeedRefresh, contrastOutlineConfirmation, contrastOutlineNeedsConfirmation, contrastOutlineReviewHint,
} from "./cropReviewApproval.js";
import { queueUpload, retryQueuedUploads } from "./offlineQueue.js";
import MobileApp from "./MobileApp.jsx";
import EdgeMeasurement from "./EdgeMeasurement.jsx";
import {
  actionableCountGroups, adjacentPrintHint, AlbumPageMarkReview, captureSplitReviewHint, CaptureSplitSourceReview,
  clippedOverlapHint, clippedOverlapReason, cropProtectionErrors,
  CropOrientationField, CropVariants, cropModeLabels, getCropModes,
  albumLeafChildStyle, albumLeafTraceStyle,
  crossingFallbackTraceStyles, crossingPrintHint, crossingPrintReason, crossingPrintSaveHint, fragmentCountHint, fragmentOutlineHint, fragmentOutlineReason,
  countGroupReviewError, CountGroupReview, deferredPrintHint, deferredPrintCountHintId, DeferredPrintReviewHints,
  borrowedFallbackProposal, captureFabricProposal, filmGridCountHint, filmGridHint, filmGridReason, filmGridRimHint, frameBackingProposal,
  CompetingPhotoOutlineLegend,
  getDeferredPrintEvidence, getDeferredPrintProposals, getDeferredParentCropHints, getSourceEdgeTraces,
  getMotifBoundaryReviewHint, getMotifMarginMarks, getPrintEdgeReviewHint, getPrintOutlinePreviewHint, heldPrintMotifMargin, mergedPrintAsOneLabel, mergedPrintPairHint, missingPrintEdgeHint,
  nearCapturePrintEdgeHint,
  paperOutlineNeedsConfirmation, NeighborContentReview, nestedPrintHint, PaperEdgePreviewComparison, showsPaperEdgeComparison, printOutlineSaveHint,
  paperGridCountHint, paperGridCountSaved, paperGridFragmentHint, paperGridOuterHint, paperGridReason, paperJoinCountHint, paperJoinHint,
  paperJoinReason, scannerBackingHint, scannerBackingReason,
  isNestedFallbackProposal, requiresStandalonePrintFlag, standalonePrintReviewErrors,
  sourceFragmentCountHint, sourceFragmentProposal, sourceFragmentReviewErrors,
  standalonePrintConfirmation, standalonePrintNeedsConfirmation, DarkBackingEvidenceLegend, darkBackingProposal, darkBackingReviewErrors,
  embossedAlbumAction, embossedPocketReviewErrors,
  getSourceStripReview, SourceStripReview, sourceStripColor, sourceStripContrastColor,
  offPrintAdvisorySource, useAlbumPageMarkReview,
} from "./CropVariants.jsx";
import {
  competingPhotoRegionPaths, competingPhotoRegionStyles, getCompetingPhotoReviewOutlines,
  contrastCrossingTraceHint, contrastCrossingTraceStyle, filmNeighborEdgeColor, getContrastCrossingTraces, getFilmNeighborEdgeIndex,
  getAlbumLeafReviewGroups, getCaptureSplitReviewGroups, getCrossingFallbackReviewGroups, getFilmGridReviewSeams, getFilmGridSeam, getFragmentOutline, getFragmentReviewOutlines,
  getActionableFilmGridReviewGroups, getActionablePaperGridReviewGroups, getPaperGridOutlineParents, getPaperGridReview, getPaperGridReviewGroups,
  getMergedPrintPairs, getPaperJoinReviewGroups, getStoredMotifQuad,
  getRibbedPrintReviewOutline, measuredRibbedPrintOutlineSource, ribbedPrintOutlineHint, ribbedPrintOutlineStyle,
} from "./cropReviewOutlines.js";
import { PrintSizeFields, parsePrintSize, printSizeInputs } from "./PrintSizeFields.jsx";

const tabs = [
  { id: "analyze", label: "Analyze", icon: Play },
  { id: "albums", label: "Albums", icon: GalleryHorizontal },
  { id: "pages", label: "Pages", icon: FileImage },
  { id: "review", label: "Review", icon: Pencil },
  { id: "training", label: "Training", icon: FolderPlus },
  { id: "gallery", label: "Gallery", icon: Aperture },
  { id: "capture", label: "Capture", icon: Camera },
];

function pageAnalysisFinished(status) {
  return status === "completed" || status === "review_needed" || status === "failed";
}

function StatusBadge({ status }) {
  const tone = {
    completed: "status status-green",
    queued: "status status-blue",
    processing: "status status-blue",
    uploading: "status status-blue",
    analyzing: "status status-blue",
    ready: "status status-blue",
    review_needed: "status status-yellow",
    failed: "status status-red",
    pending: "status status-blue",
    waiting: "status",
    selected: "status",
    complete: "status status-green",
    uploaded: "status status-green",
  }[status] || "status";
  return <span className={tone}>{status?.replaceAll("_", " ") || "unknown"}</span>;
}

function SafeText({ text, className = "" }) {
  return <span className={className}>{text ?? ""}</span>;
}

function EmptyState({ label }) {
  return <div className="empty-state">{label}</div>;
}

function variantLabel(name) {
  if (name === "full_print_preview") return "Abzugsvorschau zur Prüfung";
  if (name === "image_only_preview") return "Motivvorschau zur Prüfung";
  return cropModeLabels[name] || name.replaceAll("_", " ");
}

function availableVariantEntries(detail) {
  return Object.entries(detail?.urls || {}).filter(([, url]) => Boolean(url));
}

function comparisonUrls(detail) {
  const before = detail?.urls?.original || detail?.urls?.perspective_corrected || detail?.urls?.enhanced;
  const after = detail?.urls?.conservative_enhanced || detail?.urls?.enhanced || detail?.urls?.premium || before;
  return { before, after };
}

function fileSizeLabel(bytes) {
  if (!bytes) return "0 B";
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function PhotoCompareModal({ detail, imageSrc, title = "Photo", onClose }) {
  const [slider, setSlider] = useState(50);
  const [compareEnhancement, setCompareEnhancement] = useState(false);
  const { before, after } = comparisonUrls(detail);
  const canCompare = Boolean(before && after && before !== after);
  const cropModes = getCropModes(detail);
  const showComparison = canCompare && (!cropModes || compareEnhancement);
  const displaySrc = imageSrc || after || before;

  useEffect(() => {
    setSlider(50);
  }, [detail?.id, imageSrc]);

  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  if (!displaySrc && !canCompare && !cropModes) return null;

  return (
    <div className="fullscreen-overlay" onClick={onClose}>
      <div className="photo-modal" role="dialog" aria-modal="true" aria-label={title} onClick={(event) => event.stopPropagation()}>
        <div className="photo-modal-header">
          <strong>{title}</strong>
          <div className="photo-modal-actions">
            {cropModes && canCompare && (
              <button className="crop-comparison-toggle" type="button" aria-pressed={showComparison} onClick={() => setCompareEnhancement((active) => !active)}>
                {showComparison ? "Fotoausschnitt anzeigen" : "Verbesserung vergleichen"}
              </button>
            )}
            <button className="icon-button" type="button" onClick={onClose} title="Close">
              <X size={16} />
            </button>
          </div>
        </div>
        {showComparison && (
          <div className="compare-viewer" style={{ "--compare-position": `${slider}%` }}>
            <img className="compare-image compare-before" src={before} alt="Before enhancement" />
            <div className="compare-after-wrap">
              <img className="compare-image compare-after" src={after} alt="After enhancement" />
            </div>
            <span className="compare-label compare-label-before">Before</span>
            <span className="compare-label compare-label-after">After</span>
            <span className="compare-divider" />
          </div>
        )}
        {cropModes && <CropVariants key={detail.id} photo={detail} hidden={showComparison} />}
        {!cropModes && !showComparison && <img className="single-photo-preview" src={displaySrc} alt="" />}
        {showComparison && (
          <label className="compare-control">
            <span>Before</span>
            <input
              type="range"
              min="1"
              max="99"
              value={slider}
              onChange={(event) => setSlider(Number(event.target.value))}
              aria-label="Before and after comparison"
            />
            <span>After</span>
          </label>
        )}
      </div>
    </div>
  );
}

function OrderedAnalyzer({ onAlbumReady, onOpenPhoto = () => {} }) {
  const [items, setItems] = useState([]);
  const [albumId, setAlbumId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState("");
  const [message, setMessage] = useState("");
  const [previewSrc, setPreviewSrc] = useState(null);
  const itemsRef = useRef([]);
  const analysisRef = useRef(null);
  const hasPendingUpload = items.some((item) => !item.pageId);
  const canExtract = items.some((item) => !item.page || item.analysisStatus === "failed");

  useEffect(() => {
    itemsRef.current = items;
  }, [items]);

  useEffect(() => {
    return () => {
      analysisRef.current?.abort();
      itemsRef.current.forEach((item) => URL.revokeObjectURL(item.previewUrl));
    };
  }, []);

  function addFiles(fileList) {
    const nextItems = [...fileList].map((file) => ({
      clientId: crypto.randomUUID(),
      file,
      name: file.name,
      size: file.size,
      previewUrl: URL.createObjectURL(file),
      pageId: null,
      uploadStatus: "selected",
      analysisStatus: "waiting",
      page: null,
      photos: [],
      ocrRows: [],
      error: null,
    }));
    setItems((current) => [...current, ...nextItems]);
    setMessage("");
  }

  function removeItem(clientId) {
    setItems((current) => {
      const item = current.find((candidate) => candidate.clientId === clientId);
      if (item) URL.revokeObjectURL(item.previewUrl);
      return current.filter((candidate) => candidate.clientId !== clientId);
    });
  }

  function clearRun() {
    items.forEach((item) => URL.revokeObjectURL(item.previewUrl));
    setItems([]);
    setAlbumId(null);
    setPhase("");
    setMessage("");
  }

  async function ensureAlbum() {
    if (albumId) return albumId;
    const now = new Date();
    const album = await createAlbum({
      name: `Local analysis ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`,
      description: "Created from the local analyze workspace.",
    });
    setAlbumId(album.id);
    return album.id;
  }

  async function uploadPending(targetAlbumId) {
    const pending = items.filter((item) => !item.pageId);
    if (pending.length === 0) return { orderedItems: items, analysisQueued: false };

    setPhase("Uploading");
    setItems((current) =>
      current.map((item) => (item.pageId ? item : { ...item, uploadStatus: "uploading", error: null }))
    );
    const response = await uploadFiles(targetAlbumId, pending.map((item) => item.file));
    const pageByClientId = new Map(pending.map((item, index) => [item.clientId, response.pages[index]]));
    const orderedItems = items.map((item) => {
      const page = pageByClientId.get(item.clientId);
      if (!page) return item;
      return { ...item, pageId: page.page_id, uploadStatus: "uploaded", analysisStatus: page.status };
    });
    setItems(orderedItems);
    return { orderedItems, analysisQueued: response.analysis_queued };
  }

  async function analyzeInOrder() {
    if (!canExtract || busy) return;
    const controller = new AbortController();
    analysisRef.current = controller;
    setBusy(true);
    setMessage("");
    try {
      const targetAlbumId = await ensureAlbum();
      const { orderedItems, analysisQueued } = await uploadPending(targetAlbumId);
      if (controller.signal.aborted) return;
      onAlbumReady(targetAlbumId);
      const targets = orderedItems.filter((item) => item.pageId && (!item.page || item.analysisStatus === "failed"));
      const pageIds = new Set(targets.map((item) => item.pageId));
      const retryIds = new Set(targets.filter((item) => item.analysisStatus === "failed"
        || (analysisQueued === false && !pageAnalysisFinished(item.analysisStatus))).map((item) => item.pageId));

      setPhase("Analyzing");
      setItems((current) =>
        current.map((item) => (pageIds.has(item.pageId) ? { ...item, analysisStatus: "analyzing", error: null, page: null } : item))
      );
      if (retryIds.size > 0) await analyzeAlbumNow(targetAlbumId, [...retryIds]);

      const pending = new Set(pageIds);
      while (pending.size > 0) {
        const latestPages = await listPages(targetAlbumId, controller.signal);
        const finished = latestPages.filter((page) => pending.has(page.id) && pageAnalysisFinished(page.status));
        if (finished.length > 0) {
          const [photos, ocrResponses] = await Promise.all([
            listPhotos(targetAlbumId),
            Promise.all(finished.map((page) => getPageOcr(page.id).catch(() => []))),
          ]);
          if (controller.signal.aborted) return;
          const crops = photos.filter(hasPhotoCrop);
          const results = new Map(finished.map((page, index) => [page.id, {
            analysisStatus: page.status,
            page,
            photos: crops.filter((photo) => photo.page_id === page.id),
            ocrRows: ocrResponses[index],
            error: page.status === "failed" ? page.error_message || "Photo extraction failed." : null,
          }]));
          setItems((current) => current.map((item) => ({ ...item, ...results.get(item.pageId) })));
          finished.forEach((page) => pending.delete(page.id));
        }
        if (pending.size > 0) await new Promise((resolve) => window.setTimeout(resolve, 1500));
      }
      setPhase("Complete");
    } catch (error) {
      if (controller.signal.aborted) return;
      setPhase("Failed");
      setMessage(error.message || "Analysis failed.");
      setItems((current) =>
        current.map((item) =>
          item.analysisStatus === "analyzing" || item.uploadStatus === "uploading"
            ? { ...item, uploadStatus: item.pageId ? "uploaded" : "selected", analysisStatus: "failed", error: error.message }
            : item
        )
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="analyzer-layout">
      <section className="analyzer-panel">
        <div className="section-title">
          <h2>Input</h2>
          <div className="toolbar-row">
            {phase && <StatusBadge status={phase.toLowerCase()} />}
            <label className="icon-button text-button file-pick">
              <Upload size={16} />
              Select
              <input type="file" accept="image/jpeg,image/png" multiple disabled={busy} onChange={(event) => {
                addFiles(event.target.files);
                event.target.value = "";
              }} />
            </label>
          </div>
        </div>
        <div className="analyzer-actions">
          <button className="primary-button" type="button" disabled={busy || !canExtract} onClick={analyzeInOrder}>
            {busy ? <Loader2 className="spin-icon" size={16} /> : <Upload size={16} />}
            {hasPendingUpload || items.length === 0 ? "Upload & extract" : "Retry extraction"}
          </button>
          <button className="icon-button" type="button" disabled={busy || items.length === 0} onClick={clearRun} title="Clear run">
            <RefreshCw size={16} />
          </button>
        </div>
        {message && <p className="run-message">{message}</p>}
        {items.length === 0 ? (
          <EmptyState label="No images selected." />
        ) : (
          <div className="ordered-list">
            {items.map((item, index) => (
              <div className="ordered-row" key={item.clientId}>
                <span className="order-index">{index + 1}</span>
                <img
                  src={item.previewUrl}
                  alt=""
                  style={{ cursor: "pointer" }}
                  onClick={() => setPreviewSrc(item.previewUrl)}
                />
                <div className="ordered-meta">
                  <strong>{item.name}</strong>
                  <span>{fileSizeLabel(item.size)}</span>
                  <StatusBadge status={item.uploadStatus} />
                </div>
                <button className="icon-button" type="button" disabled={busy} onClick={() => removeItem(item.clientId)} title="Remove">
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="analyzer-panel">
        <div className="section-title">
          <h2>Output</h2>
          <span className="muted-text">{items.length ? `${items.length} item${items.length === 1 ? "" : "s"}` : ""}</span>
        </div>
        {items.length === 0 ? (
          <EmptyState label="No output yet." />
        ) : (
          <div className="ordered-list output-list">
            {items.map((item, index) => (
              <div className="output-row" key={`${item.clientId}-output`}>
                <div className="output-heading">
                  <span className="order-index">{index + 1}</span>
                  <div>
                    <strong>{item.name}</strong>
                  </div>
                  <StatusBadge status={item.error ? "failed" : item.analysisStatus} />
                </div>
                {item.analysisStatus === "analyzing" && (
                  <div className="output-waiting">
                    <Loader2 className="spin-icon" size={18} />
                    <span>Analyzing</span>
                  </div>
                )}
                {item.error && <p className="error-text">{item.error}</p>}
                {!item.page && !item.error && item.analysisStatus !== "analyzing" && <p className="muted-text">Waiting for upload.</p>}
                {item.page && (
                  <div className="output-details">
                    <div className="stats-row">
                      <span>{item.photos.length} photos</span>
                      <span>{item.ocrRows.length} OCR rows</span>
                      <span>{item.page?.blur_score ? `Blur ${item.page.blur_score.toFixed(1)}` : "Blur pending"}</span>
                    </div>
                    {item.page?.status === "review_needed" && item.page.error_message && (
                      <p className="muted-text">{item.page.error_message}</p>
                    )}
                    {item.ocrRows.length > 0 && (
                      <div className="ocr-output">
                        {item.ocrRows.map((row) => (
                          <SafeText key={row.id} text={row.text_content} />
                        ))}
                      </div>
                    )}
                    {item.photos.length > 0 && (
                      <div className="mini-photo-grid">
                        {item.photos.map((photo, index) => {
                          const imageUrl = photo.image_url;
                          const cropModes = getCropModes(photo);
                          const cropUrls = photo.crop_download_urls;
                          const downloadUrl = cropModes ? cropUrls?.full_print || cropUrls?.image_only : imageUrl;
                          const motifDownload = cropModes && !cropUrls?.full_print && !!cropUrls?.image_only;
                          return (
                            <div className="mini-photo-card" key={photo.id}>
                              <button type="button" className="variant-preview-button" onClick={() => onOpenPhoto(photo)}>
                                <img src={imageUrl} alt={`Open extracted photo ${index + 1}`} />
                              </button>
                              {downloadUrl && (
                                <a className="icon-button text-button" href={downloadUrl}
                                  download={`photo-${photo.id}${motifDownload ? "-image_only" : ""}.jpg`}
                                  aria-label={motifDownload ? "Nur Bildmotiv herunterladen" : `Download extracted photo ${index + 1}`}>
                                  <Download size={14} />
                                  {motifDownload ? "Nur Bildmotiv" : "Download"}
                                </a>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </section>
      {previewSrc && <PhotoCompareModal imageSrc={previewSrc} title="Input page" onClose={() => setPreviewSrc(null)} />}
    </div>
  );
}

function AlbumList({ albums, selectedAlbumId, pages = [], photos = [], onOpen, onCreate, onOpenPhoto = () => {}, loading }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const selectedAlbum = albums.find((album) => album.id === selectedAlbumId) || null;
  const crops = photos.filter(hasPhotoCrop);

  async function submit(event) {
    event.preventDefault();
    if (!name.trim()) return;
    await onCreate({ name: name.trim(), description: description.trim() || null });
    setName("");
    setDescription("");
  }

  return (
    <div className="content-grid">
      <section className="work-surface">
        <div className="section-title">
          <h2>Album List</h2>
          {loading && <Loader2 className="spin-icon" size={18} />}
        </div>
        {albums.length === 0 ? (
          <EmptyState label="No albums yet." />
        ) : (
          <div className="album-grid">
            {albums.map((album) => {
              const progress = album.total_pages ? Math.round((album.processed_pages / album.total_pages) * 100) : 0;
              return (
                <button
                  type="button"
                  key={album.id}
                  className={`album-card ${selectedAlbumId === album.id ? "selected" : ""}`}
                  onClick={() => onOpen(album.id)}
                  onDoubleClick={() => onOpen(album.id)}
                  title={`Open ${album.name}`}
                >
                  <div className="album-card-row">
                    <strong>{album.name}</strong>
                    <StatusBadge status={album.status} />
                  </div>
                  <SafeText text={album.description || "No description"} className="muted-text" />
                  <div className="progress-track">
                    <span style={{ width: `${progress}%` }} />
                  </div>
                  <div className="stats-row">
                    <span>{album.processed_pages}/{album.total_pages} pages</span>
                    <span>{album.photos_extracted} photos</span>
                    <span>{album.review_needed_count} review</span>
                  </div>
                </button>
              );
            })}
          </div>
        )}
        {selectedAlbum && (
          <div className="album-detail-panel">
            <div className="section-title">
              <h2>{selectedAlbum.name}</h2>
              <StatusBadge status={selectedAlbum.status} />
            </div>
            {pages.length > 0 && (
              <div className="album-page-strip">
                {pages.map((page) => (
                  <div className="album-page-thumb" key={page.id}>
                    <img src={`/api/v1/pages/${page.id}/image`} alt="" />
                    <span>{page.original_filename}</span>
                  </div>
                ))}
              </div>
            )}
            {crops.length > 0 ? (
              <div className="album-photo-grid">
                {crops.map((photo) => (
                  <button type="button" className="photo-tile" key={photo.id} onClick={() => onOpenPhoto(photo)}>
                    <img src={photo.image_url} alt="" />
                    <div>
                      <StatusBadge status={photo.status} />
                      <span>{photo.segmentation_confidence ? `${Math.round(photo.segmentation_confidence * 100)}%` : "manual"}</span>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <EmptyState label={selectedAlbum.status === "processing" ? "Extracting photos…" : "No extracted photos in this album yet."} />
            )}
          </div>
        )}
      </section>
      <section className="side-surface">
        <div className="section-title">
          <h2>Create Album</h2>
          <FolderPlus size={18} />
        </div>
        <form className="stack" onSubmit={submit}>
          <label>
            <span>Name</span>
            <input value={name} onChange={(event) => setName(event.target.value)} maxLength={255} />
          </label>
          <label>
            <span>Description</span>
            <textarea value={description} onChange={(event) => setDescription(event.target.value)} rows={4} maxLength={10000} />
          </label>
          <button className="primary-button" type="submit">
            <FolderPlus size={16} />
            Create
          </button>
        </form>
      </section>
    </div>
  );
}

function PageGrid({ albumId, pages, onAnalyze, onUpload, onSelectPage, selectedPageId, analyzing, processing }) {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");

  async function uploadSelected() {
    if (!albumId || files.length === 0 || uploading) return;
    setUploading(true);
    setMessage("");
    try {
      const analysisQueued = await onUpload(files);
      setFiles([]);
      setMessage(analysisQueued ? "Upload complete. Extracting photos automatically." : "Upload complete.");
    } catch (error) {
      setMessage(error.message || "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="content-grid">
      <section className="work-surface">
        <div className="section-title">
          <h2>Page Grid</h2>
          <div className="toolbar-row">
            {processing && <StatusBadge status="processing" />}
            <button className="icon-button text-button" type="button" disabled={analyzing || pages.every((page) => page.status === "completed")} onClick={onAnalyze}>
              {analyzing ? <Loader2 className="spin-icon" size={16} /> : <Play size={16} />}
              Extract photos
            </button>
          </div>
        </div>
        {!albumId ? <EmptyState label="Select an album first." /> : null}
        {albumId && pages.length === 0 ? <EmptyState label="No uploaded pages." /> : null}
        <div className="page-grid">
          {pages.map((page) => (
            <button
              type="button"
              className={`page-tile ${selectedPageId === page.id ? "selected" : ""}`}
              key={page.id}
              onClick={() => onSelectPage(page.id)}
            >
              <div className="page-thumb">
                <img src={`/api/v1/pages/${page.id}/image`} alt="" />
              </div>
              <div className="page-meta">
                <strong>{page.original_filename}</strong>
                <StatusBadge status={page.status} />
                <span>Blur {page.blur_score ? page.blur_score.toFixed(1) : "pending"}</span>
              </div>
            </button>
          ))}
        </div>
      </section>
      <section className="side-surface">
        <div className="section-title">
          <h2>Upload</h2>
          <Upload size={18} />
        </div>
        <div className="stack">
          <input type="file" accept="image/jpeg,image/png" multiple onChange={(event) => setFiles([...event.target.files])} />
          <button className="primary-button" type="button" disabled={!albumId || files.length === 0 || uploading} onClick={uploadSelected}>
            {uploading ? <Loader2 className="spin-icon" size={16} /> : <Upload size={16} />}
            Upload {files.length || ""}
          </button>
          {message && <p className="run-message">{message}</p>}
        </div>
      </section>
    </div>
  );
}

function clampUnit(value) {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function boxToPolygon(box) {
  return [
    { x: clampUnit(box.x1), y: clampUnit(box.y1) },
    { x: clampUnit(box.x2), y: clampUnit(box.y1) },
    { x: clampUnit(box.x2), y: clampUnit(box.y2) },
    { x: clampUnit(box.x1), y: clampUnit(box.y2) },
  ];
}

function sanitizePolygon(polygon) {
  if (!Array.isArray(polygon) || polygon.length < 4) return null;
  const raw = polygon.map((point) => ({ x: Number(point?.x), y: Number(point?.y) }));
  if (!raw.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y))) return null;
  return raw.map((point) => ({ x: clampUnit(point.x), y: clampUnit(point.y) }));
}

function polygonFromPhoto(photo) {
  return sanitizePolygon(photo.visible_polygon)
    || sanitizePolygon(photo.segmentation_mask?.visible_polygon)
    || sanitizePolygon(photo.segmentation_mask?.polygon)
    || boxToPolygon(photo.bounding_box);
}

function cyclicQuad(points) {
  if (!Array.isArray(points) || points.length !== 4) return null;
  const center = points.reduce((sum, point) => ({ x: sum.x + point.x / 4, y: sum.y + point.y / 4 }), { x: 0, y: 0 });
  const ordered = [...points].sort((a, b) => Math.atan2(a.y - center.y, a.x - center.x) - Math.atan2(b.y - center.y, b.x - center.x));
  const start = ordered.reduce((best, point, index) => (point.x + point.y < ordered[best].x + ordered[best].y ? index : best), 0);
  return [...ordered.slice(start), ...ordered.slice(0, start)];
}

function convexHull(points) {
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  if (sorted.length <= 1) return sorted;
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), point) <= 0) lower.pop();
    lower.push(point);
  }
  const upper = [];
  for (const point of [...sorted].reverse()) {
    while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), point) <= 0) upper.pop();
    upper.push(point);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

function minimumAreaQuad(points) {
  const hull = convexHull(points);
  if (hull.length === 4) return cyclicQuad(hull);
  if (hull.length < 2) return null;
  let best = null;
  hull.forEach((point, index) => {
    const next = hull[(index + 1) % hull.length];
    const angle = Math.atan2(next.y - point.y, next.x - point.x);
    const cos = Math.cos(angle); const sin = Math.sin(angle);
    const rotated = hull.map((candidate) => ({ x: candidate.x * cos + candidate.y * sin, y: -candidate.x * sin + candidate.y * cos }));
    const xs = rotated.map((candidate) => candidate.x); const ys = rotated.map((candidate) => candidate.y);
    const minX = Math.min(...xs); const maxX = Math.max(...xs); const minY = Math.min(...ys); const maxY = Math.max(...ys);
    const area = (maxX - minX) * (maxY - minY);
    if (best && best.area <= area) return;
    const inverse = (candidate) => ({ x: candidate.x * cos - candidate.y * sin, y: candidate.x * sin + candidate.y * cos });
    best = { area, points: [{ x: minX, y: minY }, { x: maxX, y: minY }, { x: maxX, y: maxY }, { x: minX, y: maxY }].map(inverse) };
  });
  return cyclicQuad(best?.points || []);
}

function boundingBoxFromPolygon(polygon) {
  const xs = polygon.map((point) => point.x);
  const ys = polygon.map((point) => point.y);
  return {
    x1: Math.min(...xs),
    y1: Math.min(...ys),
    x2: Math.max(...xs),
    y2: Math.max(...ys),
  };
}

function draftShapeFromPhoto(photo) {
  const visiblePolygon = polygonFromPhoto(photo);
  const cropModes = photo.crop_modes || getCropModes(photo);
  const correction = photo.correction || photo.enhancement_applied?.correction;
  const protectedOutline = correction?.album_page_mark_review?.source === offPrintAdvisorySource
    && sanitizePolygon(cropModes?.full_print?.warp_quad);
  const printSize = photo.segmentation_mask?.print_size_mm ?? photo.print_size_mm;
  const measuredQuad = printSize && sanitizePolygon(photo.segmentation_mask?.warp_quad);
  const manualQuad = photo.segmentation_confidence === null && sanitizePolygon(photo.segmentation_mask?.warp_quad);
  const serverQuad = manualQuad || sanitizePolygon(photo.warp_quad)
    || sanitizePolygon(cropModes?.full_print?.warp_quad)
    || sanitizePolygon(photo.correction?.warp_quad)
    || sanitizePolygon(photo.enhancement_applied?.correction?.warp_quad)
    || sanitizePolygon(photo.segmentation_mask?.warp_quad);
  const warpQuad = protectedOutline?.length === 4 ? protectedOutline : measuredQuad?.length === 4 ? measuredQuad : (
    cyclicQuad(serverQuad?.length === 4 ? serverQuad : [])
    || (visiblePolygon.length === 4 ? cyclicQuad(visiblePolygon) : minimumAreaQuad(visiblePolygon))
    || boxToPolygon(photo.bounding_box)
  );
  const imageQuad = cyclicQuad(sanitizePolygon(getStoredMotifQuad(photo, cropModes?.image_only)));
  return {
    visiblePolygon,
    warpQuad,
    imageQuad,
    visibleFollowsWarp: !photo.visible_polygon && !photo.segmentation_mask?.visible_polygon && visiblePolygon.length === 4,
    boundingBox: boundingBoxFromPolygon(warpQuad),
    printSizeFields: printSizeInputs(printSize),
    orientationDegrees: photo.segmentation_mask?.orientation_degrees ?? null,
  };
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersects = yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi || 1) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

function maskFromShape(shape, imageQuad = null) {
  return {
    polygon: shape.warpQuad.map((point) => ({ x: clampUnit(point.x), y: clampUnit(point.y) })),
    visible_polygon: shape.visiblePolygon.map((point) => ({ x: clampUnit(point.x), y: clampUnit(point.y) })),
    warp_quad: shape.warpQuad.map((point) => ({ x: clampUnit(point.x), y: clampUnit(point.y) })),
    ...(imageQuad ? { image_quad: imageQuad } : {}),
    source: "manual_quad",
  };
}

function trainingImageSrc(image) {
  return image ? `/api/v1/training/segmentation/images/${encodeURIComponent(image.name)}` : "";
}

function normalizeTrainingPolygon(polygon, index = 0) {
  const rawPoints = polygon?.points || polygon?.polygon || polygon;
  if (!Array.isArray(rawPoints) || rawPoints.length < 4) return null;
  const points = rawPoints.map((point) => ({ x: clampUnit(Number(point.x)), y: clampUnit(Number(point.y)) }));
  return {
    id: polygon?.id || `poly-${Date.now()}-${index}-${Math.random().toString(16).slice(2)}`,
    points,
    source: polygon?.source || "manual",
    confidence: typeof polygon?.confidence === "number" ? polygon.confidence : null,
  };
}

function defaultTrainingPolygon() {
  return {
    id: `poly-${Date.now()}`,
    points: [
      { x: 0.22, y: 0.22 },
      { x: 0.78, y: 0.22 },
      { x: 0.78, y: 0.72 },
      { x: 0.22, y: 0.72 },
    ],
    source: "manual",
    confidence: null,
  };
}

function TrainingCanvas({ image, polygons, selectedPolygonId, onSelectPolygon, onChangePolygons }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const dragRef = useRef(null);
  const magnifierRef = useRef(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const loadedImage = imageRef.current;
    if (!canvas || !loadedImage || !image) return;
    const context = canvas.getContext("2d");
    const maxWidth = canvas.parentElement?.clientWidth || 900;
    const ratio = loadedImage.naturalHeight / loadedImage.naturalWidth || 0.75;
    canvas.width = maxWidth;
    canvas.height = Math.max(280, Math.round(maxWidth * ratio));
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(loadedImage, 0, 0, canvas.width, canvas.height);

    polygons.forEach((polygon, index) => {
      const selected = polygon.id === selectedPolygonId;
      const points = polygon.points.map((point) => ({ x: point.x * canvas.width, y: point.y * canvas.height }));
      context.beginPath();
      points.forEach((point, pointIndex) => {
        if (pointIndex === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      });
      context.closePath();
      context.fillStyle = selected ? "rgba(15, 118, 110, 0.16)" : "rgba(245, 158, 11, 0.12)";
      context.strokeStyle = selected ? "#0f766e" : "#f59e0b";
      context.lineWidth = selected ? 4 : 2;
      context.fill();
      context.stroke();

      context.fillStyle = selected ? "#0f766e" : "#17201d";
      context.font = "700 13px Inter, sans-serif";
      context.fillText(String(index + 1), points[0].x + 8, points[0].y + 18);

      points.forEach((point) => {
        context.fillStyle = "#ffffff";
        context.strokeStyle = selected ? "#0f766e" : "#17201d";
        context.lineWidth = 2;
        context.fillRect(point.x - 5, point.y - 5, 10, 10);
        context.strokeRect(point.x - 5, point.y - 5, 10, 10);
      });
    });

    const magnifier = magnifierRef.current;
    if (magnifier) {
      const radius = 58;
      const zoom = 3;
      const edgeGap = 14;
      // Polygon points are stored as proportions of the canvas. Convert them
      // before placing the lens; canvas drawing APIs use pixels.
      const point = {
        x: magnifier.point.x * canvas.width,
        y: magnifier.point.y * canvas.height,
      };
      const preferredY = point.y - radius - edgeGap;
      const lensX = Math.min(Math.max(point.x, radius + 8), canvas.width - radius - 8);
      const lensY = preferredY - radius >= 8
        ? preferredY
        : Math.min(point.y + radius + edgeGap, canvas.height - radius - 8);
      const sourceRadiusX = (radius / zoom) * (loadedImage.naturalWidth / canvas.width);
      const sourceRadiusY = (radius / zoom) * (loadedImage.naturalHeight / canvas.height);
      const sourceX = point.x * (loadedImage.naturalWidth / canvas.width);
      const sourceY = point.y * (loadedImage.naturalHeight / canvas.height);
      const sourceWidth = sourceRadiusX * 2;
      const sourceHeight = sourceRadiusY * 2;
      // Keep the sampled area on the image at its borders. This avoids a
      // transparent/blank lens when a corner is dragged right to an edge.
      const sourceLeft = Math.min(Math.max(sourceX - sourceRadiusX, 0), Math.max(0, loadedImage.naturalWidth - sourceWidth));
      const sourceTop = Math.min(Math.max(sourceY - sourceRadiusY, 0), Math.max(0, loadedImage.naturalHeight - sourceHeight));
      const focusX = lensX - radius + ((sourceX - sourceLeft) / sourceWidth) * radius * 2;
      const focusY = lensY - radius + ((sourceY - sourceTop) / sourceHeight) * radius * 2;

      context.save();
      context.beginPath();
      context.arc(lensX, lensY, radius, 0, Math.PI * 2);
      context.clip();
      context.drawImage(
        loadedImage,
        sourceLeft,
        sourceTop,
        sourceWidth,
        sourceHeight,
        lensX - radius,
        lensY - radius,
        radius * 2,
        radius * 2
      );
      context.strokeStyle = "rgba(15, 118, 110, 0.9)";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(lensX - radius, focusY);
      context.lineTo(lensX + radius, focusY);
      context.moveTo(focusX, lensY - radius);
      context.lineTo(focusX, lensY + radius);
      context.stroke();
      context.restore();

      context.strokeStyle = "#ffffff";
      context.lineWidth = 5;
      context.beginPath();
      context.arc(lensX, lensY, radius + 2, 0, Math.PI * 2);
      context.stroke();
      context.strokeStyle = "#0f766e";
      context.lineWidth = 3;
      context.beginPath();
      context.arc(lensX, lensY, radius + 2, 0, Math.PI * 2);
      context.stroke();
    }
  }, [image, polygons, selectedPolygonId]);

  useEffect(() => {
    draw();
  }, [draw]);

  const drawRef = useRef(draw);
  useEffect(() => {
    drawRef.current = draw;
  }, [draw]);

  useEffect(() => {
    let cancelled = false;
    imageRef.current = null;

    // Clear the old page immediately. Without this, a late image load from a
    // previous selection can redraw over the newly selected page and leave a
    // partially blank canvas in Chromium.
    const canvas = canvasRef.current;
    if (canvas) {
      const context = canvas.getContext("2d");
      context.clearRect(0, 0, canvas.width, canvas.height);
    }

    if (!image) {
      return () => {
        cancelled = true;
      };
    }

    const loadedImage = new Image();
    loadedImage.decoding = "async";
    loadedImage.onload = () => {
      if (cancelled) return;
      imageRef.current = loadedImage;
      drawRef.current();
    };
    loadedImage.src = trainingImageSrc(image);

    return () => {
      cancelled = true;
      loadedImage.onload = null;
      if (imageRef.current === loadedImage) imageRef.current = null;
    };
  }, [image?.name]);

  useEffect(() => {
    window.addEventListener("resize", draw);
    const container = canvasRef.current?.parentElement;
    const observer = container && typeof ResizeObserver !== "undefined" ? new ResizeObserver(draw) : null;
    observer?.observe(container);
    return () => {
      window.removeEventListener("resize", draw);
      observer?.disconnect();
    };
  }, [draw]);

  function pointFromEvent(event) {
    const rect = canvasRef.current.getBoundingClientRect();
    return { x: clampUnit((event.clientX - rect.left) / rect.width), y: clampUnit((event.clientY - rect.top) / rect.height) };
  }

  function onPointerDown(event) {
    if (!canvasRef.current) return;
    const point = pointFromEvent(event);
    for (const polygon of [...polygons].reverse()) {
      const pointIndex = polygon.points.findIndex((corner) => Math.hypot(point.x - corner.x, point.y - corner.y) < 0.035);
      if (pointIndex >= 0) {
        onSelectPolygon(polygon.id);
        dragRef.current = { polygonId: polygon.id, pointIndex };
        magnifierRef.current = { point };
        canvasRef.current.setPointerCapture(event.pointerId);
        draw();
        return;
      }
      if (pointInPolygon(point, polygon.points)) {
        onSelectPolygon(polygon.id);
        return;
      }
    }
  }

  function onPointerMove(event) {
    if (!dragRef.current) return;
    const point = pointFromEvent(event);
    const { polygonId, pointIndex } = dragRef.current;
    magnifierRef.current = { point };
    draw();
    onChangePolygons((current) =>
      current.map((polygon) =>
        polygon.id === polygonId
          ? {
              ...polygon,
              source: polygon.source === "manual_yolo" ? "manual" : polygon.source,
              points: polygon.points.map((corner, index) => (index === pointIndex ? point : corner)),
            }
          : polygon
      )
    );
  }

  function onPointerUp(event) {
    if (dragRef.current && canvasRef.current) {
      canvasRef.current.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
    magnifierRef.current = null;
    draw();
  }

  if (!image) return <EmptyState label="No training image selected." />;

  return (
    <div className="training-canvas-wrap">
      <canvas
        ref={canvasRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        aria-label="Training annotation canvas"
      />
    </div>
  );
}

function TrainingAnnotationView() {
  const [workspace, setWorkspace] = useState(null);
  const [selectedName, setSelectedName] = useState(null);
  const [polygons, setPolygons] = useState([]);
  const [selectedPolygonId, setSelectedPolygonId] = useState(null);
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("");
  const [dirty, setDirty] = useState(false);

  const images = workspace?.images || [];
  const selectedImage = images.find((image) => image.name === selectedName) || null;
  const selectedPolygon = polygons.find((polygon) => polygon.id === selectedPolygonId) || polygons[0] || null;
  const completedCount = images.filter((image) => image.has_label).length;
  const progressPercent = images.length ? Math.round((completedCount / images.length) * 100) : 0;

  const refreshWorkspace = useCallback(async (preferredName = null) => {
    const next = await listSegmentationTrainingImages();
    setWorkspace(next);
    setSelectedName((currentName) => {
      if (preferredName && next.images.some((image) => image.name === preferredName)) return preferredName;
      if (currentName && next.images.some((image) => image.name === currentName)) return currentName;
      return next.images[0]?.name || null;
    });
    return next;
  }, []);

  useEffect(() => {
    refreshWorkspace().catch((error) => setMessage(error.message));
  }, [refreshWorkspace]);

  useEffect(() => {
    if (!selectedImage) {
      setPolygons([]);
      setSelectedPolygonId(null);
      return;
    }
    let cancelled = false;
    async function loadLabels() {
      setBusy("loading");
      setMessage("");
      try {
        const payload = await getSegmentationTrainingLabels(selectedImage.name);
        let nextPolygons = (payload.polygons || []).map(normalizeTrainingPolygon).filter(Boolean);
        let loadedSuggestions = false;
        if (!payload.image.has_label && nextPolygons.length === 0) {
          const suggestions = payload.suggestions ?? (await detectSegmentationTrainingLabels(selectedImage.name)).polygons;
          nextPolygons = (suggestions || []).map(normalizeTrainingPolygon).filter(Boolean);
          loadedSuggestions = nextPolygons.length > 0;
        }
        if (!cancelled) {
          setPolygons(nextPolygons);
          setSelectedPolygonId(nextPolygons[0]?.id || null);
          setDirty(loadedSuggestions);
        }
      } catch (error) {
        if (!cancelled) setMessage(error.message || "Unable to load labels.");
      } finally {
        if (!cancelled) setBusy("");
      }
    }
    loadLabels();
    return () => {
      cancelled = true;
    };
  }, [selectedImage?.name]);

  function updatePolygons(updater) {
    setPolygons((current) => (typeof updater === "function" ? updater(current) : updater));
    setDirty(true);
  }

  async function uploadImages(fileList) {
    const files = [...fileList];
    if (files.length === 0) return;
    setBusy("uploading");
    try {
      const response = await uploadSegmentationTrainingImages(files);
      await refreshWorkspace(response.images?.[0]?.name || null);
      setMessage(`${files.length} image${files.length === 1 ? "" : "s"} uploaded.`);
    } catch (error) {
      setMessage(error.message || "Upload failed.");
    } finally {
      setBusy("");
    }
  }

  async function runAutoDetect() {
    if (!selectedImage) return;
    setBusy("detecting");
    try {
      const detected = await detectSegmentationTrainingLabels(selectedImage.name);
      const nextPolygons = (detected.polygons || []).map(normalizeTrainingPolygon).filter(Boolean);
      updatePolygons(nextPolygons);
      setSelectedPolygonId(nextPolygons[0]?.id || null);
      setMessage(`${nextPolygons.length} suggestion${nextPolygons.length === 1 ? "" : "s"} loaded.`);
    } catch (error) {
      setMessage(error.message || "Detection failed.");
    } finally {
      setBusy("");
    }
  }

  function addPolygon() {
    const polygon = defaultTrainingPolygon();
    updatePolygons((current) => [...current, polygon]);
    setSelectedPolygonId(polygon.id);
  }

  function deleteSelectedPolygon() {
    if (!selectedPolygon) return;
    updatePolygons((current) => current.filter((polygon) => polygon.id !== selectedPolygon.id));
    setSelectedPolygonId(polygons.find((polygon) => polygon.id !== selectedPolygon.id)?.id || null);
  }

  async function saveLabels() {
    if (!selectedImage) return;
    setBusy("saving");
    try {
      const payload = polygons.map((polygon) => ({
        points: polygon.points.map((point) => ({ x: clampUnit(point.x), y: clampUnit(point.y) })),
        source: "manual",
      }));
      const response = await saveSegmentationTrainingLabels(selectedImage.name, payload);
      const nextPolygons = (response.polygons || []).map(normalizeTrainingPolygon).filter(Boolean);
      setPolygons(nextPolygons);
      setSelectedPolygonId(nextPolygons[0]?.id || null);
      setDirty(false);
      const refreshed = await refreshWorkspace(selectedImage.name);
      setMessage(`Saved ${selectedImage.name}. ${refreshed.images.filter((image) => image.has_label).length}/${refreshed.images.length} pages completed.`);
    } catch (error) {
      setMessage(error.message || "Save failed.");
    } finally {
      setBusy("");
    }
  }

  function selectNextPending() {
    if (images.length === 0) return;
    const start = Math.max(0, images.findIndex((image) => image.name === selectedImage?.name));
    for (let offset = 1; offset <= images.length; offset += 1) {
      const candidate = images[(start + offset) % images.length];
      if (!candidate.has_label) {
        setSelectedName(candidate.name);
        return;
      }
    }
    setMessage("All pages have saved labels.");
  }

  async function deleteSelectedImage() {
    if (!selectedImage) return;
    setBusy("deleting");
    try {
      await deleteSegmentationTrainingImage(selectedImage.name);
      setPolygons([]);
      setSelectedPolygonId(null);
      await refreshWorkspace(null);
      setMessage("Image removed.");
    } catch (error) {
      setMessage(error.message || "Delete failed.");
    } finally {
      setBusy("");
    }
  }

  async function prepareDataset() {
    setBusy("preparing");
    try {
      const result = await prepareSegmentationTrainingDataset();
      const valid = result.validation?.valid ? "valid" : "needs review";
      setMessage(`Dataset ${valid}: ${result.validation?.images_train || 0} train, ${result.validation?.images_val || 0} val.`);
    } catch (error) {
      setMessage(error.message || "Dataset preparation failed.");
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="training-layout">
      <section className="work-surface">
        <div className="section-title">
          <h2>Training Labels</h2>
          <div className="toolbar-row">
            <button className="icon-button" type="button" disabled={!selectedImage || Boolean(busy)} onClick={runAutoDetect} title="Auto detect">
              {busy === "detecting" ? <Loader2 className="spin-icon" size={16} /> : <RefreshCw size={16} />}
            </button>
            <button className="icon-button" type="button" disabled={!selectedImage || Boolean(busy)} onClick={addPolygon} title="Add polygon">
              <FolderPlus size={16} />
            </button>
            <button className="icon-button" type="button" disabled={!selectedPolygon || Boolean(busy)} onClick={deleteSelectedPolygon} title="Delete polygon">
              <X size={16} />
            </button>
            <button className="primary-button" type="button" disabled={!selectedImage || Boolean(busy)} onClick={saveLabels}>
              {busy === "saving" ? <Loader2 className="spin-icon" size={16} /> : <Save size={16} />}
              Save
            </button>
          </div>
        </div>
        <TrainingCanvas
          image={selectedImage}
          polygons={polygons}
          selectedPolygonId={selectedPolygonId}
          onSelectPolygon={setSelectedPolygonId}
          onChangePolygons={updatePolygons}
        />
        <div className="training-status-row">
          <div>
            <strong>{selectedImage?.name || "No image"}</strong>
            <span>{selectedImage ? `${selectedImage.width || "?"} x ${selectedImage.height || "?"} px` : ""}</span>
          </div>
          <div className="toolbar-row">
            <StatusBadge status={dirty ? "review_needed" : selectedImage?.has_label ? "completed" : "pending"} />
            <button className="icon-button" type="button" disabled={!selectedImage || busy === "deleting"} onClick={deleteSelectedImage} title="Delete image">
              <X size={16} />
            </button>
          </div>
        </div>
        <div className="annotation-help">
          Mark each physical print with one polygon. Drag its corners to the print edges; a magnifier appears while dragging for precise placement. Then click Save. Suggestions are only a starting point and are not saved until you save them.
        </div>
      </section>
      <section className="side-surface">
        <div className="section-title">
          <h2>Workspace</h2>
          <StatusBadge status={busy || "ready"} />
        </div>
        <div className="stack">
          <div className="annotation-progress" aria-label={`${completedCount} of ${images.length} pages saved`}>
            <div className="annotation-progress-heading">
              <strong>Progress</strong>
              <span>{completedCount}/{images.length} saved ({progressPercent}%)</span>
            </div>
            <div className="annotation-progress-track"><span style={{ width: `${progressPercent}%` }} /></div>
          </div>
          <button className="icon-button text-button" type="button" disabled={!selectedImage || Boolean(busy)} onClick={selectNextPending}>
            Next unlabelled page
          </button>
          <label className="file-pick icon-button text-button">
            <Upload size={16} />
            Upload Images
            <input type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={(event) => uploadImages(event.target.files)} />
          </label>
          <button className="icon-button text-button" type="button" disabled={busy === "preparing"} onClick={prepareDataset}>
            {busy === "preparing" ? <Loader2 className="spin-icon" size={16} /> : <Check size={16} />}
            Prepare Dataset
          </button>
          {message && <p className="run-message">{message}</p>}
        </div>
        <div className="training-image-list">
          {images.map((image) => (
            <button
              type="button"
              key={image.name}
              className={`training-image-row ${selectedImage?.name === image.name ? "selected" : ""}`}
              onClick={() => setSelectedName(image.name)}
            >
              <img src={trainingImageSrc(image)} alt="" />
              <span>
                <strong>{image.name}</strong>
                <small>{image.labels} labels</small>
              </span>
              <StatusBadge status={image.has_label ? "completed" : "pending"} />
            </button>
          ))}
          {images.length === 0 && <EmptyState label="No training images." />}
        </div>
        <div className="training-polygon-list">
          <div className="section-title compact-title">
            <h2>Polygons</h2>
            <span className="muted-text">{polygons.length}</span>
          </div>
          {polygons.map((polygon, index) => (
            <button
              type="button"
              key={polygon.id}
              className={`polygon-row ${polygon.id === selectedPolygonId ? "selected" : ""}`}
              onClick={() => setSelectedPolygonId(polygon.id)}
            >
              <span>{index + 1}</span>
              <strong>{polygon.source || "manual"}</strong>
              <small>{polygon.points.length} pts</small>
            </button>
          ))}
          {polygons.length === 0 && <EmptyState label="No polygons." />}
        </div>
      </section>
    </div>
  );
}

const printAdditionErrors = {
  mounted_print_fragments_stale: "Die Teilvorschläge haben sich geändert. Prüfe die aktuelle Seite erneut.",
  source_print_fragments_stale: sourceFragmentReviewErrors.source_print_fragments_stale,
  dark_backing_confirmation_stale: darkBackingReviewErrors.dark_backing_confirmation_stale,
  album_pocket_print_confirmation_stale: embossedPocketReviewErrors.album_pocket_print_confirmation_stale,
  standalone_print_confirmation_stale: standalonePrintReviewErrors.standalone_print_confirmation_stale,
  adjacent_print_suggestion_saved: "Dieser Abzug wurde bereits gespeichert. Öffne ihn in der aktualisierten Ansicht, wenn du seine Ecken ändern möchtest.",
  adjacent_print_suggestion_stale: "Der Vorschlag wurde geändert. Prüfe die Albumseite in der aktualisierten Ansicht erneut.",
  deferred_print_proposal_saved: "Dieser Abzug wurde bereits ergänzt. Du kannst seine Ecken in der aktualisierten Ansicht ändern.",
  deferred_print_proposal_stale: "Dieser Vorschlag wurde bereits geprüft. Die Ansicht wurde aktualisiert.",
};

function ReviewCanvas({ page, photos, onSaveBox, onAddPhoto, onRejectPhoto, onConfirmCount, onResolveCountGroup, onReprocess,
  selectedPhotoId, onSelectPhoto, reviewingCount, reviewingGeometry }) {
  const canvasRef = useRef(null);
  const [measurementPage, setMeasurementPage] = useState(null);
  const [measuredEdge, setMeasuredEdge] = useState(null);
  const [showMissingEdges, setShowMissingEdges] = useState(false);
  const [pageImage, setPageImage] = useState(null);
  const [draftShapes, setDraftShapes] = useState({});
  const [confirmedContrastOutline, setConfirmedContrastOutline] = useState(null);
  const [editMode, setEditMode] = useState("full_print");
  const [saveState, setSaveState] = useState(null);
  const [blockedCrop, setBlockedCrop] = useState(null);
  const [addition, setAddition] = useState(null);
  const [countReview, setCountReview] = useState(null);
  const [groupSelection, setGroupSelection] = useState(null);
  const dragRef = useRef(null);
  const editingImage = editMode === "image_only";
  const blockedRegions = blockedCrop?.photoId === selectedPhotoId && blockedCrop.mode === editMode
    ? blockedCrop.regions : null;
  const pageId = page?.id;
  const measuredTraces = measuredEdge?.pageId === pageId ? measuredEdge?.traces : null;
  const segmentation = page?.processing_metadata?.steps?.segmentation;
  const pendingMissingTraces = segmentation?.missing_print_edge_traces;
  const nearCaptureTraces = segmentation?.near_capture_print_edge_traces;
  const missingTraces = useMemo(() => [
    ...getSourceEdgeTraces(pendingMissingTraces), ...getSourceEdgeTraces(nearCaptureTraces),
  ], [pendingMissingTraces, nearCaptureTraces]);
  const pendingProposals = page?.processing_metadata?.steps?.segmentation?.fusion?.deferred_print_proposals;
  const deferredProposals = useMemo(() => getDeferredPrintProposals(pendingProposals), [pendingProposals]);
  const activePage = useRef(pageId);
  const newPrint = addition?.pageId === pageId ? addition : null;
  const darkBackingDraft = newPrint?.deferredProposal?.reason === darkBackingProposal;
  const standaloneDraft = standalonePrintNeedsConfirmation(newPrint?.deferredProposal?.reason);
  const needsPrintAdditionConfirmation = standaloneDraft ? !newPrint.confirmStandalonePrint
    : newPrint?.deferredProposal?.reason === sourceFragmentProposal && !newPrint.replaceMountedFragments;
  const currentCountReview = countReview?.pageId === pageId ? countReview : null;
  const paperGridConfirmed = reviewingCount && currentCountReview?.paperGridEventId != null
    && currentCountReview.paperGridEventId === page?.processing_metadata?.photo_count_revision;
  const additionSize = newPrint && parsePrintSize(newPrint.printSizeFields);
  const canAddPhoto = pageImage?.pageId === pageId && ["completed", "review_needed"].includes(page?.status);
  const selected = photos.find((photo) => photo.id === selectedPhotoId);
  const selectedModes = selected?.crop_modes || getCropModes(selected);
  const contrastOutlineReview = contrastOutlineNeedsConfirmation(selected, selectedModes?.full_print);
  const refreshContactEdges = reviewingGeometry && contactEdgesNeedRefresh(selected, selectedModes?.full_print)
    && !newPrint && !editingImage && !draftShapes[selectedPhotoId] && !selected?.reviewed_non_print;
  const overlapReview = selected?.segmentation_confidence != null
    && selected.enhancement_applied?.segmentation_review_reasons?.includes(clippedOverlapReason);
  const scannerReview = selected?.segmentation_confidence != null
    && selected.enhancement_applied?.segmentation_review_reasons?.includes(scannerBackingReason);
  const filmReview = selected?.segmentation_confidence != null
    && selected.enhancement_applied?.segmentation_review_reasons?.includes(filmGridReason);
  const paperJoinReview = selected?.segmentation_confidence != null
    && selected.enhancement_applied?.segmentation_review_reasons?.includes(paperJoinReason);
  const fragmentReview = selected?.segmentation_confidence != null
    && selected.enhancement_applied?.segmentation_review_reasons?.includes(fragmentOutlineReason);
  const crossingReview = selected?.segmentation_confidence != null
    && selected.enhancement_applied?.segmentation_review_reasons?.includes(crossingPrintReason);
  const rejectedDetection = Boolean(selected?.reviewed_non_print);
  const filmMask = filmReview && !rejectedDetection ? selected.segmentation_mask : null;
  const filmGridSeam = useMemo(() => getFilmGridSeam(filmMask), [filmMask]);
  const filmGridGroups = useMemo(() => reviewingCount && !newPrint && !editingImage ? getFilmGridReviewSeams(photos) : [],
    [reviewingCount, newPrint, editingImage, photos]);
  const paperJoinGroups = useMemo(() => reviewingCount && !newPrint && !editingImage ? getPaperJoinReviewGroups(photos) : [],
    [reviewingCount, newPrint, editingImage, photos]);
  const paperGridGroups = useMemo(() => reviewingCount && !newPrint && !editingImage ? getPaperGridReviewGroups(photos) : [],
    [reviewingCount, newPrint, editingImage, photos]);
  const paperGridOutlines = useMemo(() => reviewingCount && !newPrint && !editingImage ? getPaperGridOutlineParents(photos)
    .flatMap(photo => {
      const quad = cyclicQuad(sanitizePolygon((photo.crop_modes || getCropModes(photo))?.full_print?.suggested_warp_quad));
      return quad ? [{ photo, quad }] : [];
    }) : [], [reviewingCount, newPrint, editingImage, photos]);
  const countGroups = useMemo(() => actionableCountGroups(getActionableFilmGridReviewGroups(filmGridGroups, photos), paperJoinGroups,
    getActionablePaperGridReviewGroups(paperGridGroups, photos),
    reviewingCount && !newPrint && !editingImage ? [...getCrossingFallbackReviewGroups(photos), ...getAlbumLeafReviewGroups(photos)] : []),
  [filmGridGroups, paperJoinGroups, paperGridGroups, photos, reviewingCount, newPrint, editingImage]);
  const selectedCountGroup = groupSelection?.pageId === pageId
    ? countGroups.find(({ key }) => key === groupSelection?.key) : null;
  const competingPhotoOutlines = useMemo(() => {
    if (newPrint || editingImage || selectedCountGroup || (!reviewingCount && !reviewingGeometry)) return [];
    const comparisons = getCompetingPhotoReviewOutlines(photos);
    return reviewingCount ? comparisons : comparisons.filter(outline => (
      outline.wider_photo_id === selectedPhotoId || outline.narrower_photo_id === selectedPhotoId
    ));
  }, [newPrint, editingImage, selectedCountGroup, reviewingCount, reviewingGeometry, photos, selectedPhotoId]);
  const paperGridMask = selected?.segmentation_confidence != null && !rejectedDetection
    && selected.enhancement_applied?.segmentation_review_reasons?.includes(paperGridReason) ? selected.segmentation_mask : null;
  const paperGrid = useMemo(() => getPaperGridReview(paperGridMask), [paperGridMask]);
  const captureSplitGroups = useMemo(() => reviewingCount && !newPrint && !editingImage ? getCaptureSplitReviewGroups(photos) : [],
    [reviewingCount, newPrint, editingImage, photos]);
  const mergedPairs = useMemo(() => reviewingCount && !newPrint && !editingImage ? getMergedPrintPairs(photos) : [],
    [reviewingCount, newPrint, editingImage, photos]);
  const captureSplitReview = selected?.segmentation_confidence != null && !rejectedDetection
    && selected.segmentation_mask?.capture_split_review_traces?.length === 2;
  const deferredParentHints = rejectedDetection ? []
    : getDeferredParentCropHints(selected?.enhancement_applied?.segmentation_review_reasons);
  const detectionReviewHint = (crossingReview && crossingPrintHint) || (scannerReview && scannerBackingHint)
    || (overlapReview && clippedOverlapHint) || (filmGridSeam && filmGridHint) || (paperJoinReview && paperJoinHint)
    || (paperGrid && (paperGrid.cut ? paperGridFragmentHint : paperGridOuterHint)) || (fragmentReview && fragmentOutlineHint)
    || (captureSplitReview && "Prüfe, ob dieser Vorschlag einen eigenen Papierabzug zeigt.")
    || (contrastOutlineReview && contrastOutlineReviewHint);
  const fragmentMask = fragmentReview && !rejectedDetection ? selected.segmentation_mask : null;
  const fragmentOutline = useMemo(() => getFragmentOutline(fragmentMask), [fragmentMask]);
  const suggestedNeighbor = selected?.segmentation_mask?.adjacent_print_suggestion;
  const suggestedNested = selected?.segmentation_mask?.nested_print_suggestion;
  const pendingNested = photos.filter((photo) => !photo.reviewed_non_print && photo.segmentation_mask?.nested_print_suggestion).length;
  const alternatePolygon = selected?.segmentation_confidence != null && !rejectedDetection
    ? suggestedNested || suggestedNeighbor || (selected.segmentation_mask?.scalloped_alternate_is_inner
      && selected.segmentation_mask?.scalloped_alternate_polygon)
    : null;
  const alternateOutline = useMemo(() => cyclicQuad(sanitizePolygon(alternatePolygon)), [alternatePolygon]);
  const crossingPolygon = crossingReview && !rejectedDetection ? selected.segmentation_mask?.crossing_print_suggestion : null;
  const crossingOutline = useMemo(() => cyclicQuad(sanitizePolygon(crossingPolygon)), [crossingPolygon]);
  const savedShape = useMemo(() => selected && draftShapeFromPhoto(selected), [selected]);
  const selectedShape = selected && (draftShapes[selected.id] || savedShape);
  const selectedMergedPair = selected && !rejectedDetection && getMergedPrintPairs([selected])[0];
  const fragmentGroups = useMemo(() => reviewingCount ? getFragmentReviewOutlines(photos) : [], [reviewingCount, photos]);
  const countOutlines = useMemo(() => !newPrint && !editingImage ? fragmentGroups.filter(
    ({ photoIds }) => !selectedShape?.usingOutlineSuggestion || !photoIds.has(selectedPhotoId),
  ) : [], [newPrint, editingImage, fragmentGroups, selectedShape?.usingOutlineSuggestion, selectedPhotoId]);
  const crossingNeedsCorrection = crossingReview && (editingImage || savedShape.warpQuad.every(
    (point, index) => point.x === selectedShape.warpQuad[index].x && point.y === selectedShape.warpQuad[index].y,
  ));
  const contrastOutlineConfirmed = confirmedContrastOutline?.photoId === selected?.id
    && confirmedContrastOutline?.quad === selectedShape?.warpQuad;
  const contrastOutlineNeedsCheck = contrastOutlineReview && !contrastOutlineConfirmed;
  const fullPrintMode = rejectedDetection ? null : selectedModes?.full_print;
  const contrastCrossingTraces = useMemo(() => reviewingGeometry && !newPrint && !editingImage
    ? getContrastCrossingTraces(selected, savedShape?.warpQuad, selectedShape?.warpQuad) : [],
  [reviewingGeometry, newPrint, editingImage, selected, savedShape?.warpQuad, selectedShape?.warpQuad]);
  const sourceStrip = useMemo(() => reviewingGeometry && !newPrint && !editingImage ? getSourceStripReview(fullPrintMode) : null,
    [reviewingGeometry, newPrint, editingImage, fullPrintMode]);
  const printEdgeReviewHint = getPrintEdgeReviewHint(fullPrintMode);
  const proposedPrintQuad = fullPrintMode?.suggested_warp_quad;
  const printSuggestion = useMemo(() => cyclicQuad(sanitizePolygon(proposedPrintQuad)), [proposedPrintQuad]);
  const confirmOuter = contrastOutlineReview ? contrastOutlineConfirmed
    : Boolean(printSuggestion) || paperOutlineNeedsConfirmation(fullPrintMode);
  const needsPrintReview = editingImage && (contrastOutlineReview || confirmOuter);
  const motifReview = selectedModes?.image_only;
  const motifBoundaryReviewHint = getMotifBoundaryReviewHint(motifReview);
  const motifMarginMarks = useMemo(() => getMotifMarginMarks(motifReview), [motifReview]);
  const preliminaryMotifMarks = !newPrint && !rejectedDetection && motifReview?.margin_mark_source === heldPrintMotifMargin
    && motifMarginMarks.length > 0;
  const imageSuggestion = cyclicQuad(sanitizePolygon(motifReview?.suggested_warp_quad));
  const seamTraces = motifReview?.suggested_seam_traces;
  const traces = editingImage ? seamTraces : fullPrintMode?.suggested_edge_traces;
  const neighborOutline = editingImage ? null : fullPrintMode?.neighbor_overlap_outline;
  const missingPaperOutlines = editingImage ? null : fullPrintMode?.neighbor_content_loss_outlines;
  const exposedOverlapEdges = editingImage ? null : fullPrintMode?.exposed_overlap_edge_traces;
  const extentEdges = editingImage ? null : fullPrintMode?.partial_print_extent_traces;
  const edgeTraces = useMemo(() => [...getSourceEdgeTraces(traces), ...getSourceEdgeTraces([neighborOutline]),
    ...getSourceEdgeTraces(missingPaperOutlines)], [traces, neighborOutline, missingPaperOutlines]);
  const filmEdgeIndex = editingImage ? null : getFilmNeighborEdgeIndex(fullPrintMode);
  const motifPreview = selected?.variant_urls?.image_only_preview || selected?.urls?.image_only_preview;
  const paperPreviewUrl = selected?.variant_urls?.full_print_preview || selected?.urls?.full_print_preview;
  const untrimmedPrint = selected?.untrimmed_url || selected?.variant_urls?.source_corrected_untrimmed
    || selected?.urls?.source_corrected_untrimmed;
  const ribbedReviewOutline = reviewingGeometry && !newPrint && !editingImage && untrimmedPrint
    && selectedModes?.image_only?.status === "review_needed"
    ? getRibbedPrintReviewOutline(fullPrintMode, paperPreviewUrl, savedShape?.warpQuad, selectedShape?.warpQuad) : null;
  const fullPrintPreview = fullPrintMode?.outline_preview?.source === measuredRibbedPrintOutlineSource && !ribbedReviewOutline
    ? null : paperPreviewUrl;
  const comparePaperEdge = untrimmedPrint && showsPaperEdgeComparison(fullPrintMode);
  const albumMark = !newPrint && !editingImage && !rejectedDetection ? selected?.correction?.album_page_mark_review : null;
  const albumMarkReview = useAlbumPageMarkReview(albumMark, selectedPhotoId);
  const markBounds = albumMarkReview.bounds;
  const outlinePreviewHint = getPrintOutlinePreviewHint(fullPrintMode);
  const motifCorners = selectedShape?.imageQuad || savedShape?.warpQuad;
  const motifPaperQuad = editingImage && fullPrintMode?.status === "ready" && savedShape
    && motifUsesPaperAxes(savedShape.warpQuad, motifCorners) ? savedShape.warpQuad : null;
  const needsMotifSelection = editingImage && !selectedShape?.imageQuad;
  const measurement = selectedShape && parsePrintSize(selectedShape.printSizeFields);
  const sizeError = editingImage ? "" : measurement?.error;

  useEffect(() => {
    activePage.current = pageId;
    setMeasurementPage(null);
    setMeasuredEdge(null);
    setShowMissingEdges(false);
  }, [pageId]);

  useEffect(() => {
    setDraftShapes(Object.fromEntries(photos.map((photo) => [photo.id, draftShapeFromPhoto(photo)])));
  }, [photos]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !pageImage || pageImage.pageId !== pageId) return;
    const { image } = pageImage;
    const context = canvas.getContext("2d");
    const maxWidth = canvas.parentElement?.clientWidth || 720;
    const ratio = image.naturalHeight / image.naturalWidth || 0.75;
    canvas.width = maxWidth;
    canvas.height = Math.round(maxWidth * ratio);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    for (const outline of competingPhotoOutlines) {
      for (const { kind, path, clip, rule } of competingPhotoRegionPaths(outline, canvas.width, canvas.height)) {
        context.save();
        context.clip(new Path2D(clip));
        context.fillStyle = competingPhotoRegionStyles[kind].fill;
        context.fill(new Path2D(path), rule);
        context.restore();
      }
    }
    photos.forEach((photo) => {
      const inCountGroup = selectedCountGroup?.photoIds.has(photo.id);
      const shape = (!inCountGroup && draftShapes[photo.id]) || draftShapeFromPhoto(photo);
      const visiblePoints = shape.visiblePolygon.map((point) => ({ x: point.x * canvas.width, y: point.y * canvas.height }));
      const outer = editingImage ? draftShapeFromPhoto(photo).warpQuad : shape.warpQuad;
      const points = outer.map((point) => ({ x: point.x * canvas.width, y: point.y * canvas.height }));
      if (visiblePoints.length >= 4) {
        context.strokeStyle = inCountGroup ? "#0e7490" : "rgba(245, 158, 11, 0.9)";
        context.lineWidth = inCountGroup ? 3 : 2;
        context.beginPath();
        visiblePoints.forEach((point, index) => {
          if (index === 0) context.moveTo(point.x, point.y);
          else context.lineTo(point.x, point.y);
        });
        context.closePath();
        if (inCountGroup) {
          context.fillStyle = "rgba(14, 116, 144, 0.11)";
          context.fill();
        }
        context.stroke();
      }
      context.strokeStyle = inCountGroup ? "#0e7490" : photo.id === selectedPhotoId ? "#0f766e" : "#f59e0b";
      context.lineWidth = photo.id === selectedPhotoId ? 4 : 2;
      context.beginPath();
      points.forEach((point, index) => {
        if (index === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      });
      context.closePath();
      context.stroke();
      if (photo.id === selectedPhotoId && !editingImage) {
        context.fillStyle = "rgba(15, 118, 110, 0.1)";
        context.fill();
        if (fragmentOutline && shape.usingOutlineSuggestion) {
          const [start, end] = fragmentOutline.weakEdge;
          context.save();
          context.strokeStyle = "#a21caf";
          context.setLineDash([7, 5]);
          context.beginPath();
          context.moveTo(start.x * canvas.width, start.y * canvas.height);
          context.lineTo(end.x * canvas.width, end.y * canvas.height);
          context.stroke();
          context.restore();
        }
      }
      const dashedOutline = photo.id === selectedPhotoId
        ? !shape.usingOutlineSuggestion && (crossingOutline || printSuggestion || alternateOutline)
        : !photo.reviewed_non_print && sanitizePolygon(photo.segmentation_mask?.nested_print_suggestion);
      if (!editingImage && dashedOutline?.length === 4) {
        context.save();
        context.strokeStyle = "#a21caf";
        context.setLineDash([7, 5]);
        context.beginPath();
        dashedOutline.forEach(({ x, y }, index) => {
          if (index === 0) context.moveTo(x * canvas.width, y * canvas.height);
          else context.lineTo(x * canvas.width, y * canvas.height);
        });
        context.closePath();
        context.stroke();
        context.restore();
      }
      if (editingImage && photo.id !== selectedPhotoId) return;
      const handles = editingImage
        ? motifCorners.map((point) => ({ x: point.x * canvas.width, y: point.y * canvas.height }))
        : points;
      if (editingImage) {
        context.strokeStyle = "#2563eb";
        context.beginPath();
        context.moveTo(handles[0].x, handles[0].y);
        handles.slice(1).forEach((point) => context.lineTo(point.x, point.y));
        context.closePath();
        context.stroke();
      }
      if (photo.id === selectedPhotoId && ribbedReviewOutline) {
        const { color, contrast, dash } = ribbedPrintOutlineStyle;
        const [first, ...rest] = ribbedReviewOutline;
        context.save();
        context.beginPath();
        context.moveTo(first.x * canvas.width, first.y * canvas.height);
        rest.forEach(({ x, y }) => context.lineTo(x * canvas.width, y * canvas.height));
        context.closePath();
        context.strokeStyle = contrast;
        context.lineWidth = 4;
        context.stroke();
        context.strokeStyle = color;
        context.lineWidth = 2;
        context.setLineDash(dash);
        context.stroke();
        context.restore();
      }
      if (photo.id === selectedPhotoId && !newPrint && (edgeTraces.length || exposedOverlapEdges?.length || extentEdges?.length)) {
        context.save();
        context.setLineDash([5, 3]);
        const paperTraces = filmEdgeIndex === null ? edgeTraces : edgeTraces.filter((_, index) => index !== filmEdgeIndex);
        const filmTrace = filmEdgeIndex === null ? [] : [edgeTraces[filmEdgeIndex]];
        for (const [lines, color, strokeWidth] of [[paperTraces, "#a21caf", 2], [filmTrace, filmNeighborEdgeColor, 2],
          [getSourceEdgeTraces(exposedOverlapEdges), "#0e7490", 2], [getSourceEdgeTraces(extentEdges), "#0891b2", 1]]) {
          context.strokeStyle = color;
          context.lineWidth = strokeWidth;
          for (const [first, ...rest] of lines) {
            context.beginPath();
            context.moveTo(first.x * canvas.width, first.y * canvas.height);
            rest.forEach(({ x, y }) => context.lineTo(x * canvas.width, y * canvas.height));
            context.stroke();
          }
        }
        context.restore();
      }
      if (photo.id === selectedPhotoId && contrastCrossingTraces.length) {
        const { color, contrast, dash } = contrastCrossingTraceStyle;
        context.save();
        context.beginPath();
        for (const [start, end] of contrastCrossingTraces) {
          context.moveTo(start.x * canvas.width, start.y * canvas.height);
          context.lineTo(end.x * canvas.width, end.y * canvas.height);
        }
        context.strokeStyle = contrast;
        context.lineWidth = 4;
        context.stroke();
        context.strokeStyle = color;
        context.lineWidth = 2;
        context.setLineDash(dash);
        context.stroke();
        context.restore();
      }
      if (photo.id === selectedPhotoId && sourceStrip) {
        const [first, ...rest] = sourceStrip.outline;
        context.save();
        context.lineJoin = "round";
        context.beginPath();
        context.moveTo(first.x * canvas.width, first.y * canvas.height);
        rest.forEach(({ x, y }) => context.lineTo(x * canvas.width, y * canvas.height));
        if (!sourceStrip.edgeOnly) context.closePath();
        context.strokeStyle = sourceStripContrastColor;
        context.lineWidth = 4;
        context.setLineDash([]);
        context.stroke();
        context.strokeStyle = sourceStripColor;
        context.lineWidth = 2;
        context.setLineDash([7, 4]);
        context.stroke();
        context.restore();
      }
      if ((editingImage || preliminaryMotifMarks) && photo.id === selectedPhotoId && motifMarginMarks.length) {
        context.save();
        context.strokeStyle = "#b45309";
        context.lineWidth = 2;
        for (const [first, ...region] of motifMarginMarks) {
          context.beginPath();
          context.moveTo(first.x * canvas.width, first.y * canvas.height);
          region.forEach(({ x, y }) => context.lineTo(x * canvas.width, y * canvas.height));
          context.closePath();
          context.stroke();
        }
        context.restore();
      }
      context.fillStyle = "#ffffff";
      handles.forEach((point, index) => {
        context.fillRect(point.x - 5, point.y - 5, 10, 10);
        context.strokeRect(point.x - 5, point.y - 5, 10, 10);
        if (!editingImage && photo.id === selectedPhotoId) {
          context.save();
          const x = point.x > canvas.width - 24 ? point.x - 13 : point.x + 13;
          const y = point.y < 16 ? point.y + 14 : point.y - 9;
          context.font = "bold 14px Inter, sans-serif";
          context.textAlign = "center";
          context.textBaseline = "middle";
          context.strokeStyle = "#ffffff";
          context.lineWidth = 4;
          context.strokeText(String(index + 1), x, y);
          context.fillStyle = "#134e4a";
          context.fillText(String(index + 1), x, y);
          context.restore();
        }
      });
    });
    const visibleTraces = showMissingEdges ? [...missingTraces, ...(measuredTraces || [])] : measuredTraces;
    if (visibleTraces?.length && !editingImage) {
      context.save();
      context.strokeStyle = "#2563eb";
      context.lineWidth = 3;
      context.setLineDash([5, 4]);
      for (const trace of visibleTraces) {
        context.beginPath();
        context.moveTo(trace[0].x * canvas.width, trace[0].y * canvas.height);
        trace.slice(1).forEach(({ x, y }) => context.lineTo(x * canvas.width, y * canvas.height));
        context.stroke();
      }
      context.restore();
    }
    if (!newPrint && !editingImage && deferredProposals.length > 0) {
      context.save();
      context.strokeStyle = "#7e22ce";
      context.fillStyle = "#7e22ce";
      context.lineWidth = 3;
      context.font = "bold 14px sans-serif";
      deferredProposals.forEach(({ visible_polygon: polygon, uncertain_edge: uncertainEdge, reason }, index) => {
        const corners = polygon.map(({ x, y }) => [x * canvas.width, y * canvas.height]);
        const [[left, top]] = corners;
        corners.forEach((corner, side) => {
          context.setLineDash(side === uncertainEdge ? [2, 5] : [9, 5]);
          context.beginPath();
          context.moveTo(...corner);
          context.lineTo(...corners[(side + 1) % corners.length]);
          context.stroke();
        });
        const label = `?${index + 1}`;
        const x = Math.min(left + 6, canvas.width - 28);
        const y = Math.max(top - 6, 16);
        if (standalonePrintNeedsConfirmation(reason)) {
          context.save();
          context.setLineDash([]);
          context.lineWidth = 4;
          context.strokeText(label, x, y);
          context.fillStyle = "#fff";
          context.fillText(label, x, y);
          context.restore();
        } else {
          context.fillText(label, x, y);
        }
      });
      context.restore();
    }
    if (countOutlines.length || captureSplitGroups.length || mergedPairs.length) {
      context.save();
      context.strokeStyle = "#2563eb";
      context.lineWidth = 3;
      const lines = countOutlines.flatMap(({ supportedSides, weakEdge }) => [[supportedSides, [9, 5]], [weakEdge, [2, 5]]]);
      lines.push(...captureSplitGroups.flatMap(({ traces }) => traces.map(trace => [trace, [9, 5]])));
      lines.push(...mergedPairs.flatMap(({ outlines, separator }) => [
        ...outlines.map(outline => [[...outline, outline[0]], [8, 5]]),
        [separator, [2, 5]],
      ]));
      for (const [line, dashes] of lines) {
        context.setLineDash(dashes);
        context.beginPath();
        context.moveTo(line[0].x * canvas.width, line[0].y * canvas.height);
        line.slice(1).forEach(({ x, y }) => context.lineTo(x * canvas.width, y * canvas.height));
        context.stroke();
      }
      context.restore();
    }
    if (!newPrint && !editingImage && (filmGridGroups.length || filmGridSeam || paperJoinGroups.length)) {
      const lines = filmGridGroups.map(({ line }) => line);
      if (!lines.length && filmGridSeam) lines.push(filmGridSeam);
      lines.push(...paperJoinGroups.flatMap(({ traces }) => traces));
      context.save();
      context.strokeStyle = "#0e7490";
      context.lineWidth = 3;
      context.setLineDash([8, 5]);
      for (const [start, end] of lines) {
        context.beginPath();
        context.moveTo(start.x * canvas.width, start.y * canvas.height);
        context.lineTo(end.x * canvas.width, end.y * canvas.height);
        context.stroke();
      }
      context.setLineDash([2, 6]);
      for (const { rim } of filmGridGroups) {
        if (!rim) continue;
        const [start, end] = rim;
        context.beginPath();
        context.moveTo(start.x * canvas.width, start.y * canvas.height);
        context.lineTo(end.x * canvas.width, end.y * canvas.height);
        context.stroke();
      }
      context.restore();
    }
    if (selectedCountGroup?.kind === "crossing" || selectedCountGroup?.kind === "album_leaf") {
      const isAlbumLeaf = selectedCountGroup.kind === "album_leaf";
      const evidence = selectedCountGroup.traces.map((points, index) => ({
        points, ...(isAlbumLeaf ? albumLeafTraceStyle : crossingFallbackTraceStyles[index]),
      }));
      if (isAlbumLeaf) evidence.push({ points: selectedCountGroup.child, closed: true, ...albumLeafChildStyle });
      context.save();
      for (const { points: [first, ...rest], color, contrast, dash, closed } of evidence) {
        context.beginPath();
        context.moveTo(first.x * canvas.width, first.y * canvas.height);
        rest.forEach(({ x, y }) => context.lineTo(x * canvas.width, y * canvas.height));
        if (closed) context.closePath();
        if (contrast) {
          context.strokeStyle = contrast;
          context.lineWidth = 5;
          context.setLineDash([]);
          context.stroke();
        }
        context.strokeStyle = color;
        context.lineWidth = 3;
        context.setLineDash(dash);
        context.stroke();
      }
      context.restore();
    }
    if (!newPrint && !editingImage && (paperGridGroups.length || paperGrid?.cut)) {
      const groups = selectedCountGroup?.kind === "paper-grid" ? [selectedCountGroup] : paperGridGroups.length
        ? paperGridGroups : [{ outline: paperGrid.outline, cuts: new Map([[selectedPhotoId, paperGrid.cut]]) }];
      context.save();
      context.strokeStyle = "#1d4ed8";
      context.lineWidth = 3;
      for (const { outline, cuts } of groups) {
        context.setLineDash([9, 5]);
        context.beginPath();
        context.moveTo(outline[0].x * canvas.width, outline[0].y * canvas.height);
        outline.slice(1).forEach(({ x, y }) => context.lineTo(x * canvas.width, y * canvas.height));
        context.closePath();
        context.stroke();
        context.setLineDash([2, 5]);
        for (const [start, end] of cuts.values()) {
          context.beginPath();
          context.moveTo(start.x * canvas.width, start.y * canvas.height);
          context.lineTo(end.x * canvas.width, end.y * canvas.height);
          context.stroke();
        }
      }
      context.restore();
    }
    if (markBounds.length) {
      context.save();
      context.strokeStyle = "#facc15";
      context.lineWidth = 3;
      context.setLineDash([]);
      for (const { x1, y1, x2, y2 } of markBounds) {
        context.strokeRect(x1 * canvas.width, y1 * canvas.height, (x2 - x1) * canvas.width, (y2 - y1) * canvas.height);
      }
      context.restore();
    }
    if (blockedRegions?.length && !newPrint) {
      context.save();
      context.setLineDash([]);
      context.strokeStyle = "#b91c1c";
      context.lineWidth = 4;
      for (const { x1, y1, x2, y2 } of blockedRegions) {
        context.strokeRect(x1 * canvas.width, y1 * canvas.height, (x2 - x1) * canvas.width, (y2 - y1) * canvas.height);
      }
      context.restore();
    }
    if (newPrint?.corners.length) {
      if (newPrint.deferredProposal?.fragmentPolygons?.length) {
        context.save();
        context.strokeStyle = "#a21caf";
        context.fillStyle = "rgba(162, 28, 175, 0.12)";
        context.lineWidth = 2;
        context.setLineDash([5, 3]);
        for (const [first, ...points] of newPrint.deferredProposal.fragmentPolygons) {
          context.beginPath();
          context.moveTo(first.x * canvas.width, first.y * canvas.height);
          points.forEach(({ x, y }) => context.lineTo(x * canvas.width, y * canvas.height));
          context.closePath();
          context.fill();
          context.stroke();
        }
        context.restore();
      }
      const points = newPrint.corners.map(({ x, y }) => [x * canvas.width, y * canvas.height]);
      context.strokeStyle = "#2563eb";
      context.fillStyle = "#2563eb";
      context.lineWidth = 3;
      context.beginPath();
      context.moveTo(...points[0]);
      points.slice(1).forEach((point) => context.lineTo(...point));
      if (points.length === 4) context.closePath();
      context.stroke();
      points.forEach(([x, y], index) => {
        context.beginPath();
        context.arc(x, y, 6, 0, 2 * Math.PI);
        context.fill();
        context.font = "bold 14px sans-serif";
        context.fillText(String(index + 1), x + 9, y - 7);
      });
    }
    if (!editingImage) {
      const offers = newPrint ? [newPrint.deferredProposal] : deferredProposals;
      context.save();
      context.lineWidth = 3;
      for (const { color, dash, traces } of offers.flatMap(getDeferredPrintEvidence)) {
        context.strokeStyle = color;
        context.setLineDash(dash);
        for (const [first, last] of traces) {
          context.beginPath();
          context.moveTo(first.x * canvas.width, first.y * canvas.height);
          context.lineTo(last.x * canvas.width, last.y * canvas.height);
          context.stroke();
        }
      }
      context.restore();
    }
  }, [alternateOutline, blockedRegions, captureSplitGroups, competingPhotoOutlines, contrastCrossingTraces, countOutlines, crossingOutline, deferredProposals, draftShapes, edgeTraces, editingImage, exposedOverlapEdges, extentEdges, mergedPairs,
    filmEdgeIndex, filmGridGroups, filmGridSeam, fragmentOutline, measuredTraces, paperGrid, paperGridGroups, paperJoinGroups,
    printSuggestion, ribbedReviewOutline,
    markBounds, missingTraces, motifCorners, motifMarginMarks, newPrint, pageId, pageImage, photos, preliminaryMotifMarks, selectedCountGroup, selectedPhotoId, showMissingEdges, sourceStrip]);

  useEffect(() => {
    draw();
    window.addEventListener("resize", draw);
    return () => window.removeEventListener("resize", draw);
  }, [draw]);

  useEffect(() => {
    if (!pageId) return;
    const image = new Image();
    image.onload = () => setPageImage({ pageId, image });
    image.src = `/api/v1/pages/${pageId}/image`;
    return () => { image.onload = null; };
  }, [pageId]);

  function pointToBox(event) {
    const rect = canvasRef.current.getBoundingClientRect();
    return { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height };
  }

  function onPointerDown(event) {
    if (saveState?.saving) return;
    const point = pointToBox(event);
    if (newPrint) {
      if (newPrint.saving || newPrint.invalidated) return;
      if (newPrint.corners.length === 4) {
        const pointIndex = newPrint.corners.findIndex((corner) => Math.hypot(point.x - corner.x, point.y - corner.y) < 0.025);
        if (pointIndex >= 0) {
          dragRef.current = { addition: true, pointIndex };
          canvasRef.current.setPointerCapture(event.pointerId);
        }
        return;
      }
      const corners = [...newPrint.corners, { x: clampUnit(point.x), y: clampUnit(point.y) }];
      setAddition({ ...newPrint, corners: cyclicQuad(corners) || corners, error: null });
      return;
    }
    const targets = editingImage && selected
      ? [selected, ...photos.filter((photo) => photo.id !== selectedPhotoId).reverse()]
      : [...photos].reverse();
    for (const photo of targets) {
      const shape = draftShapes[photo.id] || draftShapeFromPhoto(photo);
      const corners = editingImage
        ? (photo.id === selectedPhotoId ? motifCorners : [])
        : shape.warpQuad;
      const cornerIndex = corners.findIndex((corner) => Math.hypot(point.x - corner.x, point.y - corner.y) < 0.025);
      if (cornerIndex >= 0) {
        onSelectPhoto(photo.id);
        dragRef.current = { photoId: photo.id, pointIndex: cornerIndex };
        if (editingImage) {
          const { width, height } = canvasRef.current.getBoundingClientRect();
          Object.assign(dragRef.current, {
            corner: corners[cornerIndex], clientX: event.clientX, clientY: event.clientY, width, height,
            paperQuad: motifPaperQuad, motifQuad: corners,
          });
        }
        setSaveState(null);
        canvasRef.current.setPointerCapture(event.pointerId);
        return;
      }
      if (pointInPolygon(point, shape.visiblePolygon)) {
        onSelectPhoto(photo.id);
        return;
      }
    }
  }

  function onPointerMove(event) {
    if (!dragRef.current) return;
    const point = pointToBox(event);
    const drag = dragRef.current;
    const { addition: draggingAddition, photoId, pointIndex } = drag;
    if (draggingAddition) {
      setAddition((current) => current && {
        ...current,
        corners: current.corners.map((corner, index) =>
          index === pointIndex ? { x: clampUnit(point.x), y: clampUnit(point.y) } : corner),
        error: null,
      });
      return;
    }
    if (Boolean(drag.corner) !== editingImage) return;
    let nextPoint = { x: clampUnit(point.x), y: clampUnit(point.y) };
    if (drag.corner) {
      const dx = event.clientX - drag.clientX;
      const dy = event.clientY - drag.clientY;
      if (!drag.moved && Math.hypot(dx, dy) < 2) return;
      drag.moved = true;
      nextPoint = { x: clampUnit(drag.corner.x + dx / drag.width), y: clampUnit(drag.corner.y + dy / drag.height) };
    }
    const resizedMotif = drag.paperQuad && resizeMotifInPaperPlane(drag.paperQuad, drag.motifQuad, pointIndex, nextPoint);
    if (drag.paperQuad && !resizedMotif) return;
    setDraftShapes((current) => {
      const photo = photos.find((candidate) => candidate.id === photoId);
      if (!photo) return current;
      const saved = draftShapeFromPhoto(photo);
      const currentShape = current[photoId] || saved;
      const corners = editingImage ? currentShape.imageQuad || saved.warpQuad : currentShape.warpQuad;
      const nextQuad = resizedMotif || corners.map((corner, index) =>
        index === pointIndex ? nextPoint : corner
      );
      const unchangedPaperCorners = editingImage && !saved.imageQuad && nextQuad.every(
        (corner, index) => Math.hypot(corner.x - saved.warpQuad[index].x, corner.y - saved.warpQuad[index].y) < 1e-8,
      );
      const shape = editingImage ? { ...currentShape, imageQuad: unchangedPaperCorners ? null : nextQuad } : {
        ...currentShape,
        warpQuad: nextQuad,
        visiblePolygon: currentShape.visibleFollowsWarp ? nextQuad : currentShape.visiblePolygon,
        boundingBox: boundingBoxFromPolygon(nextQuad),
      };
      return { ...current, [photoId]: shape };
    });
  }

  function onPointerUp(event) {
    if (event.type === "pointerup" && dragRef.current?.corner) onPointerMove(event);
    if (dragRef.current && canvasRef.current.hasPointerCapture(event.pointerId)) {
      canvasRef.current.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
  }

  function loadPrintOutline(photo, quad) {
    if (!photo || !quad) return;
    onSelectPhoto(photo.id);
    setEditMode("full_print");
    setDraftShapes((current) => ({ ...current, [photo.id]: {
      ...(current[photo.id] || draftShapeFromPhoto(photo)),
      warpQuad: quad, visiblePolygon: quad, visibleFollowsWarp: true, confirmMergedPrintAsOne: false,
      boundingBox: boundingBoxFromPolygon(quad), imageQuad: null, usingOutlineSuggestion: true,
    } }));
    setSaveState(null);
  }

  async function saveSelected() {
    if (!selected || sizeError || needsMotifSelection || crossingNeedsCorrection || needsPrintReview || contrastOutlineNeedsCheck) return;
    const box = editingImage ? selected.bounding_box : selectedShape.boundingBox;
    const mask = editingImage
      ? maskFromShape(savedShape, selectedShape.imageQuad)
      : maskFromShape(selectedShape);
    if (selectedShape.orientationDegrees !== savedShape.orientationDegrees) {
      mask.orientation_degrees = selectedShape.orientationDegrees;
    }
    if (!editingImage) mask.print_size_mm = measurement.size;
    else {
      const savedMeasurement = selected.segmentation_mask?.print_size_mm ?? selected.print_size_mm;
      if (savedMeasurement) mask.print_size_mm = savedMeasurement;
    }
    setSaveState({ photoId: selected.id, mode: editMode, saving: true });
    try {
      const mark = albumMarkReview.confirmation;
      const result = await onSaveBox(selected.id, box, mask, confirmOuter && !editingImage, mark?.id, mark?.indices,
        Boolean(selectedMergedPair && selectedShape.confirmMergedPrintAsOne && !editingImage));
      albumMarkReview.reset();
      setBlockedCrop(null);
      const url = result.urls?.[editMode] || (!getCropModes(result) && !editingImage ? result.urls?.original : null);
      setSaveState({ photoId: selected.id, mode: editMode, url });
    } catch (error) {
      if (error.message === "album_page_mark_review_stale") albumMarkReview.reset();
      const message = error.message === "crossing_print_outline_required" ? crossingPrintSaveHint
        : error.message === "print_outline_review_required" ? printOutlineSaveHint
          : error.message === "contrast_print_outline_review_required" ? contrastOutlineReviewHint
          : standalonePrintReviewErrors[error.message] || cropProtectionErrors[error.message] || error.message;
      setBlockedCrop(error.message === "protected_region_removed" ? { ...error.detail, photoId: selected.id } : null);
      setSaveState({ photoId: selected.id, mode: editMode, error: message });
    }
  }

  async function recheckContactEdges() {
    if (!refreshContactEdges || saveState?.saving) return;
    const target = { photoId: selected.id, mode: editMode };
    setSaveState({ ...target, saving: true, checkingEdges: true });
    try {
      await onReprocess(selected.id);
      setSaveState(null);
    } catch (error) {
      setSaveState({ ...target, error: error.message || "Papierkanten konnten nicht erneut geprüft werden." });
    }
  }

  async function rejectSelected() {
    const target = { pageId, photoId: selected.id, mode: editMode };
    setSaveState({ ...target, saving: true, rejecting: true });
    try {
      await onRejectPhoto(selected.id);
      setSaveState({ ...target, rejected: true });
    } catch (error) {
      setSaveState({ ...target, error: error.message });
    }
  }

  function startAddition(corners = [], adjacentSuggestion = null, deferredProposal = null) {
    setEditMode("full_print");
    onSelectPhoto(null);
    setSaveState(null);
    setAddition({ pageId, requestId: crypto.randomUUID(), corners, adjacentSuggestion, deferredProposal,
      printSizeFields: printSizeInputs(null) });
  }

  async function confirmCount() {
    setCountReview({ pageId, saving: true });
    try {
      await onConfirmCount(page, deferredProposals.map(({ id }) => id));
      if (activePage.current === pageId) setCountReview(null);
    } catch (error) {
      if (activePage.current === pageId) setCountReview({ pageId, error: error.message });
    }
  }

  async function resolveCountGroup(group) {
    const sharedPaper = group.kind === "paper-grid";
    setCountReview({ pageId, saving: true });
    try {
      const event = await onResolveCountGroup(page, group, groupSelection?.revision);
      if (activePage.current === pageId) setCountReview(sharedPaper ? { pageId, paperGridEventId: event.id } : null);
    } catch (error) {
      if (activePage.current === pageId) {
        setCountReview({ pageId, error: error.message.endsWith("_review_group_stale") ? countGroupReviewError : error.message });
      }
    } finally {
      setGroupSelection(current => current?.pageId === pageId ? null : current);
    }
  }

  async function saveNewPrint() {
    if (!newPrint || newPrint.saving || newPrint.invalidated || newPrint.corners.length !== 4
        || additionSize.error || needsPrintAdditionConfirmation) return;
    setAddition({ ...newPrint, saving: true, error: null });
    try {
      const photo = await onAddPhoto(pageId, newPrint.requestId, newPrint.corners, additionSize.size,
        newPrint.adjacentSuggestion, newPrint.deferredProposal, {
          replaceMountedFragments: newPrint.replaceMountedFragments,
          confirmDarkBacking: darkBackingDraft && newPrint.confirmStandalonePrint,
          confirmAlbumPocketPrint: Boolean(embossedAlbumAction(newPrint.deferredProposal?.reason)) && newPrint.confirmStandalonePrint,
          confirmStandalonePrint: requiresStandalonePrintFlag(newPrint.deferredProposal?.reason) && newPrint.confirmStandalonePrint,
        });
      if (activePage.current !== pageId) return;
      setAddition(null);
      onSelectPhoto(photo.id);
      setSaveState({ photoId: photo.id, mode: "full_print", url: photo.urls?.full_print });
    } catch (error) {
      const stale = printAdditionErrors[error.message];
      const message = stale || (error.message === "protected_region_cut"
        ? "Der Ausschnitt schneidet einen geschützten Bildbereich ab. Verschiebe die Ecken, bis der Bereich vollständig sichtbar ist."
        : error.message === "mounted_print_outline_excludes_fragments"
          ? "Die Ecken müssen alle markierten Teilvorschläge umfassen. Wenn sie zu verschiedenen Abzügen gehören, prüfe sie einzeln."
          : darkBackingReviewErrors[error.message] || embossedPocketReviewErrors[error.message]
            || standalonePrintReviewErrors[error.message] || sourceFragmentReviewErrors[error.message] || error.message);
      if (activePage.current === pageId) setAddition({ ...newPrint, error: message, invalidated: Boolean(stale) });
    }
  }

  if (!page) return <EmptyState label="Select a page to review." />;

  const currentSave = saveState?.photoId === selectedPhotoId && saveState?.mode === editMode ? saveState : null;
  return (
    <div className="canvas-wrap">
      <div className="variant-tabs crop-mode-tabs" role="group" aria-label="Zu bearbeitenden Ausschnitt wählen">
        {Object.entries(cropModeLabels).map(([mode, label]) => (
          <button key={mode} type="button" aria-pressed={editMode === mode} onClick={() => setEditMode(mode)}
            disabled={saveState?.saving || Boolean(newPrint) || (crossingReview && mode === "image_only")}>
            {label}
          </button>
        ))}
      </div>
      {pageId && (!editingImage || (canAddPhoto && !newPrint)) && (
        <div className="edge-measurement-controls">
          {canAddPhoto && !newPrint && <button className="text-button compact" type="button" onClick={() => startAddition()}
            disabled={saveState?.saving}>Fehlenden Abzug ergänzen</button>}
          {!editingImage && <>
            <button className="text-button compact" type="button" onClick={() => setMeasurementPage(pageId)}
              disabled={saveState?.saving}>Kantenmessung öffnen</button>
            {missingTraces.length > 0 && (
              <button className="text-button compact" type="button" aria-pressed={showMissingEdges}
                onClick={() => setShowMissingEdges(!showMissingEdges)}>Möglichen weiteren Abzug prüfen</button>
            )}
            {Boolean(measuredTraces?.length) && <button className="text-button compact" type="button"
              onClick={() => setMeasuredEdge(null)}>Messspur ausblenden</button>}
          </>}
        </div>
      )}
      {measurementPage && measurementPage === pageId && <EdgeMeasurement key={pageId} pageId={pageId}
        onClose={() => setMeasurementPage(null)} onMeasured={(traces) => setMeasuredEdge({ pageId, traces })} />}
      {!editingImage && showMissingEdges && missingTraces.length > 0 && (
        <p className="crop-editor-hint">{nearCaptureTraces?.length ? nearCapturePrintEdgeHint : missingPrintEdgeHint}</p>
      )}
      {!newPrint && !editingImage && (deferredProposals.length > 0 || mergedPairs.length > 0) && (
        <div className="manual-print-actions">
          <p className="crop-editor-hint">
            {photos.filter((photo) => !photo.reviewed_non_print).length} erfasste Abzüge{deferredProposals.length > 0
              && (deferredProposals.length === 1 ? ", 1 unklarer Vorschlag" : `, ${deferredProposals.length} unklare Vorschläge`)}.
            {deferredProposals.some(({ reason }) => reason === sourceFragmentProposal) && ` ${sourceFragmentCountHint}`}
            {deferredProposals.some(({ reason }) => reason !== sourceFragmentProposal && !standalonePrintNeedsConfirmation(reason))
              && ` ${deferredPrintHint}`}
          </p>
          <DarkBackingEvidenceLegend proposals={deferredProposals} />
          <DeferredPrintReviewHints proposals={deferredProposals} />
          {deferredProposals.map((proposal, index) => (
            <button className="text-button compact" key={proposal.id} type="button"
              onClick={() => startAddition(proposal.polygon, null, { ...proposal, revision: page.processing_metadata?.photo_count_revision,
                fragmentPolygons: getSourceEdgeTraces(proposal.fragment_polygons).filter(part => part.length >= 4),
              })}
              aria-describedby={deferredPrintCountHintId(proposal.reason)}
              disabled={!canAddPhoto || saveState?.saving || currentCountReview?.saving}>
              {proposal.reason === sourceFragmentProposal ? "Möglichen gemeinsamen Abzug prüfen"
                : proposal.reason === darkBackingProposal ? `Dunkle Fläche prüfen${deferredProposals.length > 1 ? ` (${index + 1})` : ""}`
                : isNestedFallbackProposal(proposal.reason) ? `Innenkontur ${index + 1} prüfen`
                : proposal.reason === captureFabricProposal ? `Stoff am Aufnahmerand ${index + 1} prüfen`
                : embossedAlbumAction(proposal.reason) ?? ([frameBackingProposal, borrowedFallbackProposal].includes(proposal.reason) ? `Vorschlag ${index + 1} am Aufnahmerand prüfen`
                : proposal.reason === "possible_mounted_print" ? `Möglichen Ganzabzug ${index + 1} prüfen`
                : `Vorschlag ${index + 1} als Abzug ergänzen`)}
            </button>
          ))}
          <button className="text-button compact" type="button" onClick={confirmCount}
            disabled={!canAddPhoto || saveState?.saving || currentCountReview?.saving
              || ["paper-grid", "crossing", "album_leaf"].includes(selectedCountGroup?.kind)}>
            {currentCountReview?.saving ? "Anzahl wird gespeichert …" : "Anzahl bestätigen"}
          </button>
        </div>
      )}
      {currentCountReview?.error && <p role="alert">{currentCountReview.error}</p>}
      {paperGridConfirmed && <p className="crop-editor-hint" role="status"
        id="paper-grid-count-result">{paperGridCountSaved}</p>}
      {countOutlines.length > 0 && <p className="crop-editor-hint" id="fragment-count-review-hint">{fragmentCountHint}</p>}
      {mergedPairs.length > 0 && <p className="crop-editor-hint" id="merged-print-pair-hint">{mergedPrintPairHint}</p>}
      {captureSplitGroups.length > 0 && <p className="crop-editor-hint" id="capture-split-count-hint">{captureSplitReviewHint}</p>}
      <CaptureSplitSourceReview groups={captureSplitGroups} photos={photos}
        sourceUrl={pageImage?.pageId === pageId ? pageImage.image.src : null} />
      {filmGridGroups.length > 0 && <p className="crop-editor-hint" id="film-grid-count-hint">
        {filmGridCountHint}{filmGridGroups.some(({ rim }) => rim) && ` ${filmGridRimHint}`}
      </p>}
      {paperJoinGroups.length > 0 && <p className="crop-editor-hint" id="paper-join-count-hint">{paperJoinCountHint}</p>}
      <CountGroupReview groups={countGroups} selected={selectedCountGroup}
        sourceUrl={pageImage?.pageId === pageId ? pageImage.image.src : null}
        busy={!canAddPhoto || Boolean(saveState?.saving || currentCountReview?.saving)}
        onSelect={key => {
          setGroupSelection(key == null ? null : { pageId, key, revision: page.processing_metadata?.photo_count_revision });
          setCountReview(null);
        }}
        onConfirm={resolveCountGroup} />
      {paperGridGroups.length > 0 && <p className="crop-editor-hint" id="paper-grid-count-hint">{paperGridCountHint}</p>}
      {paperGridOutlines.length > 0 && (
        <div className="manual-print-actions">
          {paperGridOutlines.map(({ photo, quad }, index) => (
            <button key={photo.id} className="text-button compact" type="button"
              disabled={saveState?.saving || currentCountReview?.saving || selectedCountGroup?.kind === "paper-grid"}
              aria-describedby={paperGridGroups.length > 0 ? "paper-grid-count-hint"
                : paperGridConfirmed ? "paper-grid-count-result" : undefined}
              onClick={() => loadPrintOutline(photo, quad)}>
              {`Aussenumriss prüfen${paperGridOutlines.length > 1 ? ` (${index + 1})` : ""}`}
            </button>
          ))}
        </div>
      )}
      {!newPrint && !editingImage && !selected && pendingNested > 0 && (
        <p className="crop-editor-hint" role="status">
          {pendingNested === 1 ? "1 unklarer Vorschlag" : `${pendingNested} unklare Vorschläge`}. Wähle einen Abzug mit gestrichelter Innenkontur.
        </p>
      )}
      {newPrint && (
        <p className="crop-editor-hint" aria-live="polite">
          {newPrint.corners.length === 4 ? "Prüfe und verschiebe die Ecken an den sichtbaren Papierrand."
            : `Markiere die vier Ecken des Abzugs auf der Albumseite. ${newPrint.corners.length} von 4 gesetzt.`}
        </p>
      )}
      {!newPrint && !editingImage && alternateOutline && (
        <>
          <p className="crop-editor-hint">{suggestedNested ? nestedPrintHint : suggestedNeighbor ? adjacentPrintHint
            : "Prüfe den gestrichelten Vorschlag am Original. Ergänze ihn, falls dort ein eigener Abzug aufliegt."}</p>
          <button className="text-button compact" type="button" disabled={!canAddPhoto || saveState?.saving}
            onClick={() => startAddition(alternateOutline, suggestedNested
              ? { photo_id: selected.id, polygon: suggestedNested, kind: "nested" }
              : suggestedNeighbor ? { photo_id: selected.id, polygon: suggestedNeighbor } : null)}>
            {suggestedNeighbor && !suggestedNested ? "Vorschlag für eigenen Abzug bearbeiten" : "Gestrichelten Vorschlag als Abzug ergänzen"}
          </button>
        </>
      )}
      {!newPrint && !editingImage && printEdgeReviewHint && <p className="crop-editor-hint">{printEdgeReviewHint}</p>}
      {sourceStrip && <SourceStripReview strip={sourceStrip} sourceUrl={`/api/v1/pages/${pageId}/image`} hintId="source-strip-review-hint" />}
      {!newPrint && deferredParentHints.map(({ id, text }) => <p key={id} className="crop-editor-hint">{text}</p>)}
      {!newPrint && !editingImage && <NeighborContentReview fullPrint={fullPrintMode} sourceUrl={`/api/v1/pages/${pageId}/image`} />}
      {markBounds.length > 0 && <AlbumPageMarkReview mark={albumMark} review={albumMarkReview}
        sourceUrl={`/api/v1/pages/${pageId}/image`} disabled={saveState?.saving} />}
      {editingImage && crossingReview && <p className="crop-editor-hint">{crossingPrintSaveHint}</p>}
      {needsPrintReview && !crossingReview && <p className="crop-editor-hint">{printOutlineSaveHint}</p>}
      {preliminaryMotifMarks && <p className="crop-editor-hint" id="held-motif-margin-hint">{motifBoundaryReviewHint}</p>}
      {editingImage && selected && !crossingReview && !needsPrintReview && (
        <>
          {!preliminaryMotifMarks && motifBoundaryReviewHint && <p className="crop-editor-hint">{motifBoundaryReviewHint}</p>}
          {!preliminaryMotifMarks && motifMarginMarks.length > 0 && <p className="crop-editor-hint" id="motif-margin-marks-hint">
            Die orange Markierung zeigt eine Stelle, die der vorgeschlagene Zuschnitt auslassen würde.
          </p>}
          <p className="crop-editor-hint">
            {edgeTraces.length > 0 && "Die gestrichelten Linien zeigen gemessene Abschnitte einer möglichen Motivkante. "}
            {!savedShape.imageQuad && "Die blauen Ecken starten am Papierrand. "}
            Verschiebe sie an die Motivgrenze und lass alle Bildfelder und zum Motiv gehörenden Beschriftungen im Zuschnitt.
            {motifPaperQuad && " Beim Ziehen bleiben die Motivkanten parallel zum entzerrten Papier."}
          </p>
        </>
      )}
      {editingImage && selected && !crossingReview && !needsPrintReview && (imageSuggestion || !savedShape.imageQuad) && (
        <button className="text-button compact" type="button" disabled={saveState?.saving} onClick={() => {
          const imageQuad = imageSuggestion || savedShape.warpQuad;
          setDraftShapes((current) => ({ ...current, [selected.id]: { ...selectedShape, imageQuad } }));
          setSaveState(null);
        }}>
          {imageSuggestion ? "Motivvorschlag in den Editor laden" : "Motiv reicht bis zum Papierrand"}
        </button>
      )}
      {!newPrint && !editingImage && (fragmentOutline || crossingOutline || printSuggestion) && (
        <>
          <button className="text-button compact" type="button" disabled={saveState?.saving}
            onClick={() => loadPrintOutline(selected, crossingOutline || printSuggestion || cyclicQuad(fragmentOutline?.quad))}>
            Aussenumriss in den Editor laden
          </button>
          {fragmentOutline && selectedShape.usingOutlineSuggestion
            && <p className="crop-editor-hint">Prüfe besonders die gestrichelte Papierkante.</p>}
        </>
      )}
      <canvas
        ref={canvasRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        className={newPrint ? "mark-print-corners" : undefined}
        aria-label="Albumseite mit markierten Abzügen"
        aria-describedby={[
          countOutlines.length && "fragment-count-review-hint", filmGridGroups.length && "film-grid-count-hint",
          paperGridGroups.length && "paper-grid-count-hint",
          paperJoinGroups.length && "paper-join-count-hint",
          captureSplitGroups.length && "capture-split-count-hint",
          mergedPairs.length && "merged-print-pair-hint",
          competingPhotoOutlines.length && "competing-photo-outline-hint",
          contrastCrossingTraces.length && "contrast-crossing-review-hint",
          ribbedReviewOutline && "ribbed-print-outline-hint",
          sourceStrip && "source-strip-review-hint",
          preliminaryMotifMarks ? "held-motif-margin-hint"
            : editingImage && motifMarginMarks.length && !crossingReview && !needsPrintReview && "motif-margin-marks-hint",
        ].filter(Boolean).join(" ") || undefined}
      />
      {competingPhotoOutlines.length > 0 && <CompetingPhotoOutlineLegend id="competing-photo-outline-hint" />}
      {contrastCrossingTraces.length > 0 && <p className="crop-editor-hint" id="contrast-crossing-review-hint">{contrastCrossingTraceHint}</p>}
      {ribbedReviewOutline && <p className="crop-editor-hint" id="ribbed-print-outline-hint">{ribbedPrintOutlineHint}</p>}
      {!newPrint && detectionReviewHint && !rejectedDetection && (
        <div className="manual-print-actions">
          <p className="crop-editor-hint">{detectionReviewHint}</p>
          <button className="text-button compact" type="button" onClick={rejectSelected} disabled={saveState?.saving}>
            {currentSave?.rejecting ? "Vorschlag wird geprüft …" : "Falsches Objekt"}
          </button>
        </div>
      )}
      {newPrint && (
        <div className="manual-print-actions">
          {standaloneDraft && <>
            {darkBackingDraft ? <DarkBackingEvidenceLegend proposals={[newPrint.deferredProposal]} />
              : <DeferredPrintReviewHints proposals={[newPrint.deferredProposal]} draft />}
            <label className="mounted-print-fragment-review">
              <input type="checkbox" checked={Boolean(newPrint.confirmStandalonePrint)}
                onChange={event => setAddition(current => ({ ...current, confirmStandalonePrint: event.target.checked, error: null }))}
                disabled={newPrint.saving || newPrint.invalidated} />
              {standalonePrintConfirmation}
            </label>
          </>}
          {newPrint.deferredProposal?.fragmentPolygons?.length >= 2 && (
            <label className="mounted-print-fragment-review">
              <input type="checkbox" checked={Boolean(newPrint.replaceMountedFragments)}
                onChange={event => setAddition(current => ({ ...current, replaceMountedFragments: event.target.checked, error: null }))}
                disabled={newPrint.saving || newPrint.invalidated} />
              Alle violett markierten Teilvorschläge gehören zu diesem Abzug.
            </label>
          )}
          {newPrint.corners.length === 4 && <PrintSizeFields
            value={newPrint.printSizeFields}
            onChange={(fields) => setAddition((current) => ({ ...current, printSizeFields: fields, error: null }))}
            disabled={newPrint.saving || newPrint.invalidated}
            error={additionSize.error}
          />}
          <button className="primary-button compact" type="button" onClick={saveNewPrint}
            disabled={newPrint.saving || newPrint.invalidated || newPrint.corners.length !== 4 || Boolean(additionSize.error)
              || needsPrintAdditionConfirmation}>
            {newPrint.saving ? "Abzug wird berechnet …" : standaloneDraft ? "Als Abzug erfassen"
              : newPrint.replaceMountedFragments ? "Als einen Abzug speichern" : "Abzug speichern"}
          </button>
          <button className="text-button compact" type="button" onClick={() => startAddition([], newPrint.adjacentSuggestion, newPrint.deferredProposal)}
            disabled={newPrint.saving || newPrint.invalidated || newPrint.corners.length === 0}>
            Ecken zurücksetzen
          </button>
          <button className="text-button compact" type="button" onClick={() => setAddition(null)} disabled={newPrint.saving}>
            Abbrechen
          </button>
          {newPrint.error && <p role="alert">{newPrint.error}</p>}
        </div>
      )}
      {!newPrint && !editingImage && selectedMergedPair && selectedShape && (
        <div className="manual-print-actions">
          {selectedMergedPair.outlines.map((outline, index) => (
            <button key={index} className="text-button compact" type="button" disabled={saveState?.saving}
              onClick={() => loadPrintOutline(selected, cyclicQuad(sanitizePolygon(outline)))}>
              Umriss {index + 1} laden
            </button>
          ))}
          <label className="mounted-print-fragment-review">
            <input type="checkbox" checked={Boolean(selectedShape.confirmMergedPrintAsOne)} disabled={saveState?.saving}
              onChange={event => {
                setDraftShapes(current => ({ ...current, [selected.id]: { ...selectedShape, confirmMergedPrintAsOne: event.target.checked } }));
                setSaveState(null);
              }} />
            {mergedPrintAsOneLabel}
          </label>
        </div>
      )}
      {refreshContactEdges && (
        <button className="text-button compact" type="button" disabled={saveState?.saving} onClick={recheckContactEdges}>
          <RefreshCw size={16} />
          {currentSave?.checkingEdges ? "Papierkanten werden geprüft …" : "Papierkanten erneut prüfen"}
        </button>
      )}
      {!newPrint && !editingImage && contrastOutlineReview && selectedShape && (
        <label className="mounted-print-fragment-review">
          <input type="checkbox" checked={contrastOutlineConfirmed} disabled={saveState?.saving}
            onChange={event => setConfirmedContrastOutline(event.target.checked
              ? { photoId: selected.id, quad: selectedShape.warpQuad } : null)} />
          {contrastOutlineConfirmation}
        </label>
      )}
      {!editingImage && selectedShape && (
        <PrintSizeFields
          value={selectedShape.printSizeFields}
          onChange={(fields) => {
            setDraftShapes((current) => ({ ...current, [selected.id]: { ...selectedShape, printSizeFields: fields } }));
            setSaveState(null);
          }}
          disabled={saveState?.saving}
          error={sizeError}
        />
      )}
      {selectedShape && (
        <CropOrientationField value={selectedShape.orientationDegrees} disabled={saveState?.saving} onChange={(value) => {
          setDraftShapes((current) => ({ ...current, [selected.id]: { ...selectedShape, orientationDegrees: value } }));
          setSaveState(null);
        }} />
      )}
      {!newPrint && <button className="primary-button compact" type="button"
        disabled={!selected || saveState?.saving || Boolean(sizeError) || needsMotifSelection || crossingNeedsCorrection
          || needsPrintReview || contrastOutlineNeedsCheck}
        onClick={saveSelected}>
        <Save size={16} />
        {currentSave?.saving && !currentSave.rejecting && !currentSave.checkingEdges ? "Ausschnitt wird berechnet …" : "Zuschnitt speichern"}
      </button>}
      {currentSave?.error && <p role="alert">{currentSave.error}</p>}
      {(rejectedDetection || (saveState?.rejected && saveState.pageId === pageId))
        && <p role="status">Vorschlag als falsches Objekt markiert.</p>}
      {currentSave?.url && <img className="review-crop-preview" src={currentSave.url} alt={cropModeLabels[editMode]} />}
      {editingImage && motifPreview && !currentSave?.url && (
        <>
          <img className="review-crop-preview" src={motifPreview} alt="Motivvorschau zur Prüfung"
            aria-describedby={needsPrintReview && motifBoundaryReviewHint ? "held-motif-preview-hint" : undefined} />
          {needsPrintReview && motifBoundaryReviewHint &&
            <p className="crop-mode-status" id="held-motif-preview-hint">{motifBoundaryReviewHint}</p>}
        </>
      )}
      {!editingImage && !newPrint && fullPrintPreview && !currentSave?.url && (
        comparePaperEdge ? <PaperEdgePreviewComparison currentUrl={untrimmedPrint} possibleUrl={fullPrintPreview}
          hint={outlinePreviewHint} outlinePreview={fullPrintMode?.outline_preview} /> : <>
          <img className="review-crop-preview" src={fullPrintPreview} alt="Abzugsvorschau zur Prüfung" />
          {outlinePreviewHint && <p className="crop-mode-status">{outlinePreviewHint}</p>}
        </>
      )}
      {currentSave && !currentSave.saving && !currentSave.error && !currentSave.url && !currentSave.rejected && (
        <p role="status">Der gespeicherte Ausschnitt muss geprüft werden.</p>
      )}
    </div>
  );
}

function ReviewView({ albumId, albumStatus, pages, photos, reviewPhotos, selectedPageId, setSelectedPageId, refresh }) {
  const [reviewType, setReviewType] = useState("segmentation");
  const [reviewItems, setReviewItems] = useState([]);
  const [ocrRows, setOcrRows] = useState([]);
  const [selectedPhotoId, setSelectedPhotoId] = useState(null);

  const selectedPage = pages.find((page) => page.id === selectedPageId) || pages[0] || null;
  const pagePhotos = useMemo(() => {
    const enriched = new Map();
    reviewItems.forEach((item) => {
      const candidates = item.item_type === "segmentation" ? item.payload?.detections : [item.payload];
      (candidates || []).forEach((payload) => {
        if (payload?.photo_id) enriched.set(payload.photo_id, payload);
      });
    });
    return (reviewType === "geometry" ? reviewPhotos : photos)
      .filter((photo) => photo.page_id === selectedPage?.id)
      .map((photo) => ({ ...photo, ...(enriched.get(photo.id) || {}) }));
  }, [photos, reviewPhotos, reviewItems, reviewType, selectedPage]);
  const currentPhotoId = pagePhotos.some((photo) => photo.id === selectedPhotoId) ? selectedPhotoId : null;

  useEffect(() => {
    if (selectedPage && selectedPage.id !== selectedPageId) setSelectedPageId(selectedPage.id);
  }, [selectedPage, selectedPageId, setSelectedPageId]);

  useEffect(() => {
    if (!albumId) return;
    getReviewQueue(albumId, reviewType).then(setReviewItems).catch(() => setReviewItems([]));
  }, [albumId, albumStatus, reviewType]);

  useEffect(() => {
    if (!selectedPage) {
      setOcrRows([]);
      return;
    }
    getPageOcr(selectedPage.id).then(setOcrRows).catch(() => setOcrRows([]));
  }, [selectedPage]);

  async function saveOcr(row) {
    await patchOcr(row.id, { text_content: row.text_content, text_type: row.text_type || "unknown", is_verified: true });
    setOcrRows(await getPageOcr(selectedPage.id));
    await refresh();
  }

  async function saveBox(photoId, box, segmentationMask = null, confirmPrintOutline = false,
    albumMarkReviewId = null, albumMarkReviewIndices = null, confirmMergedPrintAsOne = false) {
    await patchBoundingBox(photoId, box, segmentationMask, confirmPrintOutline, albumMarkReviewId, albumMarkReviewIndices,
      confirmMergedPrintAsOne);
    const [detail] = await Promise.all([getPhoto(photoId), refreshCorrections()]);
    return detail;
  }

  async function addPhoto(pageId, requestId, corners, printSize, adjacentSuggestion, deferredProposal, confirmation) {
    let photo;
    try {
      photo = await addManualPhoto(pageId, requestId, corners, printSize, adjacentSuggestion, deferredProposal, confirmation);
    } catch (error) {
      if (printAdditionErrors[error.message]) await refreshCorrections().catch(() => {});
      throw error;
    }
    const [detail] = await Promise.all([getPhoto(photo.id), refreshCorrections()]);
    return detail;
  }

  async function rejectPhoto(photoId) {
    await submitReviewEvent({
      task_type: "geometry", task_id: photoId, answer: "wrong_object",
      idempotency_key: crypto.randomUUID(), metadata: { ui: "desktop" },
    });
    await refreshCorrections();
  }

  async function confirmCount(page, deferredProposalIds) {
    try {
      await submitReviewEvent({
        task_type: "segmentation", task_id: page.id, answer: "correct", idempotency_key: crypto.randomUUID(),
        metadata: { ui: "desktop", photo_count_revision: page.processing_metadata?.photo_count_revision,
          deferred_proposal_ids: deferredProposalIds },
      });
    } finally {
      await refreshCorrections().catch(() => {});
    }
  }

  async function resolveCountGroup(page, group, displayedRevision) {
    try {
      if (group.kind === "paper-grid") {
        return await confirmSharedPaperGrid(page.id, displayedRevision, group, "desktop");
      }
      return await excludeCountReviewGroup(page.id, displayedRevision, group, "desktop");
    } finally {
      await refreshCorrections().catch(() => {});
    }
  }

  async function reprocessSelected(photoId) {
    if (!photoId) return;
    await reprocessPhoto(photoId);
    await refreshCorrections();
  }

  async function refreshCorrections() {
    await Promise.all([refresh(), getReviewQueue(albumId, reviewType).then(setReviewItems)]);
  }

  return (
    <div className="review-layout">
      <section className="work-surface">
        <div className="section-title">
          <h2>Review Interface</h2>
          <div className="segmented-control">
            {["segmentation", "ocr", "geometry"].map((type) => (
              <button type="button" key={type} className={reviewType === type ? "active" : ""} onClick={() => setReviewType(type)}>
                {type}
              </button>
            ))}
          </div>
        </div>
        <ReviewCanvas
          page={selectedPage}
          photos={pagePhotos}
          selectedPhotoId={currentPhotoId}
          onSelectPhoto={setSelectedPhotoId}
          onSaveBox={saveBox}
          onAddPhoto={addPhoto}
          onRejectPhoto={rejectPhoto}
          onConfirmCount={confirmCount}
          onResolveCountGroup={resolveCountGroup}
          onReprocess={reprocessSelected}
          reviewingCount={reviewType === "segmentation"}
          reviewingGeometry={reviewType === "geometry"}
        />
      </section>
      <section className="side-surface">
        <div className="section-title">
          <h2>OCR Text</h2>
          <button className="icon-button" type="button" disabled={!currentPhotoId}
            onClick={() => reprocessSelected(currentPhotoId)} title="Reprocess selected photo">
            <RefreshCw size={16} />
          </button>
        </div>
        <div className="ocr-list">
          {ocrRows.map((row) => (
            <div className="ocr-row" key={row.id}>
              <textarea
                value={row.text_content}
                onChange={(event) =>
                  setOcrRows((current) => current.map((item) => (item.id === row.id ? { ...item, text_content: event.target.value } : item)))
                }
              />
              <div className="ocr-actions">
                <select
                  value={row.text_type || "unknown"}
                  onChange={(event) =>
                    setOcrRows((current) => current.map((item) => (item.id === row.id ? { ...item, text_type: event.target.value } : item)))
                  }
                >
                  <option value="caption">caption</option>
                  <option value="directory_name">directory</option>
                  <option value="unknown">unknown</option>
                </select>
                <button className="icon-button" type="button" onClick={() => saveOcr(row)} title="Approve OCR">
                  <Check size={16} />
                </button>
              </div>
            </div>
          ))}
          {ocrRows.length === 0 && <EmptyState label="No OCR results for this page." />}
        </div>
        <div className="review-queue">
          <h3>Queue</h3>
          {reviewItems.map((item) => (
            <button
              type="button"
              key={item.id}
              className="queue-row"
              onClick={() => {
                if (item.page_id) setSelectedPageId(item.page_id);
                if (item.payload?.photo_id) setSelectedPhotoId(item.payload.photo_id);
              }}
            >
              <span>{item.item_type === "geometry" && item.payload?.motif_review_pending
                && item.payload?.crop_modes?.full_print?.status === "ready" ? "Nur Bildmotiv prüfen" : item.item_type}</span>
              <StatusBadge status={item.status} />
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}

function GalleryView({ photos, processing, onOpenPhoto = () => {} }) {
  const [detail, setDetail] = useState(null);
  const [selectedVariant, setSelectedVariant] = useState("original");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const crops = photos.filter(hasPhotoCrop);

  const variantEntries = availableVariantEntries(detail);
  const selectedVariantUrl = detail?.urls?.[selectedVariant] || detail?.urls?.original;
  const selectedVariantRecord = detail?.enhancement_applied?.variants?.[selectedVariant];

  useEffect(() => {
    if (!detail) return;
    const entries = availableVariantEntries(detail);
    const preferred = entries.find(([name]) => name === "full_print")
      || entries.find(([name]) => name === detail?.enhancement_applied?.selected_variant)
      || entries.find(([name]) => name === "original") || entries[0];
    if (preferred) setSelectedVariant(preferred[0]);
  }, [detail?.id]);

  async function runSearch(event) {
    event.preventDefault();
    if (!query.trim()) {
      setResults([]);
      return;
    }
    setResults(await search(query.trim()));
  }

  async function openGalleryPhoto(photoId) {
    const nextDetail = await getPhoto(photoId);
    setDetail(nextDetail);
    onOpenPhoto(nextDetail);
  }

  return (
    <div className="content-grid">
      <section className="work-surface">
        <div className="section-title">
          <h2>Photo Gallery</h2>
          <form className="search-form" onSubmit={runSearch}>
            <input value={query} onChange={(event) => setQuery(event.target.value)} />
            <button className="icon-button" type="submit" title="Search">
              <Search size={16} />
            </button>
          </form>
        </div>
        {results.length > 0 && (
          <div className="search-results">
            {results.map((result) => (
              <button type="button" key={`${result.page_id}-${result.text_content}`} onClick={() => result.photo_id && openGalleryPhoto(result.photo_id)}>
                <SafeText text={result.highlight} />
              </button>
            ))}
          </div>
        )}
        {!crops.length && <EmptyState label={processing ? "Extracting photos…" : "No extracted photos in this album yet."} />}
        <div className="gallery-grid">
          {crops.map((photo) => (
            <button type="button" className="photo-tile" key={photo.id} onClick={() => openGalleryPhoto(photo.id)}>
              <img src={photo.image_url} alt="" />
              <div>
                <StatusBadge status={photo.status} />
                <span>{photo.segmentation_confidence ? `${Math.round(photo.segmentation_confidence * 100)}%` : "manual"}</span>
              </div>
            </button>
          ))}
        </div>
      </section>
      <section className="side-surface">
        <div className="section-title">
          <h2>EXIF</h2>
          <Aperture size={18} />
        </div>
        {detail ? (
          <div className="detail-list">
            {variantEntries.length > 0 && (
              <div className="variant-tabs">
                {variantEntries.map(([name]) => (
                  <button
                    type="button"
                    key={name}
                    className={selectedVariant === name ? "active" : ""}
                    onClick={() => setSelectedVariant(name)}
                  >
                    {variantLabel(name)}
                  </button>
                ))}
              </div>
            )}
            <div className="variant-preview">
              {detail.urls?.original && selectedVariant !== "original" && (
                <button className="variant-preview-button" type="button" onClick={() => onOpenPhoto(detail)}>
                  <img className="detail-image" src={detail.urls.original} alt="" />
                </button>
              )}
              {selectedVariantUrl && (
                <button className="variant-preview-button" type="button" onClick={() => onOpenPhoto(detail)}>
                  <img className="detail-image" src={selectedVariantUrl} alt="" />
                </button>
              )}
            </div>
            <span>ID {detail.id}</span>
            <span>pHash {detail.phash || "pending"}</span>
            <span>Duplicate {detail.is_duplicate_of || "no"}</span>
            <span>Aspect {detail.aspect_ratio?.toFixed(2) || "unknown"}</span>
            {selectedVariantRecord?.metrics && (
              <div className="variant-metrics">
                <span>Blur {Math.round(selectedVariantRecord.metrics.blur_score || 0)}</span>
                {typeof selectedVariantRecord.metrics.source_similarity === "number" && (
                  <span>Similarity {Math.round(selectedVariantRecord.metrics.source_similarity * 100)}%</span>
                )}
                {selectedVariantRecord.warnings?.map((warning) => (
                  <StatusBadge key={warning} status={warning} />
                ))}
              </div>
            )}
            <pre>{JSON.stringify(detail.enhancement_applied, null, 2)}</pre>
            {detail.ocr_results?.map((ocr) => (
              <SafeText key={ocr.id} text={ocr.text_content} className="caption-line" />
            ))}
          </div>
        ) : (
          <EmptyState label="Select a photo." />
        )}
      </section>
    </div>
  );
}

function CaptureView({ albumId, refresh, uploadQueueScope }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [streaming, setStreaming] = useState(false);
  const [startingCamera, setStartingCamera] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [cameraMessage, setCameraMessage] = useState("");
  const [tilt, setTilt] = useState(0);
  const [batch, setBatch] = useState([]);
  const [message, setMessage] = useState("");
  const [progress, setProgress] = useState(0);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
  }, []);

  useEffect(() => {
    function onOrientation(event) {
      const levelTilt = Math.max(Math.abs(event.beta || 0), Math.abs(event.gamma || 0));
      setTilt(Math.round(levelTilt));
    }
    window.addEventListener("deviceorientation", onOrientation);
    function onUploadSync(event) {
      if (event.data?.type === "retry-upload-queue") {
        retryQueuedUploads(setMessage, uploadQueueScope).then(refresh);
      }
    }
    navigator.serviceWorker?.addEventListener("message", onUploadSync);
    return () => {
      window.removeEventListener("deviceorientation", onOrientation);
      navigator.serviceWorker?.removeEventListener("message", onUploadSync);
    };
  }, [refresh, uploadQueueScope]);

  async function startCamera() {
    setCameraMessage("");
    setStartingCamera(true);
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 4096 }, height: { ideal: 3072 } },
        audio: false,
      });
      const video = videoRef.current;
      if (!video) { stream.getTracks().forEach(track => track.stop()); return; }
      streamRef.current?.getTracks().forEach(track => track.stop());
      streamRef.current = stream;
      video.srcObject = stream;
      await video.play();
      if (stream === streamRef.current) setStreaming(true);
    } catch {
      if (stream && stream === streamRef.current) {
        stream.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        setStreaming(false);
      }
      if (videoRef.current) setCameraMessage("Camera could not be started. You can add an original photo to the batch.");
    } finally {
      if (videoRef.current) setStartingCamera(false);
    }
  }

  async function captureFrame() {
    const video = videoRef.current;
    const stream = streamRef.current;
    if (!video || !stream || capturing) return;
    setCapturing(true);
    setCameraMessage("");
    try {
      const { file, fromVideo } = await captureCameraImage(video, .92);
      if (stream !== streamRef.current) return;
      setBatch((current) => [...current, file]);
      if (fromVideo) setCameraMessage("Captured a video frame. For fine paper edges, add the original camera photo to the batch.");
    } catch {
      if (stream === streamRef.current) setCameraMessage("The camera could not take this photo. Try again or add an original photo to the batch.");
    } finally {
      if (stream === streamRef.current) setCapturing(false);
    }
  }

  async function uploadBatch() {
    if (!albumId || batch.length === 0) return;
    try {
      for (const file of batch) {
        if (!navigator.onLine) {
          await queueUpload(albumId, [file], uploadQueueScope);
          setMessage("Upload queued for retry.");
        } else if (file.size > 20 * 1024 * 1024) {
          await uploadFileChunked(albumId, file, setProgress);
        } else {
          await uploadFiles(albumId, [file]);
        }
      }
      setBatch([]);
      setProgress(0);
      await refresh();
    } catch (error) {
      await queueUpload(albumId, batch, uploadQueueScope);
      setMessage(error.message || "Upload queued for retry.");
    }
  }

  return (
    <div className="content-grid">
      <section className="work-surface">
        <div className="section-title">
          <h2>Mobile Upload</h2>
          <StatusBadge status={tilt > 10 ? "review_needed" : "completed"} />
        </div>
        <div className="camera-frame">
          <video ref={videoRef} autoPlay playsInline muted />
          {!streaming && <EmptyState label="Camera is off." />}
        </div>
        <div className="toolbar-row">
          <button className="icon-button text-button" type="button" disabled={startingCamera || capturing} onClick={startCamera}>
            <Camera size={16} />
            Camera
          </button>
          <button className="icon-button text-button" type="button" disabled={!streaming || startingCamera || capturing || tilt > 10} onClick={captureFrame}>
            <Aperture size={16} />
            Capture
          </button>
          <span className="muted-text">Tilt {tilt} degrees</span>
        </div>
        {cameraMessage && <p className="muted-text" role="status">{cameraMessage}</p>}
      </section>
      <section className="side-surface">
        <div className="section-title">
          <h2>Batch</h2>
          <Upload size={18} />
        </div>
        <input type="file" accept="image/jpeg,image/png" multiple onChange={(event) => {
          const files = [...event.target.files];
          setBatch(current => [...current, ...files]);
        }} />
        <button className="primary-button" type="button" disabled={!albumId || capturing || batch.length === 0} onClick={uploadBatch}>
          <Upload size={16} />
          Upload {batch.length}
        </button>
        {progress > 0 && <div className="progress-track"><span style={{ width: `${Math.round(progress * 100)}%` }} /></div>}
        {message && <p className="muted-text">{message}</p>}
        <div className="batch-list">
          {batch.map((file) => (
            <span key={`${file.name}-${file.lastModified}`}>{file.name}</span>
          ))}
        </div>
      </section>
    </div>
  );
}

export default function App({ uploadQueueScope }) {
  if (window.location.pathname.startsWith("/mobile")) return <MobileApp uploadQueueScope={uploadQueueScope} />;
  const [activeTab, setActiveTab] = useState("analyze");
  const [albums, setAlbums] = useState([]);
  const [selectedAlbumId, setSelectedAlbumId] = useState(null);
  const [pages, setPages] = useState([]);
  const [photos, setPhotos] = useState([]);
  const [selectedPageId, setSelectedPageId] = useState(null);
  const [reviewPhotos, setReviewPhotos] = useState([]);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [notice, setNotice] = useState("");
  const [photoViewerDetail, setPhotoViewerDetail] = useState(null);
  const selectedAlbumRef = useRef(null);

  const selectedAlbum = albums.find((album) => album.id === selectedAlbumId) || null;

  const selectAlbum = useCallback((albumId, nextTab) => {
    if (selectedAlbumRef.current !== albumId) {
      selectedAlbumRef.current = albumId;
      setSelectedAlbumId(albumId);
      setSelectedPageId(null);
      setPages([]);
      setPhotos([]);
      setReviewPhotos([]);
    }
    if (nextTab) setActiveTab(nextTab);
  }, []);

  const refreshAlbums = useCallback(async () => {
    setLoading(true);
    try {
      const nextAlbums = await listAlbums();
      setAlbums(nextAlbums);
      if (!selectedAlbumRef.current && nextAlbums.length > 0) selectAlbum(nextAlbums[0].id);
    } finally {
      setLoading(false);
    }
  }, [selectAlbum]);

  const refreshAlbumData = useCallback(async () => {
    if (!selectedAlbumId) return;
    const nextAlbums = await listAlbums();
    const [nextPages, nextPhotos, history] = await Promise.all([
      listPages(selectedAlbumId), listPhotos(selectedAlbumId),
      activeTab === "review" ? listPhotos(selectedAlbumId, { includeRejected: true }) : null,
    ]);
    if (selectedAlbumRef.current !== selectedAlbumId) return;
    setPages(nextPages);
    setPhotos(nextPhotos);
    if (history) setReviewPhotos(history);
    setAlbums(nextAlbums);
    setSelectedPageId((current) => nextPages.some((page) => page.id === current) ? current : nextPages[0]?.id || null);
  }, [activeTab, selectedAlbumId]);

  useEffect(() => {
    refreshAlbums().catch((error) => setNotice(error.message));
  }, [refreshAlbums]);

  useEffect(() => {
    refreshAlbumData().catch((error) => setNotice(error.message));
  }, [activeTab, refreshAlbumData]);

  useEffect(() => {
    if (activeTab === "analyze" || selectedAlbum?.status !== "processing") return;
    let refreshing = false;
    const timer = window.setInterval(async () => {
      if (refreshing) return;
      refreshing = true;
      try {
        await refreshAlbumData();
      } catch (error) {
        setNotice(error.message);
      } finally {
        refreshing = false;
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [activeTab, refreshAlbumData, selectedAlbum?.status]);

  async function handleCreateAlbum(payload) {
    const album = await createAlbum(payload);
    selectAlbum(album.id, "pages");
    setAlbums(await listAlbums());
  }

  async function handleUpload(files) {
    const response = await uploadFiles(selectedAlbumId, files);
    await refreshAlbumData();
    return response.analysis_queued;
  }

  async function handleAnalyze() {
    if (!selectedAlbumId || analyzing) return;
    const pageIds = pages.filter((page) => page.status !== "completed").map((page) => page.id);
    if (!pageIds.length) return;
    setAnalyzing(true);
    setNotice("");
    try {
      await analyzeAlbumNow(selectedAlbumId, pageIds);
      await refreshAlbumData();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setAnalyzing(false);
    }
  }

  async function openPhoto(photo) {
    try {
      const detail = photo.urls ? photo : await getPhoto(photo.id || photo);
      setPhotoViewerDetail(detail);
    } catch (error) {
      setNotice(error.message || "Unable to open photo.");
    }
  }

  return (
    <main>
      <header className="app-header">
        <div>
          <h1>Album Digitizer</h1>
          <p>{selectedAlbum ? selectedAlbum.name : "Local review workstation"}</p>
        </div>
        <nav>
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                type="button"
                key={tab.id}
                className={activeTab === tab.id ? "active" : ""}
                onClick={() => {
                  setActiveTab(tab.id);
                  if (tab.id === "training") setNotice("");
                }}
                title={tab.label}
              >
                <Icon size={18} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </header>

      {notice && activeTab !== "training" && (
        <button type="button" className="notice" onClick={() => setNotice("")}>
          {notice}
        </button>
      )}

      {activeTab === "analyze" && (
        <OrderedAnalyzer
          onOpenPhoto={openPhoto}
          onAlbumReady={selectAlbum}
        />
      )}
      {activeTab === "albums" && (
        <AlbumList
          albums={albums}
          loading={loading}
          selectedAlbumId={selectedAlbumId}
          pages={pages}
          photos={photos}
          onOpen={(albumId) => selectAlbum(albumId, "albums")}
          onOpenPhoto={openPhoto}
          onCreate={handleCreateAlbum}
        />
      )}
      {activeTab === "pages" && (
        <PageGrid
          albumId={selectedAlbumId}
          pages={pages}
          selectedPageId={selectedPageId}
          analyzing={analyzing}
          processing={selectedAlbum?.status === "processing"}
          onAnalyze={handleAnalyze}
          onUpload={handleUpload}
          onSelectPage={setSelectedPageId}
        />
      )}
      {activeTab === "review" && (
        <ReviewView
          albumId={selectedAlbumId}
          albumStatus={selectedAlbum?.status}
          pages={pages}
          photos={photos}
          reviewPhotos={reviewPhotos}
          selectedPageId={selectedPageId}
          setSelectedPageId={setSelectedPageId}
          refresh={refreshAlbumData}
        />
      )}
      {activeTab === "training" && <TrainingAnnotationView />}
      {activeTab === "gallery" && <GalleryView photos={photos} processing={selectedAlbum?.status === "processing"} onOpenPhoto={openPhoto} />}
      {activeTab === "capture" && <CaptureView albumId={selectedAlbumId} refresh={refreshAlbumData} uploadQueueScope={uploadQueueScope} />}
      {photoViewerDetail && (
        <PhotoCompareModal detail={photoViewerDetail} title={`Photo ${String(photoViewerDetail.id).slice(0, 8)}`} onClose={() => setPhotoViewerDetail(null)} />
      )}
    </main>
  );
}
