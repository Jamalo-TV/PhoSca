import DOMPurify from "dompurify";
import {
  Aperture,
  Camera,
  Check,
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
  analyzeAlbum,
  analyzeAlbumNow,
  createAlbum,
  getPageOcr,
  getPhoto,
  getReviewQueue,
  listAlbums,
  listPages,
  listPhotos,
  patchBoundingBox,
  patchOcr,
  reprocessPhoto,
  search,
  uploadFileChunked,
  uploadFiles,
} from "./api.js";
import { queueUpload, retryQueuedUploads } from "./offlineQueue.js";

const tabs = [
  { id: "analyze", label: "Analyze", icon: Play },
  { id: "albums", label: "Albums", icon: GalleryHorizontal },
  { id: "pages", label: "Pages", icon: FileImage },
  { id: "review", label: "Review", icon: Pencil },
  { id: "gallery", label: "Gallery", icon: Aperture },
  { id: "capture", label: "Capture", icon: Camera },
];

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
  return <span className={className} dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(text || "") }} />;
}

function EmptyState({ label }) {
  return <div className="empty-state">{label}</div>;
}

function variantLabel(name) {
  return name.replaceAll("_", " ");
}

function availableVariantEntries(detail) {
  return Object.entries(detail?.urls || {}).filter(([, url]) => Boolean(url));
}

function comparisonUrls(detail) {
  const before = detail?.urls?.original || detail?.urls?.perspective_corrected || detail?.urls?.enhanced;
  const after = detail?.urls?.enhanced || detail?.urls?.premium || before;
  return { before, after };
}

function fileSizeLabel(bytes) {
  if (!bytes) return "0 B";
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function PhotoCompareModal({ detail, imageSrc, title = "Photo", onClose }) {
  const [slider, setSlider] = useState(50);
  const { before, after } = comparisonUrls(detail);
  const canCompare = Boolean(before && after && before !== after);
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

  if (!displaySrc && !canCompare) return null;

  return (
    <div className="fullscreen-overlay" onClick={onClose}>
      <div className="photo-modal" onClick={(event) => event.stopPropagation()}>
        <div className="photo-modal-header">
          <strong>{title}</strong>
          <button className="icon-button" type="button" onClick={onClose} title="Close">
            <X size={16} />
          </button>
        </div>
        {canCompare ? (
          <div className="compare-viewer" style={{ "--compare-position": `${slider}%` }}>
            <img className="compare-image compare-before" src={before} alt="Before enhancement" />
            <div className="compare-after-wrap">
              <img className="compare-image compare-after" src={after} alt="After enhancement" />
            </div>
            <span className="compare-label compare-label-before">Before</span>
            <span className="compare-label compare-label-after">After</span>
            <span className="compare-divider" />
          </div>
        ) : (
          <img className="single-photo-preview" src={displaySrc} alt="" />
        )}
        {canCompare && (
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

  useEffect(() => {
    itemsRef.current = items;
  }, [items]);

  useEffect(() => {
    return () => {
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
      output: null,
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
    onAlbumReady?.(album.id);
    return album.id;
  }

  async function uploadPending(existingAlbumId = null, sourceItems = items) {
    const targetAlbumId = existingAlbumId || (await ensureAlbum());
    const pending = sourceItems.filter((item) => !item.pageId);
    if (pending.length === 0) return { targetAlbumId, orderedItems: sourceItems };

    setPhase("Uploading");
    setItems((current) =>
      current.map((item) => (pending.some((pendingItem) => pendingItem.clientId === item.clientId) ? { ...item, uploadStatus: "uploading" } : item))
    );
    const response = await uploadFiles(targetAlbumId, pending.map((item) => item.file));
    const pageByClientId = new Map(pending.map((item, index) => [item.clientId, response.pages[index]]));
    const orderedItems = sourceItems.map((item) => {
      const page = pageByClientId.get(item.clientId);
      if (!page) return item;
      return { ...item, pageId: page.page_id, uploadStatus: "uploaded", analysisStatus: "ready" };
    });
    setItems(orderedItems);
    return { targetAlbumId, orderedItems };
  }

  async function uploadOnly() {
    if (items.length === 0 || busy) return;
    setBusy(true);
    setMessage("");
    try {
      await uploadPending();
      setPhase("Uploaded");
      setMessage("Upload complete.");
    } catch (error) {
      setMessage(error.message || "Upload failed.");
      setItems((current) => current.map((item) => (item.uploadStatus === "uploading" ? { ...item, uploadStatus: "selected" } : item)));
    } finally {
      setBusy(false);
    }
  }

  async function analyzeInOrder() {
    if (items.length === 0 || busy) return;
    setBusy(true);
    setMessage("");
    try {
      const targetAlbumId = await ensureAlbum();
      const uploaded = await uploadPending(targetAlbumId);
      const orderedItems = uploaded.orderedItems;
      const pageIds = orderedItems.map((item) => item.pageId).filter(Boolean);
      if (pageIds.length === 0) return;

      setPhase("Analyzing");
      setItems((current) =>
        current.map((item) => (item.pageId ? { ...item, analysisStatus: "analyzing", error: null, output: null } : item))
      );
      const result = await analyzeAlbumNow(targetAlbumId, pageIds);
      const [latestPages, latestPhotos, ocrResponses] = await Promise.all([
        listPages(targetAlbumId),
        listPhotos(targetAlbumId),
        Promise.all(pageIds.map((pageId) => getPageOcr(pageId).catch(() => []))),
      ]);

      const pageById = new Map(latestPages.map((page) => [page.id, page]));
      const resultByPageId = new Map((result.pages || []).map((pageResult) => [pageResult.page_id, pageResult]));
      const ocrByPageId = new Map(pageIds.map((pageId, index) => [pageId, ocrResponses[index]]));

      setItems(
        orderedItems.map((item) => {
          if (!item.pageId) return item;
          const pageResult = resultByPageId.get(item.pageId);
          return {
            ...item,
            uploadStatus: "uploaded",
            analysisStatus: pageResult?.status || "completed",
            page: pageById.get(item.pageId) || null,
            photos: latestPhotos.filter((photo) => photo.page_id === item.pageId),
            ocrRows: ocrByPageId.get(item.pageId) || [],
            output: pageResult || null,
            error: pageResult?.error || null,
          };
        })
      );
      await onAlbumReady?.(targetAlbumId);
      setPhase("Complete");
      setMessage("Analysis complete.");
    } catch (error) {
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
              <input type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={(event) => addFiles(event.target.files)} />
            </label>
          </div>
        </div>
        <div className="analyzer-actions">
          <button className="icon-button text-button" type="button" disabled={busy || items.length === 0} onClick={uploadOnly}>
            {busy && phase === "Uploading" ? <Loader2 className="spin-icon" size={16} /> : <Upload size={16} />}
            Upload
          </button>
          <button className="primary-button" type="button" disabled={busy || items.length === 0} onClick={analyzeInOrder}>
            {busy && phase === "Analyzing" ? <Loader2 className="spin-icon" size={16} /> : <Play size={16} />}
            Analyze
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
                    <span>{item.page?.status || item.analysisStatus}</span>
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
                {!item.output && !item.error && item.analysisStatus !== "analyzing" && <p className="muted-text">Waiting for analysis.</p>}
                {item.output && !item.error && (
                  <div className="output-details">
                    <div className="stats-row">
                      <span>{item.photos.length} photos</span>
                      <span>{item.ocrRows.length} OCR rows</span>
                      <span>{item.page?.blur_score ? `Blur ${item.page.blur_score.toFixed(1)}` : "Blur pending"}</span>
                    </div>
                    {item.ocrRows.length > 0 && (
                      <div className="ocr-output">
                        {item.ocrRows.map((row) => (
                          <SafeText key={row.id} text={row.text_content} />
                        ))}
                      </div>
                    )}
                    {item.photos.length > 0 && (
                      <div className="mini-photo-grid">
                        {item.photos.map((photo) => (
                          <img
                            key={photo.id}
                            src={`/api/v1/photos/${photo.id}/image`}
                            alt=""
                            style={{ cursor: "pointer" }}
                            onClick={() => onOpenPhoto(photo)}
                          />
                        ))}
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

function AlbumList({ albums, selectedAlbumId, pages = [], photos = [], onSelect, onCreate, onOpenPhoto = () => {}, loading }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const selectedAlbum = albums.find((album) => album.id === selectedAlbumId) || null;

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
                  onClick={() => onSelect(album.id)}
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
            {photos.length > 0 ? (
              <div className="album-photo-grid">
                {photos.map((photo) => (
                  <button type="button" className="photo-tile" key={photo.id} onClick={() => onOpenPhoto(photo)}>
                    <img src={`/api/v1/photos/${photo.id}/image`} alt="" />
                    <div>
                      <StatusBadge status={photo.status} />
                      <span>{photo.segmentation_confidence ? `${Math.round(photo.segmentation_confidence * 100)}%` : "manual"}</span>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <EmptyState label="No extracted photos in this album yet." />
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

function PageGrid({ albumId, pages, onAnalyze, onUpload, onSelectPage, selectedPageId }) {
  const [files, setFiles] = useState([]);

  async function uploadSelected() {
    if (!albumId || files.length === 0) return;
    await onUpload(files);
    setFiles([]);
  }

  return (
    <div className="content-grid">
      <section className="work-surface">
        <div className="section-title">
          <h2>Page Grid</h2>
          <button className="icon-button text-button" type="button" onClick={() => onAnalyze()}>
            <Play size={16} />
            Analyze
          </button>
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
          <button className="primary-button" type="button" disabled={!albumId || files.length === 0} onClick={uploadSelected}>
            <Upload size={16} />
            Upload {files.length || ""}
          </button>
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
  const points = polygon.slice(0, 4).map((point) => ({ x: clampUnit(Number(point.x)), y: clampUnit(Number(point.y)) }));
  return points.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y)) ? points : null;
}

function polygonFromPhoto(photo) {
  return sanitizePolygon(photo.segmentation_mask?.polygon) || boxToPolygon(photo.bounding_box);
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
  const polygon = polygonFromPhoto(photo);
  return { polygon, boundingBox: boundingBoxFromPolygon(polygon) };
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

function maskFromPolygon(polygon) {
  return {
    polygon: polygon.map((point) => ({ x: clampUnit(point.x), y: clampUnit(point.y) })),
    source: "manual_quad",
  };
}

function ReviewCanvas({ page, photos, onSaveBox, selectedPhotoId, onSelectPhoto }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [draftShapes, setDraftShapes] = useState({});
  const dragRef = useRef(null);

  useEffect(() => {
    setDraftShapes(Object.fromEntries(photos.map((photo) => [photo.id, draftShapeFromPhoto(photo)])));
  }, [photos]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !page) return;
    const context = canvas.getContext("2d");
    const maxWidth = canvas.parentElement?.clientWidth || 720;
    const ratio = image.naturalHeight / image.naturalWidth || 0.75;
    canvas.width = maxWidth;
    canvas.height = Math.round(maxWidth * ratio);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    photos.forEach((photo) => {
      const shape = draftShapes[photo.id] || draftShapeFromPhoto(photo);
      const points = shape.polygon.map((point) => ({ x: point.x * canvas.width, y: point.y * canvas.height }));
      context.strokeStyle = photo.id === selectedPhotoId ? "#0f766e" : "#f59e0b";
      context.lineWidth = photo.id === selectedPhotoId ? 4 : 2;
      context.beginPath();
      points.forEach((point, index) => {
        if (index === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      });
      context.closePath();
      context.stroke();
      if (photo.id === selectedPhotoId) {
        context.fillStyle = "rgba(15, 118, 110, 0.1)";
        context.fill();
      }
      context.fillStyle = "#ffffff";
      points.forEach((point) => {
        context.fillRect(point.x - 5, point.y - 5, 10, 10);
        context.strokeRect(point.x - 5, point.y - 5, 10, 10);
      });
    });
  }, [draftShapes, page, photos, selectedPhotoId]);

  useEffect(() => {
    draw();
  }, [draw]);

  useEffect(() => {
    if (!page) return;
    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      draw();
    };
    image.src = `/api/v1/pages/${page.id}/image`;
  }, [draw, page]);

  function pointToBox(event) {
    const rect = canvasRef.current.getBoundingClientRect();
    return { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height };
  }

  function onPointerDown(event) {
    const point = pointToBox(event);
    for (const photo of [...photos].reverse()) {
      const shape = draftShapes[photo.id] || draftShapeFromPhoto(photo);
      const cornerIndex = shape.polygon.findIndex((corner) => Math.hypot(point.x - corner.x, point.y - corner.y) < 0.035);
      if (cornerIndex >= 0) {
        onSelectPhoto(photo.id);
        dragRef.current = { photoId: photo.id, pointIndex: cornerIndex };
        canvasRef.current.setPointerCapture(event.pointerId);
        return;
      }
      if (pointInPolygon(point, shape.polygon)) {
        onSelectPhoto(photo.id);
        return;
      }
    }
  }

  function onPointerMove(event) {
    if (!dragRef.current) return;
    const point = pointToBox(event);
    const { photoId, pointIndex } = dragRef.current;
    setDraftShapes((current) => {
      const photo = photos.find((candidate) => candidate.id === photoId);
      const currentShape = current[photoId] || (photo ? draftShapeFromPhoto(photo) : null);
      if (!currentShape) return current;
      const polygon = currentShape.polygon.map((corner, index) =>
        index === pointIndex ? { x: clampUnit(point.x), y: clampUnit(point.y) } : corner
      );
      return { ...current, [photoId]: { polygon, boundingBox: boundingBoxFromPolygon(polygon) } };
    });
  }

  function onPointerUp(event) {
    if (dragRef.current) {
      canvasRef.current.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
  }

  async function saveSelected() {
    if (!selectedPhotoId || !draftShapes[selectedPhotoId]) return;
    const shape = draftShapes[selectedPhotoId];
    await onSaveBox(selectedPhotoId, shape.boundingBox, maskFromPolygon(shape.polygon));
  }

  if (!page) return <EmptyState label="Select a page to review." />;

  return (
    <div className="canvas-wrap">
      <canvas
        ref={canvasRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        aria-label="Segmented album page"
      />
      <button className="primary-button compact" type="button" disabled={!selectedPhotoId} onClick={saveSelected}>
        <Save size={16} />
        Save Corners
      </button>
    </div>
  );
}

function ReviewView({ albumId, pages, photos, selectedPageId, setSelectedPageId, refresh }) {
  const [reviewType, setReviewType] = useState("segmentation");
  const [reviewItems, setReviewItems] = useState([]);
  const [ocrRows, setOcrRows] = useState([]);
  const [selectedPhotoId, setSelectedPhotoId] = useState(null);

  const selectedPage = pages.find((page) => page.id === selectedPageId) || pages[0] || null;
  const pagePhotos = useMemo(() => photos.filter((photo) => photo.page_id === selectedPage?.id), [photos, selectedPage]);

  useEffect(() => {
    if (selectedPage && selectedPage.id !== selectedPageId) setSelectedPageId(selectedPage.id);
  }, [selectedPage, selectedPageId, setSelectedPageId]);

  useEffect(() => {
    if (!albumId) return;
    getReviewQueue(albumId, reviewType).then(setReviewItems).catch(() => setReviewItems([]));
  }, [albumId, reviewType]);

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

  async function saveBox(photoId, box, segmentationMask = null) {
    await patchBoundingBox(photoId, box, segmentationMask);
    await refresh();
  }

  async function reprocessSelected() {
    if (!selectedPhotoId) return;
    await reprocessPhoto(selectedPhotoId);
    await refresh();
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
          selectedPhotoId={selectedPhotoId}
          onSelectPhoto={setSelectedPhotoId}
          onSaveBox={saveBox}
        />
      </section>
      <section className="side-surface">
        <div className="section-title">
          <h2>OCR Text</h2>
          <button className="icon-button" type="button" disabled={!selectedPhotoId} onClick={reprocessSelected} title="Reprocess selected photo">
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
              onClick={() => item.page_id && setSelectedPageId(item.page_id)}
            >
              <span>{item.item_type}</span>
              <StatusBadge status={item.status} />
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}

function GalleryView({ photos, onOpenPhoto = () => {} }) {
  const [detail, setDetail] = useState(null);
  const [selectedVariant, setSelectedVariant] = useState("enhanced");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);

  const variantEntries = availableVariantEntries(detail);
  const selectedVariantUrl = detail?.urls?.[selectedVariant] || detail?.urls?.enhanced || detail?.urls?.original;
  const selectedVariantRecord = detail?.enhancement_applied?.variants?.[selectedVariant];

  useEffect(() => {
    if (!detail) return;
    const entries = availableVariantEntries(detail);
    const preferred = entries.find(([name]) => name === "enhanced") || entries[0];
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
        <div className="gallery-grid">
          {photos.map((photo) => (
            <button type="button" className="photo-tile" key={photo.id} onClick={() => openGalleryPhoto(photo.id)}>
              <img src={`/api/v1/photos/${photo.id}/image`} alt="" />
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

function CaptureView({ albumId, refresh }) {
  const videoRef = useRef(null);
  const [streaming, setStreaming] = useState(false);
  const [tilt, setTilt] = useState(0);
  const [batch, setBatch] = useState([]);
  const [message, setMessage] = useState("");
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    function onOrientation(event) {
      const levelTilt = Math.max(Math.abs(event.beta || 0), Math.abs(event.gamma || 0));
      setTilt(Math.round(levelTilt));
    }
    window.addEventListener("deviceorientation", onOrientation);
    navigator.serviceWorker?.addEventListener("message", (event) => {
      if (event.data?.type === "retry-upload-queue") {
        retryQueuedUploads(setMessage).then(refresh);
      }
    });
    return () => window.removeEventListener("deviceorientation", onOrientation);
  }, [refresh]);

  async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 3840 }, height: { ideal: 2160 } },
      audio: false,
    });
    videoRef.current.srcObject = stream;
    setStreaming(true);
  }

  function captureFrame() {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], `capture-${Date.now()}.jpg`, { type: "image/jpeg" });
      setBatch((current) => [...current, file]);
    }, "image/jpeg", 0.92);
  }

  async function uploadBatch() {
    if (!albumId || batch.length === 0) return;
    try {
      for (const file of batch) {
        if (!navigator.onLine) {
          await queueUpload(albumId, [file]);
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
      await queueUpload(albumId, batch);
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
          <button className="icon-button text-button" type="button" onClick={startCamera}>
            <Camera size={16} />
            Camera
          </button>
          <button className="icon-button text-button" type="button" disabled={!streaming || tilt > 10} onClick={captureFrame}>
            <Aperture size={16} />
            Capture
          </button>
          <span className="muted-text">Tilt {tilt} degrees</span>
        </div>
      </section>
      <section className="side-surface">
        <div className="section-title">
          <h2>Batch</h2>
          <Upload size={18} />
        </div>
        <input type="file" accept="image/jpeg,image/png" multiple onChange={(event) => setBatch([...batch, ...event.target.files])} />
        <button className="primary-button" type="button" disabled={!albumId || batch.length === 0} onClick={uploadBatch}>
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

export default function App() {
  const [activeTab, setActiveTab] = useState("analyze");
  const [albums, setAlbums] = useState([]);
  const [selectedAlbumId, setSelectedAlbumId] = useState(null);
  const [pages, setPages] = useState([]);
  const [photos, setPhotos] = useState([]);
  const [selectedPageId, setSelectedPageId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState("");
  const [photoViewerDetail, setPhotoViewerDetail] = useState(null);

  const selectedAlbum = albums.find((album) => album.id === selectedAlbumId) || null;

  const refreshAlbums = useCallback(async () => {
    setLoading(true);
    try {
      const nextAlbums = await listAlbums();
      setAlbums(nextAlbums);
      if (!selectedAlbumId && nextAlbums.length > 0) setSelectedAlbumId(nextAlbums[0].id);
    } finally {
      setLoading(false);
    }
  }, [selectedAlbumId]);

  const refreshAlbumData = useCallback(async () => {
    if (!selectedAlbumId) return;
    const [nextPages, nextPhotos, nextAlbums] = await Promise.all([listPages(selectedAlbumId), listPhotos(selectedAlbumId), listAlbums()]);
    setPages(nextPages);
    setPhotos(nextPhotos);
    setAlbums(nextAlbums);
    if (!selectedPageId && nextPages.length > 0) setSelectedPageId(nextPages[0].id);
  }, [selectedAlbumId, selectedPageId]);

  useEffect(() => {
    refreshAlbums().catch((error) => setNotice(error.message));
  }, [refreshAlbums]);

  useEffect(() => {
    refreshAlbumData().catch((error) => setNotice(error.message));
  }, [refreshAlbumData]);

  async function handleCreateAlbum(payload) {
    const album = await createAlbum(payload);
    setSelectedAlbumId(album.id);
    await refreshAlbums();
  }

  async function handleUpload(files) {
    await uploadFiles(selectedAlbumId, files);
    await refreshAlbumData();
  }

  async function handleAnalyze() {
    const result = await analyzeAlbum(selectedAlbumId);
    setNotice(`Queued job ${result.job_id}`);
    setTimeout(() => refreshAlbumData().catch(() => {}), 1500);
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
                onClick={() => setActiveTab(tab.id)}
                title={tab.label}
              >
                <Icon size={18} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </header>

      {notice && (
        <button type="button" className="notice" onClick={() => setNotice("")}>
          {notice}
        </button>
      )}

      {activeTab === "analyze" && (
        <OrderedAnalyzer
          onOpenPhoto={openPhoto}
          onAlbumReady={async (albumId) => {
            setSelectedAlbumId(albumId);
            const [nextAlbums, nextPages, nextPhotos] = await Promise.all([listAlbums(), listPages(albumId), listPhotos(albumId)]);
            setAlbums(nextAlbums);
            setPages(nextPages);
            setPhotos(nextPhotos);
            if (nextPages.length > 0) setSelectedPageId(nextPages[0].id);
          }}
        />
      )}
      {activeTab === "albums" && (
        <AlbumList
          albums={albums}
          loading={loading}
          selectedAlbumId={selectedAlbumId}
          pages={pages}
          photos={photos}
          onSelect={setSelectedAlbumId}
          onOpenPhoto={openPhoto}
          onCreate={handleCreateAlbum}
        />
      )}
      {activeTab === "pages" && (
        <PageGrid
          albumId={selectedAlbumId}
          pages={pages}
          selectedPageId={selectedPageId}
          onAnalyze={handleAnalyze}
          onUpload={handleUpload}
          onSelectPage={setSelectedPageId}
        />
      )}
      {activeTab === "review" && (
        <ReviewView
          albumId={selectedAlbumId}
          pages={pages}
          photos={photos}
          selectedPageId={selectedPageId}
          setSelectedPageId={setSelectedPageId}
          refresh={refreshAlbumData}
        />
      )}
      {activeTab === "gallery" && <GalleryView photos={photos} onOpenPhoto={openPhoto} />}
      {activeTab === "capture" && <CaptureView albumId={selectedAlbumId} refresh={refreshAlbumData} />}
      {photoViewerDetail && (
        <PhotoCompareModal detail={photoViewerDetail} title={`Photo ${String(photoViewerDetail.id).slice(0, 8)}`} onClose={() => setPhotoViewerDetail(null)} />
      )}
    </main>
  );
}
