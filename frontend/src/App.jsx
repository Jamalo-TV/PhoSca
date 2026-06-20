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
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  analyzeAlbum,
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
  { id: "albums", label: "Albums", icon: GalleryHorizontal },
  { id: "pages", label: "Pages", icon: FileImage },
  { id: "review", label: "Review", icon: Pencil },
  { id: "gallery", label: "Gallery", icon: Aperture },
  { id: "capture", label: "Capture", icon: Camera },
];

function StatusBadge({ status }) {
  const tone = {
    completed: "status status-green",
    uploaded: "status status-blue",
    queued: "status status-blue",
    processing: "status status-blue",
    review_needed: "status status-yellow",
    failed: "status status-red",
    pending: "status status-blue",
  }[status] || "status";
  return <span className={tone}>{status?.replace("_", " ") || "unknown"}</span>;
}

function SafeText({ text, className = "" }) {
  return <span className={className} dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(text || "") }} />;
}

function EmptyState({ label }) {
  return <div className="empty-state">{label}</div>;
}

function AlbumList({ albums, selectedAlbumId, onSelect, onCreate, loading }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

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

function ReviewCanvas({ page, photos, onSaveBox, selectedPhotoId, onSelectPhoto }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [draftBoxes, setDraftBoxes] = useState({});
  const dragRef = useRef(null);

  useEffect(() => {
    setDraftBoxes(Object.fromEntries(photos.map((photo) => [photo.id, photo.bounding_box])));
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
      const box = draftBoxes[photo.id] || photo.bounding_box;
      const x = box.x1 * canvas.width;
      const y = box.y1 * canvas.height;
      const width = (box.x2 - box.x1) * canvas.width;
      const height = (box.y2 - box.y1) * canvas.height;
      context.strokeStyle = photo.id === selectedPhotoId ? "#0f766e" : "#f59e0b";
      context.lineWidth = photo.id === selectedPhotoId ? 4 : 2;
      context.strokeRect(x, y, width, height);
      context.fillStyle = "#ffffff";
      [[x, y], [x + width, y], [x + width, y + height], [x, y + height]].forEach(([cx, cy]) => {
        context.fillRect(cx - 5, cy - 5, 10, 10);
        context.strokeRect(cx - 5, cy - 5, 10, 10);
      });
    });
  }, [draftBoxes, page, photos, selectedPhotoId]);

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
    for (const photo of photos) {
      const box = draftBoxes[photo.id] || photo.bounding_box;
      const corners = [
        ["x1", "y1", box.x1, box.y1],
        ["x2", "y1", box.x2, box.y1],
        ["x2", "y2", box.x2, box.y2],
        ["x1", "y2", box.x1, box.y2],
      ];
      const corner = corners.find(([, , x, y]) => Math.hypot(point.x - x, point.y - y) < 0.035);
      if (corner) {
        onSelectPhoto(photo.id);
        dragRef.current = { photoId: photo.id, xKey: corner[0], yKey: corner[1] };
        canvasRef.current.setPointerCapture(event.pointerId);
        return;
      }
      if (point.x >= box.x1 && point.x <= box.x2 && point.y >= box.y1 && point.y <= box.y2) {
        onSelectPhoto(photo.id);
      }
    }
  }

  function onPointerMove(event) {
    if (!dragRef.current) return;
    const point = pointToBox(event);
    const { photoId, xKey, yKey } = dragRef.current;
    setDraftBoxes((current) => {
      const next = { ...(current[photoId] || photos.find((photo) => photo.id === photoId)?.bounding_box) };
      next[xKey] = Math.max(0, Math.min(1, point.x));
      next[yKey] = Math.max(0, Math.min(1, point.y));
      if (next.x1 > next.x2) [next.x1, next.x2] = [next.x2, next.x1];
      if (next.y1 > next.y2) [next.y1, next.y2] = [next.y2, next.y1];
      return { ...current, [photoId]: next };
    });
  }

  function onPointerUp(event) {
    if (dragRef.current) {
      canvasRef.current.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
  }

  async function saveSelected() {
    if (!selectedPhotoId || !draftBoxes[selectedPhotoId]) return;
    await onSaveBox(selectedPhotoId, draftBoxes[selectedPhotoId]);
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
        Save Box
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

  async function saveBox(photoId, box) {
    await patchBoundingBox(photoId, box);
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

function GalleryView({ photos }) {
  const [detail, setDetail] = useState(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);

  async function runSearch(event) {
    event.preventDefault();
    if (!query.trim()) {
      setResults([]);
      return;
    }
    setResults(await search(query.trim()));
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
              <button type="button" key={`${result.page_id}-${result.text_content}`} onClick={() => result.photo_id && getPhoto(result.photo_id).then(setDetail)}>
                <SafeText text={result.highlight} />
              </button>
            ))}
          </div>
        )}
        <div className="gallery-grid">
          {photos.map((photo) => (
            <button type="button" className="photo-tile" key={photo.id} onClick={() => getPhoto(photo.id).then(setDetail)}>
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
            <img className="detail-image" src={`/api/v1/photos/${detail.id}/image`} alt="" />
            <span>ID {detail.id}</span>
            <span>pHash {detail.phash || "pending"}</span>
            <span>Duplicate {detail.is_duplicate_of || "no"}</span>
            <span>Aspect {detail.aspect_ratio?.toFixed(2) || "unknown"}</span>
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
  const [activeTab, setActiveTab] = useState("albums");
  const [albums, setAlbums] = useState([]);
  const [selectedAlbumId, setSelectedAlbumId] = useState(null);
  const [pages, setPages] = useState([]);
  const [photos, setPhotos] = useState([]);
  const [selectedPageId, setSelectedPageId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState("");

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

      {activeTab === "albums" && (
        <AlbumList
          albums={albums}
          loading={loading}
          selectedAlbumId={selectedAlbumId}
          onSelect={setSelectedAlbumId}
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
      {activeTab === "gallery" && <GalleryView photos={photos} />}
      {activeTab === "capture" && <CaptureView albumId={selectedAlbumId} refresh={refreshAlbumData} />}
    </main>
  );
}

