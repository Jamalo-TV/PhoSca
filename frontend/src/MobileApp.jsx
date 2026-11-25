import { Camera, Check, ChevronLeft, CircleHelp, CloudOff, Images, ListChecks, RefreshCw, Save, Upload, WandSparkles, ZoomIn } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createAlbum, getPhoto, getReviewProgress, getReviewQueue, hasPhotoCrop, listAlbums, listPages, listPhotos, patchBoundingBox, submitReviewEvent, uploadFileChunked, uploadFiles } from "./api.js";
import { resamplePerspective, samplePreviewSource } from "./cropPreview.js";
import {
  adjacentPrintHint, clippedOverlapHint, clippedOverlapReason, CropOrientationField, CropVariants, fragmentOutlineHint,
  fragmentOutlineReason, getCropModes, getFragmentOutline,
  getPrintEdgeReviewHint,
} from "./CropVariants.jsx";
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
  const videoRef = useRef(null); const streamRef = useRef(null);
  const refreshQueue = useCallback(() => listQueuedUploads(uploadQueueScope).then(setQueue), [uploadQueueScope]);
  useEffect(() => { refreshQueue(); }, [refreshQueue]);
  useEffect(() => () => streamRef.current?.getTracks().forEach(track => track.stop()), []);

  async function startCamera() {
    setCameraError("");
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) return setCameraError("Die Live-Kamera benötigt die HTTPS-Adresse dieser Seite.");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 4096 }, height: { ideal: 3072 } }, audio: false });
      streamRef.current?.getTracks().forEach(track => track.stop()); streamRef.current = stream;
      videoRef.current.srcObject = stream; await videoRef.current.play(); setStreaming(true);
    } catch (error) {
      setCameraError(error.name === "NotAllowedError" ? "Kamerazugriff wurde nicht erlaubt." : `Kamera konnte nicht gestartet werden: ${error.message}`);
    }
  }

  function captureInBrowser() {
    const video = videoRef.current; if (!video?.videoWidth || busy) return;
    const canvas = document.createElement("canvas"); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext("2d", { alpha: false }).drawImage(video, 0, 0);
    canvas.toBlob(blob => {
      if (!blob) return; if (lastCapture) URL.revokeObjectURL(lastCapture); setLastCapture(URL.createObjectURL(blob));
      receive([new File([blob], `capture-${new Date().toISOString().replaceAll(":", "-")}.jpg`, { type: "image/jpeg", lastModified: Date.now() })]);
      navigator.vibrate?.(35);
    }, "image/jpeg", 0.96);
  }

  async function addBatch() {
    const name = prompt("Name des neuen Aufnahme-Batches", `Album ${new Date().toLocaleDateString("de-CH")}`);
    if (!name) return; const album = await createAlbum({ name, description: "Mobiler Capture-Batch" }); setActiveAlbumId(album.id); await reload();
  }

  async function receive(files) {
    if (!activeAlbumId || !files.length) return; setBusy(true); setMessage("");
    for (const file of files) {
      try {
        if (!navigator.onLine) throw new Error("offline");
        const result = file.size > 20 * 1024 * 1024 ? await uploadFileChunked(activeAlbumId, file) : await uploadFiles(activeAlbumId, [file]);
        if (!result?.pages?.[0]?.verified) throw new Error("Serverbestätigung fehlt");
        setMessage(result.analysis_queued ? "Auf PC gespeichert. Fotos werden automatisch ausgeschnitten." : "Auf PC gespeichert und geprüft.");
      } catch {
        await queueUpload(activeAlbumId, [file], uploadQueueScope); setMessage("Lokal gesichert – wartet auf Upload.");
      }
    }
    await refreshQueue(); await reload(); setBusy(false);
  }

  async function retry() { setBusy(true); await retryQueuedUploads(setMessage, uploadQueueScope); await refreshQueue(); await reload(); setBusy(false); }

  return <div className="mobile-shell"><MobileHeader title="Seiten aufnehmen" onBack={goHome} /><main className="mobile-content">
    <section className="mobile-card batch-picker"><label>Album / Batch</label><div><select value={activeAlbumId || ""} onChange={e => setActiveAlbumId(e.target.value)}><option value="" disabled>Batch auswählen</option>{albums.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}</select><button className="mobile-icon light" onClick={addBatch}>+</button></div></section>
    <section className="capture-stage"><div className={`live-camera ${streaming ? "streaming" : ""}`}><video ref={videoRef} autoPlay playsInline muted />{!streaming && <div className="page-guide"><Camera size={44}/><strong>Kamera direkt im Browser</strong><span>Die Aufnahme landet automatisch auf deinem PC.</span></div>}{streaming && <div className="camera-guide"><span>Ganze Albumseite mit etwas Rand ausrichten</span></div>}{lastCapture && <img className="last-capture" src={lastCapture} alt="Letzte Aufnahme" />}</div>
      {!streaming ? <button className="capture-button" disabled={!activeAlbumId} onClick={startCamera}><Camera/><span>Kamera starten</span></button> : <button className={`shutter-button ${busy ? "busy" : ""}`} disabled={busy || !activeAlbumId} onClick={captureInBrowser} aria-label="Foto jetzt aufnehmen"><span /></button>}
      {cameraError && <p className="camera-error">{cameraError}</p>}<label className="secondary-capture"><Images size={19}/> Aus Galerie wählen<input type="file" accept="image/jpeg,image/png" multiple disabled={!activeAlbumId || busy} onChange={e => receive([...e.target.files])}/></label>
    </section>{message && <div className="mobile-message"><Check size={18}/>{message}</div>}
    <section className="queue-card"><div><strong>Upload-Warteschlange</strong><span>{queue.length ? `${queue.length} ausstehend` : "Alles auf dem PC gespeichert"}</span></div>{queue.length > 0 && <button onClick={retry} disabled={busy}><RefreshCw size={18}/> Erneut versuchen</button>}{queue.map(item => <div className="queue-row" key={item.id}><CloudOff/><span><b>{item.file.name}</b><small>{item.status} · Versuch {item.attempts || 0}</small></span><button onClick={() => removeQueuedUpload(item.id).then(refreshQueue)}>×</button></div>)}</section>
  </main></div>;
}

function DetectionOverlay({ item, editable = false, onSaved }) {
  const sourceBox = item.payload?.bounding_box;
  const detections = sourceBox ? [{ bounding_box: sourceBox, photo_id: item.payload?.photo_id }] : (item.payload?.detections || []);
  const visiblePolygon = sanitizeReviewPolygon(item.payload?.visible_polygon || item.payload?.polygon);
  const storedMeasurement = item.payload?.print_size_mm ?? item.payload?.segmentation_mask?.print_size_mm;
  const storedOrientation = item.payload?.segmentation_mask?.orientation_degrees ?? null;
  const fragmentReview = item.payload?.confidence != null && item.payload?.review_reasons?.includes(fragmentOutlineReason);
  const fragmentOutline = fragmentReview && getFragmentOutline(item.payload?.segmentation_mask);
  const printEdgeReviewHint = getPrintEdgeReviewHint(item.payload?.crop_modes?.full_print);
  const adjacentOutline = item.payload?.confidence != null
    && cyclicReviewQuad(sanitizeReviewPolygon(item.payload?.segmentation_mask?.adjacent_print_suggestion));
  const scallopedAlternate = item.payload?.confidence != null && item.payload?.segmentation_mask?.scalloped_alternate_is_inner
    && cyclicReviewQuad(sanitizeReviewPolygon(item.payload?.segmentation_mask?.scalloped_alternate_polygon));
  const savedQuad = sanitizeReviewPolygon(item.payload?.segmentation_mask?.warp_quad);
  const measuredQuad = storedMeasurement && (savedQuad || sanitizeReviewPolygon(item.payload?.warp_quad));
  const initialOuter = () => isValidQuad(measuredQuad) ? measuredQuad : initialQuad(
    item.payload?.confidence === null && savedQuad ? savedQuad : item.payload?.warp_quad,
    visiblePolygon,
    sourceBox,
  );
  const [quad, setQuad] = useState(initialOuter); const [saving, setSaving] = useState(false);
  const [usingOutlineSuggestion, setUsingOutlineSuggestion] = useState(false);
  const [editingPreview, setEditingPreview] = useState(false);
  const [measurementFields, setMeasurementFields] = useState(() => printSizeInputs(storedMeasurement));
  const [orientation, setOrientation] = useState(storedOrientation);
  const [savedOrientation, setSavedOrientation] = useState(storedOrientation);
  const [savedCropUrl, setSavedCropUrl] = useState(""); const [saveError, setSaveError] = useState("");
  const [magnifierEnabled, setMagnifierEnabled] = useState(true);
  const [magnifier, setMagnifier] = useState({ visible: false, x: 0, y: 0 });
  useEffect(() => {
    setQuad(initialOuter());
    setUsingOutlineSuggestion(false);
    setEditingPreview(false);
    setMeasurementFields(printSizeInputs(storedMeasurement));
    setOrientation(storedOrientation);
    setSavedOrientation(storedOrientation);
    setSavedCropUrl("");
    setSaveError("");
  }, [item.id]);
  const box = quadToBox(quad);
  const validQuad = isValidQuad(quad);
  const { size, error: sizeError } = parsePrintSize(measurementFields);
  function moveCorner(event, cornerIndex) {
    if (!editable || !quad) return; event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
    const svg = event.currentTarget.ownerSVGElement; const rect = svg.getBoundingClientRect();
    const move = e => {
      e.preventDefault();
      const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)); const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      setMagnifier({ visible: magnifierEnabled, x, y });
      setEditingPreview(true);
      setSavedCropUrl(""); setSaveError("");
      setQuad(current => current.map((point,index) => index === cornerIndex ? { x, y } : point));
    };
    const initialX = quad[cornerIndex].x; const initialY = quad[cornerIndex].y;
    setMagnifier({ visible: magnifierEnabled, x: initialX, y: initialY });
    const up = () => { setMagnifier(current => ({ ...current, visible: false })); removeEventListener("pointermove", move); removeEventListener("pointerup", up); };
    addEventListener("pointermove", move); addEventListener("pointerup", up, { once: true });
  }
  async function save() {
    if (!validQuad || sizeError) return; setSaving(true); setSaveError("");
    try {
      const saved = await patchBoundingBox(item.payload.photo_id, box, {
        polygon: quad,
        visible_polygon: usingOutlineSuggestion ? quad : (visiblePolygon || quad),
        warp_quad: quad,
        print_size_mm: size,
        ...(orientation !== savedOrientation ? { orientation_degrees: orientation } : {}),
        source: "manual_mobile_quad",
      });
      const savedDegrees = saved.segmentation_mask?.orientation_degrees ?? null;
      setOrientation(savedDegrees);
      setSavedOrientation(savedDegrees);
      const nextUrl=`/api/v1/photos/${item.payload.photo_id}/image?variant=original&v=${Date.now()}`;
      setSavedCropUrl(nextUrl); await onSaved?.({ bounding_box: box, polygon: quad });
    } catch (error) { setSaveError(error.message || "Ausschnitt konnte nicht gespeichert werden."); }
    finally { setSaving(false); }
  }
  const shown = box ? [{ bounding_box: box, photo_id: item.payload?.photo_id }] : detections;
  const overlayLabel = item.payload?.detection_count == null ? "1 Crop" : `${item.payload.detection_count} ${item.payload.detection_count === 1 ? "Print" : "Prints"}`;
  const lensPosition = value => ((0.5 - 3.6 * value) / (1 - 3.6)) * 100;
  const lensX=Math.max(20,Math.min(80,magnifier.x*100+(magnifier.x>0.5?-24:24)));
  const lensY=Math.max(20,Math.min(80,magnifier.y*100+(magnifier.y<0.3?25:-25)));
  const uncertainLine = usingOutlineSuggestion && fragmentOutline && (
    <polyline points={fragmentOutline.weakEdge.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")}
      fill="none" stroke="#a21caf" strokeWidth="6" strokeDasharray="16 9" />
  );
  const alternateOutline = adjacentOutline || scallopedAlternate;
  const alternateOverlay = alternateOutline && (
    <polygon points={alternateOutline.map(({ x, y }) => `${x * 1000},${y * 1000}`).join(" ")}
      fill="none" stroke="#a21caf" strokeWidth="6" strokeDasharray="16 9" />
  );
  return <><div className={`review-image overlay-image ${validQuad ? "" : "invalid-quad"}`} onContextMenu={event=>event.preventDefault()}><div className="review-image-frame"><img src={item.preview_url} alt="Zu prüfende Albumseite" draggable="false"/><svg viewBox="0 0 1000 1000" preserveAspectRatio="none">{editable && quad ? <g>{visiblePolygon && <polygon points={visiblePolygon.map(point=>`${point.x*1000},${point.y*1000}`).join(" ")} fill="none" stroke="#f59e0b" strokeWidth="5" strokeDasharray="14 9"/>}<polygon className="editable-quad" points={quad.map(point=>`${point.x*1000},${point.y*1000}`).join(" ")}/>{uncertainLine}{alternateOverlay}{quad.map((point,index)=><g key={index}><circle className="drag-handle" cx={point.x*1000} cy={point.y*1000} r="25" onPointerDown={event=>moveCorner(event,index)}/><text className="corner-label" x={point.x*1000+28} y={point.y*1000-22}>{index+1}</text></g>)}</g> : shown.map((d, i) => { const b=d.bounding_box; return <g key={d.photo_id || i}><rect x={b.x1*1000} y={b.y1*1000} width={(b.x2-b.x1)*1000} height={(b.y2-b.y1)*1000}/><text x={b.x1*1000+12} y={b.y1*1000+38}>{i+1}</text></g>;})}</svg>{magnifier.visible && <div className="crop-magnifier" style={{ left:`${lensX}%`, top:`${lensY}%`, backgroundImage:`url(${item.preview_url})`, backgroundPosition:`${lensPosition(magnifier.x)}% ${lensPosition(magnifier.y)}%` }}><i /></div>}</div><span>{validQuad ? overlayLabel : "Ecken kreuzen sich"}</span></div>
    {editable && <>
      {item.payload?.confidence != null && item.payload?.review_reasons?.includes(clippedOverlapReason) && (
        <p className="quad-hint">{clippedOverlapHint}</p>
      )}
      {fragmentReview && <p className="quad-hint">{fragmentOutlineHint}</p>}
      {printEdgeReviewHint && <p className="quad-hint">{printEdgeReviewHint}</p>}
      {adjacentOutline && <p className="quad-hint">{adjacentPrintHint} Einen eigenen Abzug kannst du im Desktop-Editor ergänzen.</p>}
      {scallopedAlternate && <p className="quad-hint">Prüfe den gestrichelten Vorschlag am Original. Wähle «Prints zusammen», falls dort ein eigener Abzug aufliegt.</p>}
      {fragmentOutline && (
        <button className="magnifier-toggle" type="button" disabled={saving} onClick={() => {
          const suggestion = cyclicReviewQuad(fragmentOutline.quad);
          if (!suggestion) return;
          setQuad(suggestion);
          setUsingOutlineSuggestion(true);
          setEditingPreview(true);
          setSavedCropUrl("");
          setSaveError("");
        }}>Aussenumriss in den Editor laden</button>
      )}
      {uncertainLine && <p className="quad-hint">Prüfe besonders die gestrichelte Papierkante.</p>}
      <div className="quad-hint">Vier nummerierte Ecken frei verschieben · danach wird das Foto geradegerichtet</div>
      <button className={`magnifier-toggle ${magnifierEnabled ? "active" : ""}`} onClick={()=>setMagnifierEnabled(value=>!value)}><ZoomIn size={18}/>{magnifierEnabled ? "Lupe beim Ziehen aktiv" : "Lupe einschalten"}</button>
      <PrintSizeFields
        value={measurementFields}
        onChange={(fields) => {
          setMeasurementFields(fields);
          setEditingPreview(true);
          setSavedCropUrl("");
          setSaveError("");
        }}
        disabled={saving}
        error={sizeError}
      />
      <CropOrientationField value={orientation} disabled={saving} onChange={(value) => {
        setOrientation(value);
        setEditingPreview(true);
        setSavedCropUrl("");
        setSaveError("");
      }} />
      <button className="save-crop" onClick={save} disabled={saving || !validQuad || Boolean(sizeError)}>
        <Save size={18}/>{saving ? "Richtet das Bild …" : "Viereck speichern & geraderichten"}
      </button>
      {saveError && <p className="crop-save-error">{saveError}</p>}
      {quad && <LiveCropPreview src={item.preview_url} quad={quad} savedCropUrl={savedCropUrl}
        outlinePreviewUrl={editingPreview ? null : item.payload?.variant_urls?.full_print_preview}
        measuredAspect={size ? size.width / size.height : null} orientation={orientation} />}
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

function drawQuadPreview(canvas, { image, pixels }, quad, measuredAspect, orientation) {
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
  const aspect = measuredAspect || rawWidth / rawHeight;
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

function LiveCropPreview({ src, quad, savedCropUrl, outlinePreviewUrl, measuredAspect, orientation }) {
  const canvasRef = useRef(null);
  const [sample, setSample] = useState(null);
  const previewUrl = savedCropUrl || outlinePreviewUrl;
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
      if (canvasRef.current) drawQuadPreview(canvasRef.current, sample, quad, measuredAspect, orientation);
    });
    return () => cancelAnimationFrame(frame);
  }, [sample, src, quad, measuredAspect, orientation, previewUrl]);
  const title = savedCropUrl ? "Gerichteter Ausschnitt" : outlinePreviewUrl ? "Abzugsvorschau zur Prüfung" : "Live-Ausschnitt";
  const detail = savedCropUrl ? "Perspektive wurde aus den vier Ecken korrigiert" : outlinePreviewUrl
    ? "Vorschau folgt der gezackten Papierkante" : "Vorschau folgt allen vier Ecken live";
  return <section className={`live-crop-preview ${savedCropUrl ? "rectified" : ""}`}>
    <header><ZoomIn size={17}/><div><strong>{title}</strong><span>{detail}</span></div></header>
    {previewUrl ? <img src={previewUrl} alt={title} /> : <canvas ref={canvasRef}/>}
  </section>;
}

function CropReview({ albumId, goHome }) {
  const [type, setType] = useState("segmentation"); const [items, setItems] = useState([]); const [index, setIndex] = useState(0); const [message, setMessage] = useState("");
  const [answerError, setAnswerError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const leavingRef=useRef(false); const goHomeRef=useRef(goHome);
  const load = useCallback(async () => { setItems(await getReviewQueue(albumId || undefined, type)); setIndex(0); }, [albumId, type]);
  useEffect(() => { load().catch(e => setMessage(e.message)); }, [load]); const item = items[index];
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
  async function record(value) {
    if (!item || submitting) return;
    setSubmitting(true);
    setAnswerError("");
    const metadata = { page_id: item.page_id, ui: "mobile-v2" };
    if (type === "segmentation") metadata.photo_count_revision = item.payload.photo_count_revision;
    try {
      await submitReviewEvent({
        task_type: type, task_id: item.id, answer: value,
        reason_code: value === "correct" ? null : value,
        idempotency_key: crypto.randomUUID(), metadata,
      });
      setMessage("Antwort gespeichert");
      setIndex((current) => current + 1);
    } catch (error) {
      setAnswerError(error.message);
      await load().catch(() => {});
    } finally {
      setSubmitting(false);
    }
  }
  return <div className="mobile-shell crop-review-shell"><MobileHeader title="Cropping prüfen" onBack={leaveReview}/><main className="mobile-content"><div className="review-top"><div><span>Kurze Sitzung</span><strong>{Math.min(index + 1, items.length || 5)} / {Math.min(items.length, 5) || 5}</strong></div><div className="review-progress"><i style={{width:`${Math.min(100, index / Math.max(1, Math.min(items.length,5)) * 100)}%`}}/></div></div>
    <div className="task-switch"><button className={type === "segmentation" ? "active" : ""} onClick={() => setType("segmentation")} disabled={submitting}>Printanzahl</button><button className={type === "geometry" ? "active" : ""} onClick={() => setType("geometry")} disabled={submitting}>Crop-Qualität</button></div>
    {answerError && <p role="alert">{answerError}</p>}
    {item ? <><DetectionOverlay item={item} editable={type === "geometry"} onSaved={() => setMessage("Vier Ecken gespeichert und Perspektive korrigiert")}/><section className="answer-grid">{cropAnswers[type].map(([value,label]) => <button key={value} className={value === "correct" ? "good" : ""} disabled={submitting} onClick={() => record(value)}>{label}</button>)}</section></> : <section className="review-done"><Check size={48}/><h2>{message || "Keine Aufgaben offen"}</h2><p>Neue Aufgaben erscheinen nach der Bildanalyse automatisch.</p><button onClick={load}><RefreshCw/> Neu laden</button></section>}{item && <button className="skip-task" disabled={submitting} onClick={() => setIndex(i => i + 1)}>Überspringen</button>}
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
        {detail && (getCropModes(detail) ? <CropVariants photo={detail} /> : (
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
