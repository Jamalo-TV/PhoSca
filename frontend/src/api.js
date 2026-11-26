const jsonHeaders = { "Content-Type": "application/json" };
const sessionUrl = "/api/v1/auth/session";

async function apiFetch(url, options) {
  const headers = new Headers(options?.headers);
  if (options?.method && !["GET", "HEAD", "OPTIONS"].includes(options.method)) {
    headers.set("X-CSRF-Token", "1");
  }
  const response = await fetch(url, { ...options, headers });
  if (response.status === 401 && url !== sessionUrl) {
    window.dispatchEvent(new Event("phosca-session-expired"));
  }
  return response;
}

export async function getBrowserSession() {
  return parseResponse(await apiFetch(sessionUrl));
}

export async function signIn(apiKey) {
  return parseResponse(await apiFetch(sessionUrl, { method: "POST", headers: { "X-API-Key": apiKey } }));
}

export async function signOut() {
  return parseResponse(await apiFetch(sessionUrl, { method: "DELETE" }));
}

async function parseResponse(response) {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || payload.error || detail;
    } catch {
      detail = response.statusText;
    }
    if (typeof detail?.code === "string") {
      throw Object.assign(new Error(detail.code), { detail });
    }
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}

export async function listAlbums() {
  return parseResponse(await apiFetch("/api/v1/albums"));
}

export async function createAlbum(payload) {
  return parseResponse(await apiFetch("/api/v1/albums", { method: "POST", headers: jsonHeaders, body: JSON.stringify(payload) }));
}

export async function getAlbum(albumId) {
  return parseResponse(await apiFetch(`/api/v1/albums/${albumId}`));
}

export async function measurePageEdge(pageId, points, signal) {
  const query = new URLSearchParams();
  ["start", "middle", "end"].forEach((name, index) => {
    query.set(`${name}_x`, points[index].x);
    query.set(`${name}_y`, points[index].y);
  });
  return parseResponse(await apiFetch(`/api/v1/pages/${pageId}/edge-trace?${query}`, { signal }));
}

export async function listPages(albumId, signal) {
  return parseResponse(await apiFetch(`/api/v1/albums/${albumId}/pages`, { signal }));
}

export async function listPhotos(albumId, { includeRejected = false } = {}) {
  const query = includeRejected ? "?include_rejected=true" : "";
  return parseResponse(await apiFetch(`/api/v1/albums/${albumId}/photos${query}`));
}

export function hasPhotoCrop(photo) {
  const variants = photo.enhancement_applied?.variants;
  return Boolean(photo.original_storage_path || variants?.source_corrected?.path || variants?.original?.path || photo.storage_path);
}

export async function analyzeAlbumNow(albumId, pageIds = null) {
  return parseResponse(
    await apiFetch(`/api/v1/albums/${albumId}/analyze-now`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ page_ids: pageIds }),
    })
  );
}

export async function getPhoto(photoId) {
  return parseResponse(await apiFetch(`/api/v1/photos/${photoId}`));
}

export async function getPresets() {
  return parseResponse(await apiFetch("/api/v1/presets"));
}

export async function getPhotoVariants(photoId) {
  return parseResponse(await apiFetch(`/api/v1/photos/${photoId}/variants`));
}

export async function getPageOcr(pageId) {
  return parseResponse(await apiFetch(`/api/v1/pages/${pageId}/ocr`));
}

export async function getReviewQueue(albumId, type, { includeRejected = true, photoId, signal } = {}) {
  const params = new URLSearchParams({ type });
  if (albumId) params.set("album_id", albumId);
  if (type === "geometry" && !includeRejected) params.set("include_rejected", "false");
  if (type === "geometry" && photoId) params.set("photo_id", photoId);
  return parseResponse(await apiFetch(`/api/v1/review/queue?${params}`, { signal }));
}

export async function submitReviewEvent(payload) {
  return parseResponse(await apiFetch("/api/v1/review/events", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(payload),
  }));
}

export async function getReviewProgress() {
  return parseResponse(await apiFetch("/api/v1/review/progress"));
}

export async function patchBoundingBox(photoId, boundingBox, segmentationMask = null, confirmPrintOutline = false,
  albumPageMarkReviewId = null) {
  const body = segmentationMask ? { bounding_box: boundingBox, segmentation_mask: segmentationMask } : { bounding_box: boundingBox };
  if (confirmPrintOutline) body.confirm_print_outline = true;
  if (albumPageMarkReviewId) body.album_page_mark_review_id = albumPageMarkReviewId;
  return parseResponse(
    await apiFetch(`/api/v1/photos/${photoId}/bounding-box`, {
      method: "PATCH",
      headers: jsonHeaders,
      body: JSON.stringify(body),
    })
  );
}

export async function addManualPhoto(pageId, requestId, warpQuad, printSize, adjacentSuggestion, deferredProposal) {
  return parseResponse(await apiFetch(`/api/v1/pages/${pageId}/manual-photos`, {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({
      request_id: requestId, warp_quad: warpQuad,
      print_size_mm: printSize, adjacent_suggestion: adjacentSuggestion,
      deferred_proposal_id: deferredProposal?.id, photo_count_revision: deferredProposal?.revision,
    }),
  }));
}

export async function patchOcr(ocrId, payload) {
  return parseResponse(
    await apiFetch(`/api/v1/ocr/${ocrId}`, {
      method: "PATCH",
      headers: jsonHeaders,
      body: JSON.stringify(payload),
    })
  );
}

export async function reprocessPhoto(photoId, preset = "auto") {
  const params = new URLSearchParams({ preset });
  return parseResponse(await apiFetch(`/api/v1/photos/${photoId}/reprocess?${params}`, { method: "POST" }));
}

export async function search(q) {
  return parseResponse(await apiFetch(`/api/v1/search?q=${encodeURIComponent(q)}`));
}

export async function uploadFiles(albumId, files) {
  const form = new FormData();
  files.forEach((file) => form.append("files", file, file.name));
  return parseResponse(await apiFetch(`/api/v1/albums/${albumId}/pages/upload`, { method: "POST", headers: { "X-Upload-Retry": "1" }, body: form }));
}

export async function uploadFileChunked(albumId, file, onProgress) {
  const chunkSize = 2 * 1024 * 1024;
  const uploadId = crypto.randomUUID();
  let offset = 0;
  let finalResponse = null;
  while (offset < file.size) {
    const end = Math.min(file.size, offset + chunkSize) - 1;
    const chunk = file.slice(offset, end + 1);
    const form = new FormData();
    form.append("file", chunk, file.name);
    finalResponse = await parseResponse(
      await apiFetch(`/api/v1/albums/${albumId}/pages/upload/chunk`, {
        method: "POST",
        headers: {
          "Content-Range": `bytes ${offset}-${end}/${file.size}`,
          "X-Upload-ID": uploadId,
          "X-Filename": file.name,
          "X-Upload-Retry": "1",
        },
        body: form,
      })
    );
    offset = end + 1;
    onProgress?.(offset / file.size);
  }
  return finalResponse;
}

export async function listSegmentationTrainingImages() {
  return parseResponse(await apiFetch("/api/v1/training/segmentation"));
}

export async function uploadSegmentationTrainingImages(files) {
  const form = new FormData();
  files.forEach((file) => form.append("files", file, file.name));
  return parseResponse(await apiFetch("/api/v1/training/segmentation/images", { method: "POST", body: form }));
}

export async function deleteSegmentationTrainingImage(filename) {
  return parseResponse(await apiFetch(`/api/v1/training/segmentation/images/${encodeURIComponent(filename)}`, { method: "DELETE" }));
}

export async function getSegmentationTrainingLabels(filename) {
  return parseResponse(await apiFetch(`/api/v1/training/segmentation/images/${encodeURIComponent(filename)}/labels`));
}

export async function detectSegmentationTrainingLabels(filename) {
  return parseResponse(await apiFetch(`/api/v1/training/segmentation/images/${encodeURIComponent(filename)}/detect`, { method: "POST" }));
}

export async function saveSegmentationTrainingLabels(filename, polygons) {
  return parseResponse(
    await apiFetch(`/api/v1/training/segmentation/images/${encodeURIComponent(filename)}/labels`, {
      method: "PUT",
      headers: jsonHeaders,
      body: JSON.stringify({ polygons }),
    })
  );
}

export async function prepareSegmentationTrainingDataset() {
  return parseResponse(await apiFetch("/api/v1/training/segmentation/prepare", { method: "POST" }));
}
