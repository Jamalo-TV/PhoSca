const jsonHeaders = { "Content-Type": "application/json" };

async function parseResponse(response) {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || payload.error || detail;
    } catch {
      detail = response.statusText;
    }
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}

export async function listAlbums() {
  return parseResponse(await fetch("/api/v1/albums"));
}

export async function createAlbum(payload) {
  return parseResponse(await fetch("/api/v1/albums", { method: "POST", headers: jsonHeaders, body: JSON.stringify(payload) }));
}

export async function getAlbum(albumId) {
  return parseResponse(await fetch(`/api/v1/albums/${albumId}`));
}

export async function listPages(albumId) {
  return parseResponse(await fetch(`/api/v1/albums/${albumId}/pages`));
}

export async function listPhotos(albumId) {
  return parseResponse(await fetch(`/api/v1/albums/${albumId}/photos`));
}

export async function analyzeAlbum(albumId, pageIds = null) {
  return parseResponse(
    await fetch(`/api/v1/albums/${albumId}/analyze`, {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ page_ids: pageIds }),
    })
  );
}

export async function getPhoto(photoId) {
  return parseResponse(await fetch(`/api/v1/photos/${photoId}`));
}

export async function getPageOcr(pageId) {
  return parseResponse(await fetch(`/api/v1/pages/${pageId}/ocr`));
}

export async function getReviewQueue(albumId, type) {
  const params = new URLSearchParams({ type });
  if (albumId) params.set("album_id", albumId);
  return parseResponse(await fetch(`/api/v1/review/queue?${params}`));
}

export async function patchBoundingBox(photoId, boundingBox) {
  return parseResponse(
    await fetch(`/api/v1/photos/${photoId}/bounding-box`, {
      method: "PATCH",
      headers: jsonHeaders,
      body: JSON.stringify({ bounding_box: boundingBox }),
    })
  );
}

export async function patchOcr(ocrId, payload) {
  return parseResponse(
    await fetch(`/api/v1/ocr/${ocrId}`, {
      method: "PATCH",
      headers: jsonHeaders,
      body: JSON.stringify(payload),
    })
  );
}

export async function reprocessPhoto(photoId) {
  return parseResponse(await fetch(`/api/v1/photos/${photoId}/reprocess`, { method: "POST" }));
}

export async function search(q) {
  return parseResponse(await fetch(`/api/v1/search?q=${encodeURIComponent(q)}`));
}

export async function uploadFiles(albumId, files) {
  const form = new FormData();
  files.forEach((file) => form.append("files", file, file.name));
  return parseResponse(await fetch(`/api/v1/albums/${albumId}/pages/upload`, { method: "POST", body: form }));
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
      await fetch(`/api/v1/albums/${albumId}/pages/upload/chunk`, {
        method: "POST",
        headers: {
          "Content-Range": `bytes ${offset}-${end}/${file.size}`,
          "X-Upload-ID": uploadId,
          "X-Filename": file.name,
        },
        body: form,
      })
    );
    offset = end + 1;
    onProgress?.(offset / file.size);
  }
  return finalResponse;
}

