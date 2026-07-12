import { openDB } from "idb";
import { uploadFileChunked, uploadFiles } from "./api.js";

const DB_NAME = "album-digitizer";
const STORE = "upload-queue";

async function db() {
  return openDB(DB_NAME, 1, {
    upgrade(database) {
      database.createObjectStore(STORE, { keyPath: "id" });
    },
  });
}

export async function queueUpload(albumId, files) {
  const database = await db();
  for (const file of files) {
    await database.put(STORE, {
      id: crypto.randomUUID(), albumId, file, createdAt: Date.now(),
      status: "queued", attempts: 0, error: null,
    });
  }
  if ("serviceWorker" in navigator && "SyncManager" in window) {
    const registration = await navigator.serviceWorker.ready;
    await registration.sync.register("retry-upload-queue");
  }
}

export async function listQueuedUploads() {
  return (await db()).getAll(STORE);
}

export async function removeQueuedUpload(id) {
  return (await db()).delete(STORE, id);
}

export async function retryQueuedUploads(onMessage) {
  const database = await db();
  const uploads = await database.getAll(STORE);
  for (const upload of uploads) {
    try {
      await database.put(STORE, { ...upload, status: "uploading", attempts: (upload.attempts || 0) + 1, error: null });
      if (upload.file.size > 20 * 1024 * 1024) {
        await uploadFileChunked(upload.albumId, upload.file);
      } else {
        await uploadFiles(upload.albumId, [upload.file]);
      }
      await database.delete(STORE, upload.id);
      onMessage?.("Auf PC gespeichert.", upload.id, "verified");
    } catch (error) {
      await database.put(STORE, { ...upload, status: "failed", attempts: (upload.attempts || 0) + 1, error: error.message });
      onMessage?.("Upload wartet auf Netzwerk.", upload.id, "failed");
    }
  }
}
