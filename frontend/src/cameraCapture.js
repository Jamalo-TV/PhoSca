const imageExtensions = { "image/jpeg": "jpg", "image/png": "png" };

export async function captureCameraImage(video, jpegQuality) {
  const track = video?.srcObject?.getVideoTracks()[0];
  if (track?.readyState !== "live") throw new Error("camera_not_ready");

  let blob;
  if (typeof globalThis.ImageCapture === "function") {
    try {
      const photo = await new globalThis.ImageCapture(track).takePhoto();
      if (photo?.size && Object.hasOwn(imageExtensions, photo.type)) blob = photo;
    } catch {
      // Some browsers expose ImageCapture even when the active camera cannot provide a still.
    }
  }
  if (track.readyState !== "live" || video.srcObject?.getVideoTracks()[0] !== track) {
    throw new Error("camera_not_ready");
  }

  const fromVideo = !blob;
  if (fromVideo) {
    if (!video.videoWidth || !video.videoHeight || video.readyState < 2 || track.muted) {
      throw new Error("camera_frame_unavailable");
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d", { alpha: false }).drawImage(video, 0, 0);
    blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", jpegQuality));
    if (!blob?.size || !Object.hasOwn(imageExtensions, blob.type)) throw new Error("camera_encoding_failed");
  }

  const timestamp = new Date().toISOString().replaceAll(":", "-");
  const file = new File([blob], `capture-${timestamp}.${imageExtensions[blob.type]}`, { type: blob.type });
  return { file, fromVideo };
}
