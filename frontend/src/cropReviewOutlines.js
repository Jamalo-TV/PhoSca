function isNormalizedPoint(point) {
  return [point?.x, point?.y].every((value) => Number.isFinite(value) && value >= 0 && value <= 1);
}

export function getFragmentOutline(mask) {
  const { quad, uncertain_edge: edge } = mask?.fragment_outline_suggestion || {};
  if (!Array.isArray(quad) || quad.length !== 4 || !Number.isInteger(edge) || edge < 0 || edge > 3
      || !quad.every(isNormalizedPoint)) {
    return null;
  }
  const next = (edge + 1) % 4;
  return { quad, weakEdge: [quad[edge], quad[next]], supportedSides: [...quad.slice(next), ...quad.slice(0, next)] };
}

export function getFilmGridSeam(mask) {
  const line = mask?.film_grid_review_seam;
  return Array.isArray(line) && line.length === 2 && line.every(isNormalizedPoint) ? line : null;
}

export function getFilmGridReviewSeams(photos) {
  const groups = new Map();
  for (const photo of photos) {
    const id = photo.photo_id ?? photo.id;
    const { segmentation_confidence: confidence = photo.confidence } = photo;
    const line = getFilmGridSeam(photo.segmentation_mask);
    if (!id || photo.reviewed_non_print || confidence == null || !line) continue;
    const key = JSON.stringify(line.map(({ x, y }) => [x, y]));
    if (!groups.has(key)) groups.set(key, { key, line, photoIds: new Set() });
    groups.get(key).photoIds.add(id);
  }
  return [...groups.values()].filter(({ photoIds }) => photoIds.size > 1);
}

export function getFragmentReviewOutlines(photos) {
  const groups = new Map();
  for (const photo of photos) {
    const id = photo.photo_id ?? photo.id;
    const { segmentation_confidence: confidence = photo.confidence } = photo;
    if (!id || photo.reviewed_non_print || confidence == null) continue;
    const outline = getFragmentOutline(photo.segmentation_mask);
    if (!outline) continue;
    const { supportedSides, weakEdge } = outline;
    const key = JSON.stringify(supportedSides.map(({ x, y }) => [x, y]));
    if (!groups.has(key)) groups.set(key, { supportedSides, weakEdge, key, photoIds: new Set() });
    groups.get(key).photoIds.add(id);
  }
  return [...groups.values()].filter(({ photoIds }) => photoIds.size > 1);
}
