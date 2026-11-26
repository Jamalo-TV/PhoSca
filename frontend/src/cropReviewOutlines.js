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

export function getPaperGridReview(mask) {
  const { outline, cut } = mask?.paper_grid_review || {};
  if (!Array.isArray(outline) || outline.length !== 4 || !outline.every(isNormalizedPoint)) return null;
  if (cut == null) return { outline, cut: null };
  return Array.isArray(cut) && cut.length === 2 && cut.every(isNormalizedPoint) ? { outline, cut } : null;
}

function getTraceReviewGroups(photos, field) {
  const groups = new Map();
  for (const photo of photos) {
    const id = photo.photo_id ?? photo.id;
    const { segmentation_confidence: confidence = photo.confidence } = photo;
    const traces = photo.segmentation_mask?.[field];
    if (!id || photo.reviewed_non_print || confidence == null || !Array.isArray(traces)
        || traces.length !== 2 || !traces.every(line => Array.isArray(line) && line.length === 2
          && line.every(isNormalizedPoint))) continue;
    const key = JSON.stringify(traces.map(line => line.map(({ x, y }) => [x, y])));
    if (!groups.has(key)) groups.set(key, { key, traces, photoIds: new Set() });
    groups.get(key).photoIds.add(id);
  }
  return [...groups.values()].filter(({ photoIds }) => photoIds.size > 1);
}

export function getCaptureSplitReviewGroups(photos) {
  return getTraceReviewGroups(photos, "capture_split_review_traces");
}

export function getPaperJoinReviewGroups(photos) {
  return getTraceReviewGroups(photos, "paper_join_review_traces").filter(({ photoIds }) => photoIds.size === 2);
}

export function getPaperGridReviewGroups(photos) {
  const groups = new Map();
  for (const photo of photos) {
    const id = photo.photo_id ?? photo.id;
    const { segmentation_confidence: confidence = photo.confidence } = photo;
    if (!id || photo.reviewed_non_print || confidence == null) continue;
    const review = getPaperGridReview(photo.segmentation_mask);
    if (!review) continue;
    const key = JSON.stringify(review.outline.map(({ x, y }) => [x, y]));
    if (!groups.has(key)) groups.set(key, { key, outline: review.outline, parents: new Set(), cuts: new Map() });
    const group = groups.get(key);
    if (review.cut) group.cuts.set(id, review.cut);
    else group.parents.add(id);
  }
  return [...groups.values()].filter(({ parents, cuts }) => parents.size === 1 && cuts.size > 0);
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

export function getStoredMotifQuad(photo, mode) {
  return photo.segmentation_mask?.image_quad
    ?? (mode?.status === "ready" ? mode.warp_quad || photo.image_quad : null);
}

export function motifFitsPaper(motif, paper) {
  const tolerance = 1e-8;
  return motif.every(point => {
    const sides = paper.map((start, index) => {
      const end = paper[(index + 1) % paper.length];
      return (end.x - start.x) * (point.y - start.y) - (end.y - start.y) * (point.x - start.x);
    });
    return sides.every(side => side >= -tolerance) || sides.every(side => side <= tolerance);
  });
}
