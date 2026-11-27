import { contrastOutlineNeedsConfirmation } from "./cropReviewApproval.js";

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

function sourceReviewLine(line) {
  return Array.isArray(line) && line.length === 2 && line.every(isNormalizedPoint) ? line : null;
}

export function sourcePixelCenter({ x, y }, width, height) {
  return [x * (width - 1) + .5, y * (height - 1) + .5];
}

export const filmNeighborEdgeColor = "#f59e0b";

export function getFilmNeighborEdgeIndex(fullPrint) {
  const index = fullPrint?.film_neighbor_edge?.trace_index;
  const traces = fullPrint?.suggested_edge_traces;
  return fullPrint?.status === "review_needed" && fullPrint.suggested_edge_source === "competing_print_edges"
    && (index === 0 || index === 1) && Array.isArray(traces) && traces.length === 2 && traces.every(sourceReviewLine)
    ? index : null;
}

export function filmSourceWindow(line, width, height) {
  if (!sourceReviewLine(line) || !Number.isFinite(width) || !Number.isFinite(height) || width < 2 || height < 2) return null;
  const boxWidth = Math.min(width, Math.max(48, .13 * Math.min(width, height)));
  const boxHeight = Math.min(height, boxWidth * 2 / 3);
  const [a, b] = line.map(point => sourcePixelCenter(point, width, height));
  const x = (a[0] + b[0]) / 2, y = (a[1] + b[1]) / 2;
  const delta = [b[0] - a[0], b[1] - a[1]];
  const length = Math.hypot(...delta);
  if (length <= 1) return null;
  const reach = Math.min(.5, .46 * Math.min(boxWidth, boxHeight) / length);
  const ticks = [[-1, -.82], [.82, 1]].map(pair => pair.map(fraction => [
    x + delta[0] * reach * fraction, y + delta[1] * reach * fraction,
  ]));
  return { view: [Math.min(width - boxWidth, Math.max(0, x - boxWidth / 2)),
    Math.min(height - boxHeight, Math.max(0, y - boxHeight / 2)), boxWidth, boxHeight], ticks };
}

export function albumLeafSourceWindow(traces, width, height) {
  const [first, second] = traces.map(line => filmSourceWindow(line, width, height));
  if (!first || !second) return null;
  const ticks = [...first.ticks, ...second.ticks];
  const points = ticks.flat();
  const xs = points.map(([x]) => x), ys = points.map(([, y]) => y);
  const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
  const boxWidth = Math.min(width, Math.max(first.view[2], (right - left) * 1.4, (bottom - top) * 2.1));
  const boxHeight = Math.min(height, boxWidth * 2 / 3);
  return { view: [Math.min(width - boxWidth, Math.max(0, (left + right - boxWidth) / 2)),
    Math.min(height - boxHeight, Math.max(0, (top + bottom - boxHeight) / 2)), boxWidth, boxHeight], ticks };
}

export function paperGridSourceWindows(line, width, height) {
  if (!sourceReviewLine(line) || ![width, height].every(size => Number.isInteger(size) && size > 1)) return [];
  const [first, last] = line.map(point => sourcePixelCenter(point, width, height));
  const dx = last[0] - first[0], dy = last[1] - first[1];
  const length = Math.hypot(dx, dy);
  if (length <= 1) return [];
  const span = Math.min(width, height, Math.min(144, Math.max(64, Math.round(.27 * length))));
  const normal = [-dy / length, dx / length];
  return [first, [(first[0] + last[0]) / 2, (first[1] + last[1]) / 2], last].map(([x, y]) => ({
    view: [Math.min(width - span, Math.max(0, Math.round(x - span / 2))),
      Math.min(height - span, Math.max(0, Math.round(y - span / 2))), span, span],
    ticks: [[-.39, -.30], [.30, .39]].map(pair => pair.map(fraction => [
      x + normal[0] * span * fraction, y + normal[1] * span * fraction,
    ])),
  }));
}

export function captureSplitSourceWindow(traces, width, height) {
  if (!Array.isArray(traces) || traces.length !== 2 || !traces.every(sourceReviewLine)
      || ![width, height].every(size => Number.isInteger(size) && size > 1)) return null;
  const [first, second] = traces;
  const sharedIndex = first.findIndex(point => second.some(other => point.x === other.x && point.y === other.y));
  if (sharedIndex < 0) return null;
  const joint = first[sharedIndex];
  const ends = [first[1 - sharedIndex], second.find(point => point.x !== joint.x || point.y !== joint.y)];
  const onFrame = value => value === 0 || value === 1;
  if (onFrame(joint.x) || onFrame(joint.y) || !ends[1]
      || ends.some(({ x, y }) => onFrame(x) === onFrame(y)) || onFrame(ends[0].x) === onFrame(ends[1].x)) return null;
  const lines = traces.map(line => line.map(point => sourcePixelCenter(point, width, height)));
  const points = lines.flat();
  const xs = points.map(([x]) => x), ys = points.map(([, y]) => y);
  const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
  const span = Math.min(right - left, bottom - top);
  if (span <= 1) return null;
  const padding = Math.max(8, .08 * span);
  const x1 = Math.max(0, Math.floor(left - padding)), y1 = Math.max(0, Math.floor(top - padding));
  const x2 = Math.min(width, Math.ceil(right + padding)), y2 = Math.min(height, Math.ceil(bottom + padding));
  return { view: [x1, y1, x2 - x1, y2 - y1], lines };
}

function cropModes(photo) {
  return photo?.crop_modes || photo?.enhancement_applied?.correction?.crop_modes;
}

function bothModesHeld(modes) {
  return modes?.full_print?.status === "review_needed" && modes?.image_only?.status === "review_needed";
}

export const competingPhotoRegionStyles = {
  shared: { label: "In beiden Vorschlägen", color: "#7c3aed", fill: "rgba(139, 92, 246, .32)" },
  wider: { label: "Nur im grösseren Vorschlag", color: "#ea580c", fill: "rgba(249, 115, 22, .34)" },
  narrower: { label: "Nur im kleineren Vorschlag", color: "#0891b2", fill: "rgba(34, 211, 238, .34)" },
};
export const sharedMotifGuideStyle = { color: "#f0fdf4", contrast: "#14532d", dash: [5, 4] };

export function getCompetingPhotoOutline(photo) {
  if (!photo || photo.reviewed_non_print || photo.review_reasons?.includes("GEOMETRY_REVIEW_WRONG_OBJECT")) return null;
  const { segmentation_confidence: confidence = photo.confidence, competing_photo_outline: outline } = photo;
  const id = photo.photo_id ?? photo.id;
  if (confidence == null || !bothModesHeld(cropModes(photo)) || !id || !outline) return null;
  const { wider, narrower, wider_photo_id: widerId, narrower_photo_id: narrowerId } = outline;
  if (typeof widerId !== "string" || typeof narrowerId !== "string" || widerId === narrowerId
      || ![wider, narrower].every(points => Array.isArray(points) && points.length >= 3 && points.every(isNormalizedPoint))) return null;
  const member = id === widerId ? wider : id === narrowerId ? narrower : null;
  const raw = photo.segmentation_mask?.polygon;
  if (!member || raw?.length !== member.length
      || !member.every(({ x, y }, index) => x === raw[index]?.x && y === raw[index]?.y)) return null;
  const guide = outline.motif_guide;
  return Array.isArray(guide) && guide.length === 4 && guide.every(isNormalizedPoint)
    ? { ...outline, motifGuide: guide } : outline;
}

export function getCompetingPhotoReviewOutlines(photos) {
  const current = new Map(photos.map(photo => [photo.photo_id ?? photo.id, getCompetingPhotoOutline(photo)]));
  return [...current].flatMap(([id, outline]) => {
    if (id !== outline?.wider_photo_id) return [];
    const peer = current.get(outline.narrower_photo_id);
    return peer?.wider_photo_id === id && peer.narrower_photo_id === outline.narrower_photo_id
      && JSON.stringify([peer.wider, peer.narrower, peer.motifGuide])
        === JSON.stringify([outline.wider, outline.narrower, outline.motifGuide]) ? [outline] : [];
  });
}

export function competingPhotoRegionPaths({ wider, narrower }, width, height) {
  const closedPath = points => `M ${points.map(({ x, y }) => `${x * width} ${y * height}`).join(" L ")} Z`;
  const outer = closedPath(wider);
  const inner = closedPath(narrower);
  const exclusive = `${outer} ${inner}`;
  return [
    { kind: "shared", path: inner, clip: outer, rule: "nonzero" },
    { kind: "wider", path: exclusive, clip: outer, rule: "evenodd" },
    { kind: "narrower", path: exclusive, clip: inner, rule: "evenodd" },
  ];
}

export function sameReviewQuad(saved, edited) {
  return saved?.length === 4 && edited?.length === 4
    && saved.every(({ x, y }, index) => x === edited[index].x && y === edited[index].y);
}

export function savedFullPrintReviewUrl(photo, savedQuad, editedQuad, sizeFields, orientation) {
  if (!photo?.saved_full_print_review_url || photo.reviewed_non_print || !sameReviewQuad(savedQuad, editedQuad)
      || sizeFields?.width?.trim() || sizeFields?.height?.trim()
      || orientation !== (photo.segmentation_mask?.orientation_degrees ?? null)) return null;
  return photo.saved_full_print_review_url;
}

export const contrastCrossingTraceHint = "An den orange markierten Stellen ist Bildstruktur auf beiden Seiten der vorgeschlagenen Linie sichtbar. Prüfe dort den Papierrand am Original.";
export const contrastCrossingTraceStyle = { color: "#f59e0b", contrast: "#78350f", dash: [8, 5] };
export const captureBackgroundTraceStyle = { color: "#67e8f9", contrast: "#164e63", dash: [8, 4] };

export function getCaptureBackgroundReviewTraces(photo) {
  if (!photo || photo.reviewed_non_print || photo.review_reasons?.includes("GEOMETRY_REVIEW_WRONG_OBJECT")) return [];
  const { segmentation_confidence: confidence = photo.confidence, segmentation_mask: mask } = photo;
  const reasons = photo.enhancement_applied?.segmentation_review_reasons ?? photo.review_reasons;
  if (confidence == null || !bothModesHeld(cropModes(photo)) || mask?.warp_quad
      || !reasons?.includes("CAPTURE_BACKGROUND_COUNT_REVIEW")) return [];
  const traces = mask?.album_boundary_traces;
  return Array.isArray(traces) ? traces.filter(line => sourceReviewLine(line)
    && (line[0].x !== line[1].x || line[0].y !== line[1].y)) : [];
}

export function getContrastCrossingTraces(photo, savedQuad, editedQuad) {
  if (!photo || photo.reviewed_non_print || photo.review_reasons?.includes("GEOMETRY_REVIEW_WRONG_OBJECT")) return [];
  const modes = cropModes(photo);
  if (!bothModesHeld(modes) || !contrastOutlineNeedsConfirmation(photo, modes.full_print)) return [];
  if (!sameReviewQuad(savedQuad, editedQuad)) return [];
  const traces = photo.segmentation_mask?.contrast_crossing_traces;
  return Array.isArray(traces) && traces.length === 2 && traces.every(line => sourceReviewLine(line)
    && (line[0].x !== line[1].x || line[0].y !== line[1].y)) ? traces : [];
}

export function getCompactSceneCrossingTraces(photo) {
  if (!photo || photo.reviewed_non_print || photo.review_reasons?.includes("GEOMETRY_REVIEW_WRONG_OBJECT")) return [];
  const { segmentation_confidence: confidence = photo.confidence, segmentation_mask: mask } = photo;
  if (confidence == null || !bothModesHeld(cropModes(photo)) || mask?.warp_quad
      || mask?.source !== "classical_border_quad" || mask.detector_source !== "edge_contour") return [];
  const trace = mask.compact_scene_crossing_trace;
  return sourceReviewLine(trace) && (trace[0].x !== trace[1].x || trace[0].y !== trace[1].y) ? [trace] : [];
}

export const measuredRibbedPrintOutlineSource = "measured_ribbed_print_outline";
export const ribbedPrintOutlineHint = "Der blau gestrichelte Umriss zeigt eine mögliche Papierkante im Original.";
export const ribbedPrintOutlineStyle = { color: "#60a5fa", contrast: "#1e3a8a", dash: [8, 5] };

export function getRibbedPrintReviewOutline(fullPrint, previewUrl, savedQuad, editedQuad) {
  const preview = fullPrint?.outline_preview;
  if (fullPrint?.status !== "review_needed" || preview?.source !== measuredRibbedPrintOutlineSource
      || !previewUrl || !sameReviewQuad(savedQuad, editedQuad)) return null;
  const outline = preview.source_outline;
  return Array.isArray(outline) && outline.length === 4 && outline.every(isNormalizedPoint) ? outline : null;
}

export function getFilmGridSeam(mask) {
  return sourceReviewLine(mask?.film_grid_review_seam);
}

export function getFilmGridReviewSeams(photos) {
  const groups = new Map();
  for (const photo of photos) {
    const id = photo.photo_id ?? photo.id;
    const { segmentation_confidence: confidence = photo.confidence } = photo;
    const line = getFilmGridSeam(photo.segmentation_mask);
    if (!id || photo.reviewed_non_print || confidence == null || !line) continue;
    const key = JSON.stringify(line.map(({ x, y }) => [x, y]));
    const rim = sourceReviewLine(photo.segmentation_mask?.film_grid_review_rim);
    if (!groups.has(key)) groups.set(key, { key, line, rim, photoIds: new Set() });
    const group = groups.get(key);
    if (JSON.stringify(group.rim) !== JSON.stringify(rim)) group.rim = null;
    group.photoIds.add(id);
  }
  return [...groups.values()].filter(({ photoIds }) => photoIds.size > 1);
}

export function getActionableFilmGridReviewGroups(groups, photos) {
  const held = new Set(photos.filter(photo => bothModesHeld(cropModes(photo))).map(photo => photo.photo_id ?? photo.id));
  return groups.filter(({ photoIds }) => photoIds.size === 3 && [...photoIds].every(id => held.has(id)));
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

export function getPrintedIllustrationReviewGroups(photos) {
  const active = photos.filter(photo => !photo.reviewed_non_print);
  if (active.length !== 4) return [];
  const { printed_illustration_review_frame: frames, printed_illustration_review_id: evidenceId } = active[0].segmentation_mask || {};
  if (!/^[a-f0-9]{64}$/.test(evidenceId ?? "") || !Array.isArray(frames) || frames.length !== 1
      || !Array.isArray(frames[0]) || frames[0].length < 20 || frames[0].length > 160 || !frames[0].every(isNormalizedPoint)) return [];
  const serialized = JSON.stringify(frames);
  const photoIds = new Set(active.map(photo => photo.photo_id ?? photo.id));
  if (photoIds.size !== 4 || photoIds.has(undefined) || photoIds.has(null) || !active.every(photo => {
    const { segmentation_confidence: confidence = photo.confidence } = photo;
    return confidence != null && bothModesHeld(cropModes(photo))
      && photo.segmentation_mask?.printed_illustration_review_id === evidenceId
      && JSON.stringify(photo.segmentation_mask?.printed_illustration_review_frame) === serialized;
  })) return [];
  return [{ kind: "printed_illustration", evidenceId, key: JSON.stringify([evidenceId, ...[...photoIds].sort()]),
    frame: frames[0], photoIds }];
}

export function getCrossingFallbackReviewGroups(photos) {
  return photos.flatMap(photo => {
    const id = photo.photo_id ?? photo.id;
    const { segmentation_confidence: confidence = photo.confidence } = photo;
    const traces = photo.segmentation_mask?.crossing_fallback_review_traces;
    if (!id || photo.reviewed_non_print || confidence == null
        || !Array.isArray(traces) || traces.length !== 4
        || !traces.every(line => Array.isArray(line) && line.length === 2 && line.every(isNormalizedPoint))
        || !bothModesHeld(cropModes(photo))) return [];
    return [{ kind: "crossing", key: JSON.stringify(["crossing", id, traces]), traces, photoIds: new Set([id]) }];
  });
}

export function getAlbumLeafReviewGroups(photos) {
  const active = photos.filter(photo => {
    const { segmentation_confidence: confidence = photo.confidence } = photo;
    return !photo.reviewed_non_print && (photo.photo_id ?? photo.id) && confidence != null;
  });
  return active.flatMap(photo => {
    const id = photo.photo_id ?? photo.id;
    const { album_leaf_review_traces: traces, album_leaf_review_child: child } = photo.segmentation_mask || {};
    if (!Array.isArray(traces) || traces.length !== 2 || !traces.every(sourceReviewLine)
        || !Array.isArray(child) || child.length < 4 || !child.every(isNormalizedPoint)
        || !bothModesHeld(cropModes(photo))) return [];
    const matches = new Set(active.filter(other => {
      const polygon = other.segmentation_mask?.polygon;
      return (other.photo_id ?? other.id) !== id && polygon?.length === child.length
        && child.every(({ x, y }, index) => polygon[index]?.x === x && polygon[index]?.y === y);
    }).map(other => other.photo_id ?? other.id));
    if (matches.size !== 1) return [];
    const [childId] = matches;
    return [{ kind: "album_leaf", key: JSON.stringify(["album_leaf", id, childId, traces, child]),
      traces, child, childId, photoIds: new Set([id]) }];
  });
}

export function getPaperGridReviewGroups(photos) {
  const groups = new Map();
  for (const photo of photos) {
    const id = photo.photo_id ?? photo.id;
    const { segmentation_confidence: confidence = photo.confidence } = photo;
    if (!id || photo.reviewed_non_print) continue;
    const manual = confidence == null;
    const review = getPaperGridReview(manual ? { paper_grid_review: photo.manual_paper_grid_count_review } : photo.segmentation_mask);
    if (!review || (manual && review.cut)) continue;
    const key = JSON.stringify(review.outline.map(({ x, y }) => [x, y]));
    if (!groups.has(key)) groups.set(key, { key, outline: review.outline, parents: new Set(), cuts: new Map() });
    const group = groups.get(key);
    if (review.cut) group.cuts.set(id, review.cut);
    else group.parents.add(id);
  }
  return [...groups.values()].filter(({ parents, cuts }) => parents.size === 1 && cuts.size > 0);
}

export function getPaperGridOutlineParents(photos) {
  return photos.filter((photo) => {
    const id = photo.photo_id ?? photo.id;
    const { segmentation_confidence: confidence = photo.confidence } = photo;
    return id && !photo.reviewed_non_print && confidence != null
      && getPaperGridReview(photo.segmentation_mask)?.cut === null;
  });
}

export function getActionablePaperGridReviewGroups(groups, photos) {
  const photosById = new Map(photos.map(photo => [photo.photo_id ?? photo.id, photo]));
  return groups.filter(({ parents, cuts }) => {
    const [parentId] = parents;
    const parent = photosById.get(parentId);
    const proposed = cropModes(parent)?.full_print?.suggested_warp_quad ?? parent?.manual_paper_grid_count_review?.outline;
    return cuts.size >= 1 && cuts.size <= 2 && Array.isArray(proposed) && proposed.length === 4 && proposed.every(isNormalizedPoint)
      && [parentId, ...cuts.keys()].every(id => bothModesHeld(cropModes(photosById.get(id))));
  });
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

export function getMergedPrintPairs(photos) {
  return photos.flatMap(photo => {
    const id = photo.photo_id ?? photo.id;
    const { segmentation_confidence: confidence = photo.confidence } = photo;
    const { outlines, separator } = photo.segmentation_mask?.merged_print_pair || {};
    if (!id || photo.reviewed_non_print || confidence == null || !bothModesHeld(cropModes(photo))
        || !Array.isArray(outlines) || outlines.length !== 2 || !sourceReviewLine(separator)
        || !outlines.every(outline => Array.isArray(outline) && outline.length === 4 && outline.every(isNormalizedPoint))) {
      return [];
    }
    return [{ id, outlines, separator }];
  });
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
