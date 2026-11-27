import assert from "node:assert/strict";
import test from "node:test";
import {
  albumLeafSourceWindow, captureSplitSourceWindow, getAlbumLeafReviewGroups, getCaptureBackgroundReviewTraces,
  getCompactSceneCrossingTraces, getContrastCrossingTraces,
  competingPhotoRegionPaths, getCompetingPhotoOutline, getCompetingPhotoReviewOutlines,
  filmSourceWindow, getActionableFilmGridReviewGroups, getActionablePaperGridReviewGroups, getCaptureSplitReviewGroups, getCrossingFallbackReviewGroups,
  getFilmGridReviewSeams, getFilmNeighborEdgeIndex, getFragmentReviewOutlines, getMergedPrintPairs,
  getPaperGridOutlineParents, getPaperGridReviewGroups,
  getPaperJoinReviewGroups, getPossiblePrintRegions, getPrintedIllustrationReviewGroups, getStoredMotifQuad, motifFitsPaper, paperGridSourceWindows,
  savedFullPrintReviewUrl,
  getRibbedPrintReviewOutline, measuredRibbedPrintOutlineSource,
  getMeasuredMotifComparison, isMeasuredMotifComparison, measuredMotifComparisonSource,
} from "../src/cropReviewOutlines.js";

const quad = [{ x: .1, y: .2 }, { x: .7, y: .2 }, { x: .7, y: .8 }, { x: .1, y: .8 }];
const mask = { fragment_outline_suggestion: { quad, uncertain_edge: 1 } };

test("a measured motif comparison follows the held source without changing its geometry", () => {
  const status = "review_needed";
  const seam = [{ x: .3, y: .4 }, { x: .6, y: .4 }];
  const paper = [{ x: .72, y: .37 }, { x: .75, y: .69 }];
  const modes = {
    full_print: { status, motif_paper_edge_traces: [paper, [paper[0], paper[0]]] },
    image_only: { status, outline_preview: { source: measuredMotifComparisonSource },
      suggested_seam_source: measuredMotifComparisonSource, suggested_warp_quad: quad,
      suggested_seam_traces: [seam, [seam[0], { x: 2, y: .6 }]] },
  };
  const previewUrl = "/motif-preview";
  const expected = { previewUrl, outline: quad, seamTraces: [seam], paperTraces: [paper] };
  assert.equal(isMeasuredMotifComparison(modes.image_only), true);
  assert.equal(isMeasuredMotifComparison({ outline_preview: { source: "paper_stock_motif" } }), false);
  assert.equal(isMeasuredMotifComparison({ suggested_seam_source: measuredMotifComparisonSource }), true);
  assert.deepEqual(getMeasuredMotifComparison(modes, previewUrl, quad, structuredClone(quad)), expected);
  assert.deepEqual(getMeasuredMotifComparison({ ...modes, full_print: { status } }, previewUrl, quad, quad),
    { ...expected, paperTraces: [] });
  assert.deepEqual(getMeasuredMotifComparison({ ...modes, image_only: {
    ...modes.image_only, suggested_seam_source: "partial_motif_seams",
  } }, previewUrl, quad, quad), { ...expected, seamTraces: [] });
  assert.equal(getMeasuredMotifComparison(modes, null, quad, quad), null);
  assert.equal(getMeasuredMotifComparison(modes, previewUrl, quad, [{ x: .11, y: .2 }, ...quad.slice(1)]), null);
  for (const [mode, change] of [
    ["full_print", { status: "ready" }],
    ["image_only", { status: "ready" }],
    ["image_only", { outline_preview: { source: "paper_stock_motif" } }],
    ["image_only", { suggested_warp_quad: [quad[1], ...quad.slice(1)] }],
  ]) {
    assert.equal(getMeasuredMotifComparison({ ...modes, [mode]: { ...modes[mode], ...change } }, previewUrl, quad, quad), null);
  }
});

test("capture background guidance follows a held active proposal and its measured album edge", () => {
  const held = { status: "review_needed" };
  const trace = [{ x: .05, y: .16 }, { x: .92, y: .14 }];
  const reason = "CAPTURE_BACKGROUND_COUNT_REVIEW";
  const photo = {
    confidence: .91, review_reasons: [reason], crop_modes: { full_print: held, image_only: held },
    segmentation_mask: { album_boundary_traces: [trace, [trace[0], trace[0]], [trace[0], { x: 2, y: .14 }]] },
  };
  assert.deepEqual(getCaptureBackgroundReviewTraces(photo), [trace]);
  const { confidence, review_reasons, crop_modes, ...fields } = photo;
  assert.deepEqual(getCaptureBackgroundReviewTraces({ ...fields, segmentation_confidence: confidence,
    enhancement_applied: { segmentation_review_reasons: review_reasons, correction: { crop_modes } } }), [trace]);
  for (const changed of [
    { confidence: null }, { reviewed_non_print: true }, { review_reasons: [] },
    { review_reasons: [reason, "GEOMETRY_REVIEW_WRONG_OBJECT"] },
    { crop_modes: { ...crop_modes, full_print: { status: "ready" } } },
    { segmentation_mask: { ...photo.segmentation_mask, warp_quad: quad } },
  ]) {
    assert.deepEqual(getCaptureBackgroundReviewTraces({ ...photo, ...changed }), []);
  }
});

test("a printed illustration group requires all four current automatic held proposals", () => {
  const frame = Array.from({ length: 24 }, (_, index) => ({
    x: .5 + .4 * Math.cos(index * Math.PI / 12), y: .5 + .4 * Math.sin(index * Math.PI / 12),
  }));
  const held = { status: "review_needed" };
  const crop_modes = { full_print: held, image_only: held };
  const evidenceId = "a".repeat(64);
  const segmentation_mask = { printed_illustration_review_frame: [frame], printed_illustration_review_id: evidenceId };
  const photos = Array.from({ length: 4 }, (_, index) => ({ photo_id: String(index), confidence: .64,
    segmentation_mask: structuredClone(segmentation_mask), crop_modes }));
  const [group] = getPrintedIllustrationReviewGroups(photos);
  assert.equal(group.kind, "printed_illustration");
  assert.equal(group.evidenceId, evidenceId);
  assert.deepEqual(group.photoIds, new Set(["0", "1", "2", "3"]));
  assert.deepEqual(group.frame, frame);
  const [first, ...others] = photos;
  assert.deepEqual(getPrintedIllustrationReviewGroups(others), []);
  assert.deepEqual(getPrintedIllustrationReviewGroups([...photos, { ...first, photo_id: "new" }]), []);
  for (const override of [
    { reviewed_non_print: true },
    { confidence: null },
    { crop_modes: { ...crop_modes, image_only: { status: "ready" } } },
    { segmentation_mask: { ...segmentation_mask, printed_illustration_review_id: "b".repeat(64) } },
    { segmentation_mask: { ...segmentation_mask, printed_illustration_review_frame: [frame.slice(1)] } },
    { photo_id: "1" },
  ]) assert.deepEqual(getPrintedIllustrationReviewGroups([{ ...first, ...override }, ...others]), []);
});

test("a film-neighbor distinction only applies to its two active held lines", () => {
  const mode = { status: "review_needed", suggested_edge_source: "competing_print_edges",
    suggested_edge_traces: [quad.slice(0, 2), quad.slice(2)],
    film_neighbor_edge: { trace_index: 1 } };
  assert.equal(getFilmNeighborEdgeIndex(mode), 1);
  assert.equal(getFilmNeighborEdgeIndex({ ...mode, film_neighbor_edge: { trace_index: 0 } }), 0);
  assert.equal(getFilmNeighborEdgeIndex({ ...mode, status: "ready" }), null);
  assert.equal(getFilmNeighborEdgeIndex({ ...mode, film_neighbor_edge: { trace_index: 2 } }), null);
  assert.equal(getFilmNeighborEdgeIndex({ ...mode, suggested_edge_traces: [null, quad.slice(2)] }), null);
});

test("held motif geometry requires selection while a saved or ready motif can be edited", () => {
  const held = { status: "review_needed", warp_quad: quad, suggested_warp_quad: quad };
  assert.equal(getStoredMotifQuad({ image_quad: quad }, held), null);
  assert.equal(getStoredMotifQuad({ segmentation_mask: { image_quad: quad } }, held), quad);
  assert.equal(getStoredMotifQuad({}, { status: "ready", warp_quad: quad }), quad);
});

test("a merged pair is shown on the current held row with both guides", () => {
  const outlines = [.0, .32].map(shift => quad.map(point => ({ x: point.x, y: .2 + shift + (point.y - .2) / 3 })));
  const separator = [{ x: .06, y: .48 }, { x: .78, y: .48 }];
  const pair = { outlines, separator };
  const held = { status: "review_needed" };
  const row = { photo_id: "owner", confidence: .49, crop_modes: { full_print: held, image_only: held },
    segmentation_mask: { merged_print_pair: pair } };
  assert.deepEqual(getMergedPrintPairs([row]), [{ id: "owner", outlines, separator }]);
  assert.deepEqual(getMergedPrintPairs([
    { ...row, reviewed_non_print: true },
    { ...row, confidence: null },
    { ...row, crop_modes: { full_print: { status: "ready" }, image_only: held } },
    { ...row, segmentation_mask: { merged_print_pair: null } },
    { ...row, segmentation_mask: { merged_print_pair: { ...pair, outlines: outlines.slice(1) } } },
  ]), []);
});

test("uncertain regions only guide the current held automatic owner without becoming a merged print pair", () => {
  const upper = quad.map(({ x, y }) => ({ x, y: y / 2 }));
  const lower = quad.map(({ x, y }) => ({ x, y: .5 + y / 2 })).reverse();
  const outlines = [upper, lower];
  const held = { status: "review_needed" };
  const modes = { full_print: held, image_only: held };
  const mask = { possible_print_regions: { outlines } };
  const photo = { photo_id: "owner", confidence: .61, crop_modes: modes, segmentation_mask: mask };
  assert.deepEqual(getPossiblePrintRegions([photo]), [{ id: "owner", outlines }]);
  assert.deepEqual(getPossiblePrintRegions([{ id: "owner", segmentation_confidence: .61, segmentation_mask: mask,
    enhancement_applied: { correction: { crop_modes: modes } } }]), [{ id: "owner", outlines }]);
  assert.deepEqual(getMergedPrintPairs([photo]), []);
  assert.deepEqual(getPossiblePrintRegions([
    null,
    { ...photo, reviewed_non_print: true },
    { ...photo, review_reasons: ["GEOMETRY_REVIEW_WRONG_OBJECT"] },
    { ...photo, confidence: null },
    { ...photo, crop_modes: { full_print: held, image_only: { status: "ready" } } },
    { ...photo, segmentation_mask: { ...mask, warp_quad: quad } },
    { ...photo, segmentation_mask: { ...mask, merged_print_pair: { outlines } } },
    { ...photo, segmentation_mask: { possible_print_regions: { outlines: [upper, [...upper].reverse()] } } },
    { ...photo, segmentation_mask: { possible_print_regions: { outlines: [upper, lower.slice(1)] } } },
    { ...photo, segmentation_mask: { possible_print_regions: { outlines: [upper, [lower[0], lower[2], lower[1], lower[3]]] } } },
  ]), []);
});

test("motif corners may touch the paper while exterior corners fail on either winding", () => {
  const paper = [quad[0], { x: .65, y: .3 }, quad[2], quad[3]];
  const inside = [{ x: .2, y: .3 }, { x: .6, y: .35 }, { x: .65, y: .75 }, { x: .2, y: .75 }];
  const outside = [inside[0], { x: .66, y: .3 }, inside[2], inside[3]];
  for (const outline of [paper, [...paper].reverse()]) {
    assert.equal(motifFitsPaper(paper, outline), true);
    assert.equal(motifFitsPaper(inside, outline), true);
    assert.equal(motifFitsPaper(outside, outline), false);
  }
});

test("a crossing count hint selects only its current, held automatic proposal", () => {
  const traces = quad.map((point, index) => [point, quad[(index + 1) % quad.length]]);
  const mask = { crossing_fallback_review_traces: traces };
  const modes = { full_print: { status: "review_needed" }, image_only: { status: "review_needed" } };
  const held = { photo_id: "crossed", confidence: .77, segmentation_mask: mask, crop_modes: modes };
  const [group, ...others] = getCrossingFallbackReviewGroups([
    held, { ...held, photo_id: "removed", reviewed_non_print: true },
    { ...held, photo_id: "manual", confidence: null },
    { ...held, photo_id: "ready", crop_modes: { ...modes, full_print: { status: "ready" } } },
    { ...held, photo_id: "incomplete", segmentation_mask: { crossing_fallback_review_traces: traces.slice(0, 3) } },
  ]);
  assert.deepEqual(others, []);
  assert.equal(group.kind, "crossing");
  assert.deepEqual(group.photoIds, new Set(["crossed"]));
  assert.deepEqual(group.traces, traces);
});

test("two valid contrast crossing spans are shown only on unchanged held automatic interior outlines", () => {
  const traces = [[quad[0], quad[1]], [quad[2], quad[3]]];
  const segmentationMask = { detector_source: "contrast_region", contrast_crossing_traces: traces };
  const fullPrint = { status: "review_needed", reason_codes: ["PRINT_BOUNDARY_UNSUPPORTED"] };
  const modes = { full_print: fullPrint, image_only: { status: "review_needed" } };
  const desktop = { segmentation_confidence: .71, segmentation_mask: segmentationMask,
    enhancement_applied: { correction: { crop_modes: modes } } };
  const mobile = { confidence: .71, segmentation_mask: segmentationMask, crop_modes: modes };
  const unchanged = structuredClone(quad);
  const changed = structuredClone(quad);
  changed[1].x += .0001;
  assert.deepEqual(getContrastCrossingTraces(desktop, quad, unchanged), traces);
  assert.deepEqual(getContrastCrossingTraces(mobile, quad, unchanged), traces);
  assert.deepEqual(getContrastCrossingTraces(desktop, quad, changed), []);
  assert.deepEqual(getContrastCrossingTraces(mobile, quad, null), []);

  for (const photo of [
    { ...desktop, reviewed_non_print: true },
    { ...mobile, review_reasons: ["GEOMETRY_REVIEW_WRONG_OBJECT"] },
    { ...mobile, confidence: null },
    { ...mobile, segmentation_mask: { ...segmentationMask, detector_source: "yolo_onnx" } },
    { ...mobile, crop_modes: { ...modes, full_print: { ...fullPrint, status: "ready" } } },
    { ...mobile, crop_modes: { ...modes, image_only: { status: "ready" } } },
    { ...mobile, crop_modes: { full_print: fullPrint } },
    { ...mobile, crop_modes: { ...modes, full_print: { ...fullPrint, reason_codes: [] } } },
    { ...mobile, crop_modes: { ...modes, full_print: { ...fullPrint, reason_codes: [...fullPrint.reason_codes, "PRINT_AT_CAPTURE_EDGE"] } } },
    ...[[traces[0]], [...traces, traces[0]], [[quad[0], quad[0]], traces[1]], [quad, traces[1]],
      [[{ x: -.01, y: .2 }, quad[1]], traces[1]]].map(contrast_crossing_traces => ({
      ...mobile, segmentation_mask: { ...segmentationMask, contrast_crossing_traces },
    })),
  ]) assert.deepEqual(getContrastCrossingTraces(photo, quad, unchanged), []);
});

test("a compact scene crossing appears only on a current held classical outline", () => {
  const trace = quad.slice(0, 2);
  const mask = { source: "classical_border_quad", detector_source: "edge_contour", compact_scene_crossing_trace: trace };
  const held = { status: "review_needed" };
  const crop_modes = { full_print: held, image_only: held };
  const mobile = { confidence: .74, segmentation_mask: mask, crop_modes };
  const desktop = { segmentation_confidence: .74, segmentation_mask: mask,
    enhancement_applied: { correction: { crop_modes } } };
  assert.deepEqual(getCompactSceneCrossingTraces(mobile), [trace]);
  assert.deepEqual(getCompactSceneCrossingTraces(desktop), [trace]);
  for (const change of [
    { reviewed_non_print: true }, { review_reasons: ["GEOMETRY_REVIEW_WRONG_OBJECT"] }, { confidence: null },
    { crop_modes: { ...crop_modes, image_only: { status: "ready" } } },
    ...[{ warp_quad: quad }, { source: "yolo_seg_mask" }, { compact_scene_crossing_trace: [quad[0], quad[0]] }]
      .map(fields => ({ segmentation_mask: { ...mask, ...fields } })),
  ]) assert.deepEqual(getCompactSceneCrossingTraces({ ...mobile, ...change }), []);
});

test("a ribbed-paper preview exposes its own contour only on the unchanged held draft", () => {
  const sourceOutline = quad.map(({ x, y }) => ({ x: x + .01, y }));
  const mode = { status: "review_needed", outline_preview: {
    source: measuredRibbedPrintOutlineSource, source_outline: sourceOutline,
  } };
  const url = "/api/v1/photos/photo/image?variant=full_print_preview";
  assert.deepEqual(getRibbedPrintReviewOutline(mode, url, quad, structuredClone(quad)), sourceOutline);
  assert.equal(mode.suggested_warp_quad, undefined);

  const edited = quad.map((point, index) => index === 0 ? { ...point, x: point.x + .0001 } : point);
  assert.equal(getRibbedPrintReviewOutline(mode, url, quad, edited), null);
  assert.equal(getRibbedPrintReviewOutline(mode, null, quad, quad), null);
  assert.equal(getRibbedPrintReviewOutline({ ...mode, status: "ready" }, url, quad, quad), null);
  assert.equal(getRibbedPrintReviewOutline({ ...mode, outline_preview: {
    ...mode.outline_preview, source_outline: sourceOutline.slice(0, 3),
  } }, url, quad, quad), null);
  assert.equal(getRibbedPrintReviewOutline({ ...mode, outline_preview: {
    ...mode.outline_preview, source: "another_source",
  } }, url, quad, quad), null);
});

test("a competing outline requires both current held proposals in count review", () => {
  const narrower = [{ x: .2, y: .3 }, { x: .8, y: .31 }, { x: .51, y: .53 }, { x: .3, y: .7 }, { x: .2, y: .66 }];
  const outline = { wider_photo_id: "wide", narrower_photo_id: "narrow", wider: quad, narrower };
  const held = { status: "review_needed" };
  const modes = { full_print: held, image_only: held };
  const desktop = { id: "wide", segmentation_confidence: .7, segmentation_mask: { polygon: quad },
    enhancement_applied: { correction: { crop_modes: modes } }, competing_photo_outline: outline };
  const mobile = { photo_id: "narrow", confidence: .6, segmentation_mask: { polygon: narrower },
    crop_modes: modes, competing_photo_outline: structuredClone(outline) };

  assert.equal(getCompetingPhotoOutline(mobile), mobile.competing_photo_outline);
  assert.deepEqual(getCompetingPhotoReviewOutlines([mobile, desktop]), [outline]);
  assert.deepEqual(getCompetingPhotoReviewOutlines([desktop]), []);
  assert.deepEqual(getCompetingPhotoReviewOutlines([desktop, { ...mobile, reviewed_non_print: true }]), []);
  assert.equal(getCompetingPhotoOutline({ ...mobile, review_reasons: ["GEOMETRY_REVIEW_WRONG_OBJECT"] }), null);
  assert.deepEqual(getCompetingPhotoReviewOutlines([desktop, { ...mobile, confidence: null }]), []);
  assert.deepEqual(getCompetingPhotoReviewOutlines([desktop, { ...mobile, segmentation_mask: { polygon: quad } }]), []);
  assert.deepEqual(getCompetingPhotoReviewOutlines([desktop, { ...mobile, crop_modes: { ...modes, image_only: { status: "ready" } } }]), []);
  const changedPair = { ...mobile, competing_photo_outline: { ...outline, wider: quad.slice(1) } };
  assert.deepEqual(getCompetingPhotoReviewOutlines([desktop, changedPair]), []);

  const motifGuide = quad.map(({ x, y }) => ({ x: .03 + .90 * x, y: .03 + .90 * y }));
  const measured = { ...outline, motif_guide: motifGuide };
  const withGuide = [desktop, mobile].map(row => ({ ...row, competing_photo_outline: structuredClone(measured) }));
  assert.deepEqual(getCompetingPhotoReviewOutlines(withGuide), [{ ...measured, motifGuide }]);
  assert.deepEqual(getCompetingPhotoReviewOutlines([withGuide[0], mobile]), []);
  assert.deepEqual(getCompetingPhotoOutline({ ...mobile, segmentation_mask: {
    ...mobile.segmentation_mask, shared_motif_review: { outline: motifGuide },
  } }), outline);
});

test("competing outline paths share the exact source clip and leave exclusive regions unquantized", () => {
  const wider = [{ x: .1, y: .2 }, { x: .7, y: .2 }, { x: .4, y: .47 }, { x: .7, y: .8 }, { x: .1, y: .8 }];
  const narrower = [{ x: .3, y: .3 }, { x: .8, y: .3 }, { x: .8, y: .6 }, { x: .3, y: .6 }];
  const [shared, wideOnly, narrowOnly] = competingPhotoRegionPaths({ wider, narrower }, 200, 100);
  const widePath = "M 20 20 L 140 20 L 80 47 L 140 80 L 20 80 Z";
  const narrowPath = "M 60 30 L 160 30 L 160 60 L 60 60 Z";
  assert.deepEqual(shared, { kind: "shared", path: narrowPath, clip: widePath, rule: "nonzero" });
  assert.deepEqual(wideOnly, { kind: "wider", path: `${widePath} ${narrowPath}`, clip: widePath, rule: "evenodd" });
  assert.deepEqual(narrowOnly, { kind: "narrower", path: `${widePath} ${narrowPath}`, clip: narrowPath, rule: "evenodd" });
});

test("an album leaf can be excluded only while exactly one active matched child remains", () => {
  const traces = [[{ x: .51, y: .2 }, { x: .52, y: .8 }], [{ x: .56, y: .22 }, { x: .56, y: .79 }]];
  const held = { status: "review_needed" };
  const leaf = { id: "leaf", segmentation_confidence: .72, segmentation_mask: {
    album_leaf_review_traces: traces, album_leaf_review_child: quad,
  }, enhancement_applied: { correction: { crop_modes: { full_print: held, image_only: held } } } };
  const child = { photo_id: "print", confidence: .83, segmentation_mask: { polygon: structuredClone(quad) } };
  const [group, ...others] = getAlbumLeafReviewGroups([leaf, child]);
  assert.deepEqual(others, []);
  assert.equal(group.kind, "album_leaf");
  assert.equal(group.childId, child.photo_id);
  assert.equal(group.child, quad);
  assert.deepEqual(group.traces, traces);
  assert.deepEqual([...group.photoIds], [leaf.id]);

  const moved = structuredClone(child);
  moved.segmentation_mask.polygon[0].x += .01;
  const unheld = { ...leaf, crop_modes: { full_print: held, image_only: { status: "ready" } } };
  for (const rows of [[leaf], [leaf, moved], [leaf, { ...child, confidence: null }],
    [leaf, { ...child, reviewed_non_print: true }], [leaf, child, { ...child, photo_id: "duplicate" }], [unheld, child]]) {
    assert.deepEqual(getAlbumLeafReviewGroups(rows), []);
  }
});

test("an album leaf detail shows both rail centers with original context", () => {
  const traces = [[{ x: .4, y: .1 }, { x: .41, y: .82 }], [{ x: .48, y: .19 }, { x: .49, y: .89 }]];
  const { view: [left, top, width, height], ticks } = albumLeafSourceWindow(traces, 2048, 1536);
  assert.equal(ticks.length, 4);
  assert.ok(ticks.flat().every(([x, y]) => left < x && x < left + width && top < y && y < top + height));
  assert.ok(left >= 0 && top >= 0 && left + width <= 2048 && top + height <= 1536);
});

test("a shared outline appears once with the same three supported sides and separate weak edge", () => {
  const [outline, ...others] = getFragmentReviewOutlines([
    { id: "a", segmentation_confidence: .7, segmentation_mask: mask },
    { photo_id: "b", confidence: .8, segmentation_mask: structuredClone(mask) },
    { photo_id: "c", confidence: .9, segmentation_mask: mask },
  ]);
  assert.deepEqual(others, []);
  assert.deepEqual(outline.supportedSides, [quad[2], quad[3], quad[0], quad[1]]);
  assert.deepEqual(outline.weakEdge, [quad[1], quad[2]]);
  assert.deepEqual(outline.photoIds, new Set(["a", "b", "c"]));
});

test("resolved rows, repeats and conflicting sides cannot create a shared count outline", () => {
  const fragment = { id: "a", confidence: .7, segmentation_mask: mask };
  assert.deepEqual(getFragmentReviewOutlines([
    fragment,
    { ...fragment, photo_id: "a" },
    { ...fragment, id: "rejected", reviewed_non_print: true },
    { ...fragment, id: "edited", confidence: null },
    { ...fragment, id: "desktop-edited", segmentation_confidence: null },
    { ...fragment, id: "conflict", segmentation_mask: { fragment_outline_suggestion: { quad, uncertain_edge: 3 } } },
    { ...fragment, id: "invalid", segmentation_mask: { fragment_outline_suggestion: { quad, uncertain_edge: 9 } } },
  ]), []);
});

test("a film divider appears once while multiple active proposals still mark it", () => {
  const line = [{ x: .1, y: .48 }, { x: .7, y: .49 }];
  const row = { id: "a", confidence: .7, segmentation_mask: { film_grid_review_seam: line } };
  const second = { photo_id: "b", segmentation_confidence: .8, segmentation_mask: structuredClone(row.segmentation_mask) };
  const [group, ...others] = getFilmGridReviewSeams([row, second, { ...row, id: "c", reviewed_non_print: true }]);
  assert.deepEqual(others, []);
  assert.deepEqual(group.line, line);
  assert.deepEqual(group.photoIds, new Set(["a", "b"]));
  assert.deepEqual(getFilmGridReviewSeams([
    row, row, { ...second, segmentation_confidence: null },
    { ...row, id: "invalid", segmentation_mask: { film_grid_review_seam: [line[0], { x: 1.2, y: .5 }] } },
  ]), []);
});

test("only a rim shared by all of the active film proposals is displayed", () => {
  const line = [{ x: .1, y: .48 }, { x: .7, y: .49 }];
  const rim = [{ x: .04, y: .2 }, { x: .03, y: .8 }];
  const rows = ["a", "b", "c"].map(id => ({ id, confidence: .7,
    segmentation_mask: { film_grid_review_seam: line, film_grid_review_rim: structuredClone(rim) } }));
  const [agreed] = getFilmGridReviewSeams(rows);
  assert.deepEqual(agreed.rim, rim);
  assert.deepEqual(agreed.photoIds, new Set(["a", "b", "c"]));

  rows[1].segmentation_mask.film_grid_review_rim = [{ x: -.02, y: .2 }, rim[1]];
  const [disputed] = getFilmGridReviewSeams(rows);
  assert.equal(disputed.rim, null);
  assert.deepEqual(disputed.photoIds, agreed.photoIds);
});

test("a film exclusion is offered only when all three proposals have both crop modes held", () => {
  const seam = [{ x: .25, y: .48 }, { x: .75, y: .49 }];
  const mode = { status: "review_needed" };
  const cropModes = { full_print: mode, image_only: mode };
  const rows = ["carrier", "top", "bottom"].map(photo_id => ({ photo_id, confidence: .77,
    segmentation_mask: { film_grid_review_seam: seam }, crop_modes: cropModes }));
  const groups = getFilmGridReviewSeams(rows);
  assert.deepEqual(getActionableFilmGridReviewGroups(groups, rows), groups);
  assert.deepEqual(getActionableFilmGridReviewGroups(groups, rows.slice(0, 2)), []);
  const desktop = { id: rows[1].photo_id, enhancement_applied: { correction: { crop_modes: cropModes } } };
  assert.deepEqual(getActionableFilmGridReviewGroups(groups, [rows[0], desktop, rows[2]]), groups);
  const ready = { ...rows[2], crop_modes: { ...cropModes, image_only: { status: "ready" } } };
  assert.deepEqual(getActionableFilmGridReviewGroups(groups, [rows[0], rows[1], ready]), []);
});

test("film detail uses native pixel centers and keeps a near-edge window in the source", () => {
  const seam = [{ x: .25, y: .48 }, { x: .75, y: .52 }];
  const { view: [left, top, width, height], ticks } = filmSourceWindow(seam, 4000, 3000);
  assert.deepEqual([left + width / 2, top + height / 2, width, height], [2000, 1500, 390, 260]);
  assert.ok(ticks[0].every(([x, y]) => x < 1930 && y < 1500));
  assert.ok(ticks[1].every(([x, y]) => x > 2070 && y > 1500));
  const nearEdge = [{ x: .01, y: .26 }, { x: .01, y: .71 }];
  assert.equal(filmSourceWindow(nearEdge, 1346, 1014).view[0], 0);
  assert.equal(filmSourceWindow([nearEdge[0], nearEdge[0]], 1346, 1014), null);
  assert.equal(filmSourceWindow(null, 1346, 1014), null);
  assert.equal(filmSourceWindow(seam, 0, 1014), null);
});

test("a shared capture corner shows only its measured sides while proposals are unresolved", () => {
  const traces = [[{ x: 0, y: .65 }, { x: .6, y: .66 }], [{ x: .6, y: .66 }, { x: .59, y: 1 }]];
  const first = { id: "a", segmentation_confidence: .7, segmentation_mask: { capture_split_review_traces: traces } };
  const second = { photo_id: "b", confidence: .8, segmentation_mask: structuredClone(first.segmentation_mask) };
  const [group, ...others] = getCaptureSplitReviewGroups([first, second]);
  assert.deepEqual(others, []);
  assert.deepEqual(group.traces, traces);
  assert.deepEqual(group.photoIds, new Set(["a", "b"]));
  assert.deepEqual(getCaptureSplitReviewGroups([
    first, first, { ...second, reviewed_non_print: true },
    { ...second, photo_id: "manual", confidence: null },
    { ...second, photo_id: "invalid", segmentation_mask: { capture_split_review_traces: traces.slice(0, 1) } },
  ]), []);
});

test("the original capture corner window keeps the full visible area and shows two physical sides", () => {
  const joint = { x: 0.5276969075202942, y: 0.7033323645591736 };
  const side = { x: 0, y: 0.7024451494216919 }, row = { x: 0.5266348719596863, y: 1 };
  const traces = [[side, joint], [joint, row]];
  const { view, lines } = captureSplitSourceWindow(traces, 2179, 3000);
  assert.deepEqual(view, [0, 2035, 1222, 965]);
  assert.deepEqual(lines[0][1], lines[1][0]);
  assert.equal(lines[0][0][0], .5);
  assert.equal(lines[1][1][1], 2999.5);
  assert.deepEqual(captureSplitSourceWindow(traces.map(line => [...line].reverse()), 2179, 3000).view, view);

  const opposite = traces.map(line => line.map(({ x, y }) => ({ x: 1 - x, y: 1 - y })));
  assert.deepEqual(captureSplitSourceWindow(opposite, 2179, 3000).view, [957, 0, 1222, 965]);
  assert.equal(captureSplitSourceWindow([[side, joint], [joint, { x: 0, y: .8 }]], 2179, 3000), null);
  assert.equal(captureSplitSourceWindow([[side, joint], [joint, { x: 0, y: 1 }]], 2179, 3000), null);
  assert.equal(captureSplitSourceWindow([[side, joint], [side, joint]], 2179, 3000), null);
  assert.equal(captureSplitSourceWindow([[side, joint], [{ x: .52, y: .70 }, row]], 2179, 3000), null);
  assert.equal(captureSplitSourceWindow(traces, 0, 3000), null);
});

test("a paper join links exactly two active automatic proposals with the same native traces", () => {
  const traces = [[{ x: .43, y: .28 }, { x: .56, y: .28 }], [{ x: .43, y: .72 }, { x: .56, y: .72 }]];
  const first = { id: "first", confidence: .74, segmentation_mask: { paper_join_review_traces: traces } };
  const second = { ...first, id: "second", segmentation_mask: structuredClone(first.segmentation_mask) };
  const archived = { ...second, id: "archived", reviewed_non_print: true };
  const [group, ...others] = getPaperJoinReviewGroups([first, second, archived]);
  assert.deepEqual(others, []);
  assert.deepEqual(group.photoIds, new Set(["first", "second"]));
  assert.deepEqual(group.traces, traces);
  assert.deepEqual(getPaperJoinReviewGroups([first, second, { ...second, id: "third" }]), []);
  assert.deepEqual(getPaperJoinReviewGroups([first, { ...second, confidence: null }]), []);
});

test("each paper-grid cut follows its own active proposal", () => {
  const firstCut = [{ x: .15, y: .4 }, { x: .65, y: .4 }];
  const secondCut = [{ x: .15, y: .55 }, { x: .65, y: .55 }];
  const parent = { id: "outer", segmentation_confidence: .7, segmentation_mask: { paper_grid_review: { outline: quad } } };
  const fragment = (id, cut) => ({ photo_id: id, confidence: .8, segmentation_mask: { paper_grid_review: { outline: quad, cut } } });
  const first = fragment("first", firstCut);
  const second = fragment("second", secondCut);
  const [group, ...others] = getPaperGridReviewGroups([parent, first, second]);
  assert.deepEqual(others, []);
  assert.deepEqual(group.outline, quad);
  assert.deepEqual(group.cuts, new Map([["first", firstCut], ["second", secondCut]]));
  assert.deepEqual(getPaperGridReviewGroups([parent, { ...first, reviewed_non_print: true }, second])[0].cuts,
    new Map([["second", secondCut]]));
  assert.deepEqual(getPaperGridReviewGroups([parent, { ...first, confidence: null }]), []);
  assert.deepEqual(getPaperGridReviewGroups([{ ...parent, reviewed_non_print: true }, first, second]), []);
  assert.deepEqual(getPaperGridReviewGroups([parent, { ...parent, id: "other-outer" }, first, second]), []);
});

test("an accepted paper-grid count leaves its held parent outline available", () => {
  const held = { status: "review_needed" };
  const parent = {
    photo_id: "paper", confidence: .8, segmentation_mask: { paper_grid_review: { outline: quad } },
    crop_modes: { full_print: { ...held, suggested_warp_quad: quad }, image_only: held },
  };
  const fragment = (id, y) => ({
    photo_id: id, confidence: .6, segmentation_mask: { paper_grid_review: { outline: quad, cut: [{ x: .2, y }, { x: .6, y }] } },
    crop_modes: { full_print: held, image_only: held },
  });
  const first = fragment("first", .4);
  const second = fragment("second", .5);
  const active = [parent, first, second];
  const groups = getPaperGridReviewGroups(active);
  assert.deepEqual(getActionablePaperGridReviewGroups(groups, active), groups);
  const unheld = { ...first, crop_modes: { full_print: held, image_only: { status: "ready" } } };
  assert.deepEqual(getActionablePaperGridReviewGroups(groups, [parent, unheld, second]), []);
  const confirmed = [parent, { ...first, reviewed_non_print: true }, { ...second, reviewed_non_print: true }];
  assert.deepEqual(getPaperGridReviewGroups(confirmed), []);
  assert.deepEqual(getPaperGridOutlineParents(confirmed), [parent]);
});

test("a saved parent can confirm the same measured grid while its active fragment is held", () => {
  const held = { status: "review_needed" };
  const parent = { id: "paper", segmentation_confidence: null, manual_paper_grid_count_review: { outline: quad },
    enhancement_applied: { correction: { crop_modes: { full_print: { ...held, warp_quad: quad }, image_only: held } } } };
  const child = { photo_id: "first", confidence: .65, segmentation_mask: { paper_grid_review: { outline: quad, cut: quad.slice(0, 2) } },
    crop_modes: { full_print: held, image_only: held } };
  const active = [parent, child];
  const groups = getPaperGridReviewGroups(active);
  assert.equal(groups.length, 1);
  assert.deepEqual(getActionablePaperGridReviewGroups(groups, active), groups);
  assert.deepEqual(getPaperGridOutlineParents(active), []);
  assert.deepEqual(getPaperGridReviewGroups([{ ...parent, manual_paper_grid_count_review: null }, child]), []);
  assert.deepEqual(getPaperGridReviewGroups([parent, { ...child, reviewed_non_print: true }]), []);
});

test("paper-grid zoom keeps both ends and the middle of the native cut clear", () => {
  const line = [{ x: 724 / 1345, y: 500 / 1013 }, { x: 1242 / 1345, y: 500 / 1013 }];
  const windows = paperGridSourceWindows(line, 1346, 1014);
  assert.deepEqual(windows.map(({ view }) => view), [[655, 431, 140, 140], [914, 431, 140, 140], [1173, 431, 140, 140]]);
  for (const { ticks } of windows) {
    assert.ok(ticks[0].every(([, y]) => y <= 458.5));
    assert.ok(ticks[1].every(([, y]) => y >= 542.5));
  }
  const rotated = line.map(({ x, y }) => ({ x: 1 - y, y: x }));
  const [first] = paperGridSourceWindows(rotated, 1014, 1346);
  assert.deepEqual(first.view, [444, 655, 140, 140]);
  assert.ok(first.ticks[0].every(([x]) => x > 553));
  const atFrame = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
  const frameWindows = paperGridSourceWindows(atFrame, 1346, 1014);
  assert.deepEqual(frameWindows[0].view, [0, 0, 144, 144]);
  assert.deepEqual(frameWindows[2].view, [1202, 870, 144, 144]);
  assert.deepEqual(paperGridSourceWindows([line[0], line[0]], 1346, 1014), []);
  assert.deepEqual(paperGridSourceWindows(line, 0, 1014), []);
});

test("a saved aspect review follows only the unchanged whole-print draft", () => {
  const savedUrl = "/api/v1/photos/paper/image?variant=saved_full_print_review&v=version";
  const photo = { saved_full_print_review_url: savedUrl, segmentation_mask: { orientation_degrees: 90 } };
  const noSize = { width: "", height: "" };
  assert.equal(savedFullPrintReviewUrl(photo, quad, structuredClone(quad), noSize, 90), savedUrl);
  const moved = [{ ...quad[0], x: .101 }, ...quad.slice(1)];
  assert.equal(savedFullPrintReviewUrl(photo, quad, moved, noSize, 90), null);
  assert.equal(savedFullPrintReviewUrl(photo, quad, quad, { ...noSize, width: "20" }, 90), null);
  assert.equal(savedFullPrintReviewUrl(photo, quad, quad, noSize, 180), null);
  assert.equal(savedFullPrintReviewUrl({ ...photo, reviewed_non_print: true }, quad, quad, noSize, 90), null);
  assert.equal(savedFullPrintReviewUrl({ ...photo, saved_full_print_review_url: null }, quad, quad, noSize, 90), null);
});
