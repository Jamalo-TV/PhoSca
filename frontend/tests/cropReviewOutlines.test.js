import assert from "node:assert/strict";
import test from "node:test";
import {
  getCaptureSplitReviewGroups, getFilmGridReviewSeams, getFragmentReviewOutlines, getPaperGridReviewGroups,
  getStoredMotifQuad,
} from "../src/cropReviewOutlines.js";

const quad = [{ x: .1, y: .2 }, { x: .7, y: .2 }, { x: .7, y: .8 }, { x: .1, y: .8 }];
const mask = { fragment_outline_suggestion: { quad, uncertain_edge: 1 } };

test("held motif geometry requires selection while a saved or ready motif can be edited", () => {
  const held = { status: "review_needed", warp_quad: quad, suggested_warp_quad: quad };
  assert.equal(getStoredMotifQuad({ image_quad: quad }, held), null);
  assert.equal(getStoredMotifQuad({ segmentation_mask: { image_quad: quad } }, held), quad);
  assert.equal(getStoredMotifQuad({}, { status: "ready", warp_quad: quad }), quad);
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
});
