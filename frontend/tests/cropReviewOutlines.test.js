import assert from "node:assert/strict";
import test from "node:test";
import { getFilmGridReviewSeams, getFragmentReviewOutlines } from "../src/cropReviewOutlines.js";

const quad = [{ x: .1, y: .2 }, { x: .7, y: .2 }, { x: .7, y: .8 }, { x: .1, y: .8 }];
const mask = { fragment_outline_suggestion: { quad, uncertain_edge: 1 } };

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
