import assert from "node:assert/strict";
import test from "node:test";
import { patchBoundingBox } from "../src/api.js";

test("the regular geometry PATCH keeps both album mark offer formats compatible", async (context) => {
  const requests = [];
  context.mock.method(globalThis, "fetch", async (path, options) => {
    requests.push({ path, method: options.method, body: JSON.parse(options.body) });
    return new Response("{}");
  });
  const box = { x1: 0.1, y1: 0.2, x2: 0.8, y2: 0.9 };
  const mask = { warp_quad: [
    { x: box.x1, y: box.y1 }, { x: box.x2, y: box.y1 },
    { x: box.x2, y: box.y2 }, { x: box.x1, y: box.y2 },
  ] };
  const ordinary = { bounding_box: box, segmentation_mask: mask };

  await patchBoundingBox("photo", box, mask);
  await patchBoundingBox("photo", box, mask, true, "legacy-offer");
  await patchBoundingBox("photo", box, mask, true, "current-offer", [0, 1]);
  await patchBoundingBox("photo", box, mask, false, null, [0, 1]);

  assert.deepEqual(requests, [
    { path: "/api/v1/photos/photo/bounding-box", method: "PATCH", body: ordinary },
    { path: "/api/v1/photos/photo/bounding-box", method: "PATCH",
      body: { ...ordinary, confirm_print_outline: true, album_page_mark_review_id: "legacy-offer" } },
    { path: "/api/v1/photos/photo/bounding-box", method: "PATCH",
      body: { ...ordinary, confirm_print_outline: true, album_page_mark_review_id: "current-offer",
        album_page_mark_review_indices: [0, 1] } },
    { path: "/api/v1/photos/photo/bounding-box", method: "PATCH", body: ordinary },
  ]);
});
