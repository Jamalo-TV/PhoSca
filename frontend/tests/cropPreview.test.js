import assert from "node:assert/strict";
import test from "node:test";
import { projectedCropAspect, resamplePerspective } from "../src/cropPreview.js";

function coordinateGrid(width, height) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) data.set([20 * x, 20 * y, 180, 255], 4 * (y * width + x));
  }
  return { data, width, height };
}

function color(image, x, y) {
  const offset = 4 * (y * image.width + x);
  return [...image.data.slice(offset, offset + 4)];
}

test("the motif preview measures its aspect in the rectified paper plane", () => {
  const corners = [{ x: 0, y: 0 }, { x: 4, y: 0 }, { x: 3, y: 3 }, { x: 1, y: 3 }];
  const projection = [[1, -1 / 3, 0], [0, 1, 0], [0, -1 / 6, 1]];
  const rotated = [projection[1], [-1, 1 / 3, 0], projection[2]];
  assert.equal(projectedCropAspect(corners, projection), 2 / 3);
  assert.equal(projectedCropAspect(corners, rotated), 2 / 3);
});

test("a trapezoid is projected continuously instead of using two affine triangles", () => {
  const corners = [{ x: 2, y: 2 }, { x: 8, y: 2 }, { x: 6, y: 8 }, { x: 4, y: 8 }];
  const result = resamplePerspective(coordinateGrid(10, 10), corners, 5, 5, null);
  assert.deepEqual(color(result, 0, 0), [40, 40, 180, 255]);
  assert.deepEqual(color(result, 4, 4), [120, 160, 180, 255]);
  // The center of this destination lies at (5, 6.5) in the source.
  assert.deepEqual(color(result, 2, 2), [100, 130, 180, 255]);
  assert.equal(result.data.filter((_, i) => i % 4 === 2 && result.data[i] === 180).length, 25);
});

test("orientation maps every output corner and pixel through all quarter turns", () => {
  const source = coordinateGrid(2, 3);
  const corners = [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 2 }, { x: 0, y: 2 }];
  const indices = [[0, 1, 2, 3, 4, 5], [4, 2, 0, 5, 3, 1], [5, 4, 3, 2, 1, 0], [1, 3, 5, 0, 2, 4]];
  for (let turn = 0; turn < 4; turn++) {
    const result = resamplePerspective(source, corners, 2, 3, turn * 90);
    assert.equal(result.width, turn % 2 ? 3 : 2);
    assert.equal(result.height, turn % 2 ? 2 : 3);
    assert.deepEqual([...result.data], indices[turn].flatMap((i) => color(source, i % 2, Math.floor(i / 2))));
  }
});
