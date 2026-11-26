import assert from "node:assert/strict";
import { test } from "node:test";
import { captureCameraImage } from "../src/cameraCapture.js";

function camera() {
  const track = { readyState: "live", muted: false };
  const video = { srcObject: { getVideoTracks: () => [track] }, videoWidth: 1280, videoHeight: 720, readyState: 2 };
  return { track, video };
}

function stubGlobal(context, key, value) {
  const original = Object.getOwnPropertyDescriptor(globalThis, key);
  Object.defineProperty(globalThis, key, { configurable: true, value });
  context.after(() => {
    if (original) Object.defineProperty(globalThis, key, original);
    else delete globalThis[key];
  });
}

test("camera still returns the original native image bytes and media type", async (context) => {
  const { track, video } = camera();
  const source = new Uint8Array([91, 73, 2, 172]);
  let type = "image/jpeg";
  stubGlobal(context, "ImageCapture", class {
    constructor(received) { assert.equal(received, track); }
    async takePhoto() { return new Blob([source], { type }); }
  });
  for (const [mime, extension] of [["image/jpeg", "jpg"], ["image/png", "png"]]) {
    type = mime;
    const { file, fromVideo } = await captureCameraImage(video, .96);
    assert.equal(fromVideo, false);
    assert.equal(file.type, mime);
    assert.ok(file.name.endsWith(`.${extension}`));
    assert.deepEqual(new Uint8Array(await file.arrayBuffer()), source);
  }
});

test("missing, rejected or unsupported still capture falls back at the actual video dimensions", async (context) => {
  const { video } = camera();
  const source = new Uint8Array([24, 10, 8]);
  const canvas = {
    getContext(kind, options) {
      assert.equal(kind, "2d");
      assert.equal(options.alpha, false);
      return { drawImage: (...args) => assert.deepEqual(args, [video, 0, 0]) };
    },
    toBlob(write, type, quality) {
      assert.equal(type, "image/jpeg");
      assert.equal(quality, .92);
      write(new Blob([source], { type }));
    },
  };
  stubGlobal(context, "document", { createElement: () => canvas });

  const cases = [
    ["missing ImageCapture", undefined],
    ["rejected exposure", class { async takePhoto() { throw new Error("unsupported camera"); } }],
    ["unsupported image type", class { async takePhoto() { return new Blob(["WebP"], { type: "image/webp" }); } }],
  ];
  for (const [name, native] of cases) {
    await context.test(name, async (scenario) => {
      stubGlobal(scenario, "ImageCapture", native);
      const { file, fromVideo } = await captureCameraImage(video, .92);
      assert.equal(fromVideo, true);
      assert.equal(canvas.width, 1280);
      assert.equal(canvas.height, 720);
      assert.deepEqual(new Uint8Array(await file.arrayBuffer()), source);
    });
  }
});

test("a still exposure from a stopped or replaced camera is discarded", async (context) => {
  for (const reason of ["ended", "replaced"]) {
    await context.test(reason, async (scenario) => {
      const { track, video } = camera();
      stubGlobal(scenario, "ImageCapture", class {
        async takePhoto() {
          if (reason === "ended") track.readyState = "ended";
          else video.srcObject = camera().video.srcObject;
          return new Blob(["stale"], { type: "image/jpeg" });
        }
      });
      await assert.rejects(captureCameraImage(video, .96), /camera_not_ready/);
    });
  }
});
