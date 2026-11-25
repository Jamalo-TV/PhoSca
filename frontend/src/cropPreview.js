export function samplePreviewSource(image) {
  const scale = Math.min(1, 1800 / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { alpha: false });
  context.fillStyle = "#fff";
  context.fillRect(0, 0, width, height);
  context.drawImage(image, 0, 0, width, height);
  return context.getImageData(0, 0, width, height);
}

export function resamplePerspective(pixels, corners, width, height, orientation) {
  const [a, b, c, d] = corners;
  const dx1 = b.x - c.x;
  const dx2 = d.x - c.x;
  const dy1 = b.y - c.y;
  const dy2 = d.y - c.y;
  const dx = a.x - b.x + c.x - d.x;
  const dy = a.y - b.y + c.y - d.y;
  const determinant = dx1 * dy2 - dx2 * dy1;
  if (determinant === 0) return null;
  // Map the output rectangle to the source quadrilateral by one homography.
  const g = (dx * dy2 - dx2 * dy) / determinant;
  const h = (dx1 * dy - dx * dy1) / determinant;
  const xAlong = b.x * (1 + g) - a.x;
  const xDown = d.x * (1 + h) - a.x;
  const yAlong = b.y * (1 + g) - a.y;
  const yDown = d.y * (1 + h) - a.y;

  const turns = (orientation ?? 0) / 90;
  const outputWidth = turns % 2 ? height : width;
  const outputHeight = turns % 2 ? width : height;
  const rowLength = outputWidth * 4;
  const data = new Uint8ClampedArray(rowLength * outputHeight);
  const [origin, across, down] = [
    [0, 4, rowLength],
    [(outputWidth - 1) * 4, rowLength, -4],
    [data.length - 4, -4, -rowLength],
    [(outputHeight - 1) * rowLength, -rowLength, 4],
  ][turns];
  const maxX = pixels.width - 1;
  const maxY = pixels.height - 1;
  const sourceStride = pixels.width * 4;
  const source = pixels.data;
  const uStep = width > 1 ? 1 / (width - 1) : 0;
  const uStart = width > 1 ? 0 : .5;
  for (let row = 0; row < height; row++) {
    const v = height > 1 ? row / (height - 1) : .5;
    const denominator = 1 + h * v;
    const xStart = a.x + xDown * v;
    const yStart = a.y + yDown * v;
    let target = origin + row * down;
    for (let column = 0; column < width; column++, target += across) {
      const u = uStart + column * uStep;
      const scale = 1 / (denominator + g * u);
      const x = Math.max(0, Math.min(maxX, (xStart + xAlong * u) * scale));
      const y = Math.max(0, Math.min(maxY, (yStart + yAlong * u) * scale));
      const left = Math.floor(x);
      const top = Math.floor(y);
      const xWeight = x - left;
      const yWeight = y - top;
      const nextColumn = left < maxX ? 4 : 0;
      const tl = top * sourceStride + 4 * left;
      const tr = tl + nextColumn;
      const bl = tl + (top < maxY ? sourceStride : 0);
      const br = bl + nextColumn;
      const topLeft = (1 - xWeight) * (1 - yWeight);
      const topRight = xWeight * (1 - yWeight);
      const bottomLeft = (1 - xWeight) * yWeight;
      const bottomRight = xWeight * yWeight;
      for (let channel = 0; channel < 3; channel++) {
        data[target + channel] = source[tl + channel] * topLeft + source[tr + channel] * topRight
          + source[bl + channel] * bottomLeft + source[br + channel] * bottomRight;
      }
      data[target + 3] = 255;
    }
  }
  return { data, width: outputWidth, height: outputHeight };
}
