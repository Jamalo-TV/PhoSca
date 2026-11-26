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

export function projectedCropAspect(corners, matrix) {
  const projected = corners.map(({ x, y }) => {
    const [u, v, scale] = matrix.map(row => row[0] * x + row[1] * y + row[2]);
    return { x: u / scale, y: v / scale };
  });
  const lengths = projected.map((start, index) => {
    const end = projected[(index + 1) % projected.length];
    return Math.hypot(end.x - start.x, end.y - start.y);
  });
  const aspect = (lengths[0] + lengths[2]) / (lengths[1] + lengths[3]);
  return Number.isFinite(aspect) && aspect > 0 ? aspect : null;
}

function unitSquareProjection(corners) {
  if (corners?.length !== 4 || corners.some(point => !Number.isFinite(point?.x) || !Number.isFinite(point?.y))) {
    return null;
  }
  const [a, b, c, d] = corners;
  const dx1 = b.x - c.x;
  const dx2 = d.x - c.x;
  const dy1 = b.y - c.y;
  const dy2 = d.y - c.y;
  const dx = a.x - b.x + c.x - d.x;
  const dy = a.y - b.y + c.y - d.y;
  const determinant = dx1 * dy2 - dx2 * dy1;
  if (!Number.isFinite(determinant) || determinant === 0) return null;
  // Map the output rectangle to the source quadrilateral by one homography.
  const g = (dx * dy2 - dx2 * dy) / determinant;
  const h = (dx1 * dy - dx * dy1) / determinant;
  const xAlong = b.x * (1 + g) - a.x;
  const xDown = d.x * (1 + h) - a.x;
  const yAlong = b.y * (1 + g) - a.y;
  const yDown = d.y * (1 + h) - a.y;
  if (![1 + g, 1 + h, 1 + g + h].every(weight => Number.isFinite(weight) && weight > 0)) return null;
  return { a, g, h, xAlong, xDown, yAlong, yDown };
}

function sourceToSquare(point, { a, g, h, xAlong, xDown, yAlong, yDown }) {
  const a11 = xAlong - point.x * g;
  const a12 = xDown - point.x * h;
  const a21 = yAlong - point.y * g;
  const a22 = yDown - point.y * h;
  const dx = point.x - a.x;
  const dy = point.y - a.y;
  const determinant = a11 * a22 - a12 * a21;
  const x = (dx * a22 - a12 * dy) / determinant;
  const y = (a11 * dy - dx * a21) / determinant;
  return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
}

function squareToSource({ x, y }, { a, g, h, xAlong, xDown, yAlong, yDown }) {
  const weight = 1 + g * x + h * y;
  return { x: (a.x + xAlong * x + xDown * y) / weight, y: (a.y + yAlong * x + yDown * y) / weight };
}

const PAPER_AXIS_TOLERANCE = 1e-4;

function paperRectangle(paperQuad, motifQuad) {
  const projection = unitSquareProjection(paperQuad);
  if (!projection || motifQuad?.length !== 4 || motifQuad.some(point => !point)) return null;
  const corners = motifQuad.map(point => sourceToSquare(point, projection));
  if (corners.some(point => !point || [point.x, point.y].some(value => value < -1e-8 || value > 1 + 1e-8))) return null;
  const axes = corners.map((point, index) => {
    const next = corners[(index + 1) % 4];
    const dx = Math.abs(next.x - point.x);
    const dy = Math.abs(next.y - point.y);
    if (dx <= PAPER_AXIS_TOLERANCE && dy > PAPER_AXIS_TOLERANCE) return "y";
    if (dy <= PAPER_AXIS_TOLERANCE && dx > PAPER_AXIS_TOLERANCE) return "x";
    return null;
  });
  if (axes.includes(null) || axes[0] === axes[1] || axes[0] !== axes[2] || axes[1] !== axes[3]) return null;
  return { projection, corners, axes };
}

export function motifUsesPaperAxes(paperQuad, motifQuad) {
  return paperRectangle(paperQuad, motifQuad) !== null;
}

export function resizeMotifInPaperPlane(paperQuad, motifQuad, cornerIndex, destination) {
  const rectangle = paperRectangle(paperQuad, motifQuad);
  if (!rectangle || !Number.isInteger(cornerIndex) || cornerIndex < 0 || cornerIndex > 3 || !destination) return null;
  const { projection, corners, axes } = rectangle;
  const proposed = sourceToSquare(destination, projection);
  if (!proposed) return null;
  const start = corners[cornerIndex];
  const oppositeIndex = (cornerIndex + 2) % 4;
  const fixed = corners[oppositeIndex];
  const x = Math.max(0, Math.min(1, proposed.x));
  const y = Math.max(0, Math.min(1, proposed.y));
  if ((x - fixed.x) * Math.sign(start.x - fixed.x) <= PAPER_AXIS_TOLERANCE
      || (y - fixed.y) * Math.sign(start.y - fixed.y) <= PAPER_AXIS_TOLERANCE) return null;
  if (Math.abs(x - start.x) < 1e-10 && Math.abs(y - start.y) < 1e-10) return motifQuad;
  const next = axes[cornerIndex] === "x" ? { x: fixed.x, y } : { x, y: fixed.y };
  const previous = axes[cornerIndex] === "x" ? { x, y: fixed.y } : { x: fixed.x, y };
  const result = [...motifQuad];
  result[cornerIndex] = squareToSource({ x, y }, projection);
  result[(cornerIndex + 1) % 4] = squareToSource(next, projection);
  result[(cornerIndex + 3) % 4] = squareToSource(previous, projection);
  return result;
}

export function resamplePerspective(pixels, corners, width, height, orientation) {
  const projection = unitSquareProjection(corners);
  if (!projection) return null;
  const { a, g, h, xAlong, xDown, yAlong, yDown } = projection;

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
