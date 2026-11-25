"""Render the README examples using the current precise crop pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.evaluation.crop_fixtures import geometry_seed, synthetic_crop_pages
from app.pipeline.perspective import correct_photo_crop


def crop(page, box, mask):
    return correct_photo_crop(
        page, box, mask, dual_mode=True, auto_orient=False, enable_classical_retry=True,
    )


def faint_caption():
    scale = 2
    page = np.full((230 * scale, 340 * scale, 3), 236, np.uint8)
    page[80:384, 90:570] = 249
    y, x = np.indices((256, 432))
    page[104:360, 114:546] = np.stack([85 + x % 40, 45 + y % 46, 126 + (x + y) % 49], axis=2)
    ink = np.zeros(page.shape[:2], np.uint8)
    cv2.putText(ink, "SEP 1954", (162, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.52, 255, 2, cv2.LINE_AA)
    page -= np.uint8(np.rint(ink / 255 * 21))[..., None]
    points = [(84, 76), (574, 74), (572, 388), (86, 386)]
    mask = {"polygon": [{"x": x / 679, "y": y / 459} for x, y in points]}
    result = crop(page, {}, mask)
    state = result.metadata["crop_modes"]["image_only"]
    if result.image_only is None:
        return page, page[79:104, 137:282], None, "Motiv in Prüfung"

    rows, columns = np.nonzero(ink >= 90)
    pixels = np.column_stack((columns, rows)).astype(np.float32)
    mapped = cv2.perspectiveTransform(pixels[None], np.asarray(state["page_to_output"]))[0]
    height, width = result.image_only.shape[:2]
    kept = np.count_nonzero(((mapped >= 0) & (mapped <= [width - 1, height - 1])).all(axis=1))
    return page, page[79:104, 137:282], result.image_only, f"{kept}/{len(pixels)} Schriftpunkte erhalten"


def place_image(canvas, raster, left, top):
    if raster is None:
        return
    image = Image.fromarray(cv2.cvtColor(raster, cv2.COLOR_BGR2RGB))
    image.thumbnail((325, 177), Image.Resampling.LANCZOS)
    canvas.paste(image, (left + (350 - image.width) // 2, top + (191 - image.height) // 2))


def main(output: Path, regular: str, bold: str):
    cv2.setNumThreads(1)
    fixtures = {fixture.name: fixture for fixture in synthetic_crop_pages("locked")}
    tilted = fixtures["camera_tilt"]
    mounted = fixtures["paper_corner_mount"]
    straight = crop(tilted.image, *geometry_seed(tilted.prints[0], tilted.image.shape))
    holder = crop(mounted.image, *geometry_seed(mounted.prints[0], mounted.image.shape))
    assert all(mode["status"] == "ready" for mode in straight.metadata["crop_modes"].values())
    assert holder.metadata["crop_modes"]["full_print"]["status"] == "review_needed" and holder.image_only is not None
    page, detail, motif, result = faint_caption()
    rows = (
        ("Schräg fotografiert", (
            ("Albumausschnitt", tilted.image), ("Ganzer Abzug", straight.image), ("Nur Bildmotiv", straight.image_only),
        )),
        ("Fotoecke auf dem Papierrand", (
            ("Albumausschnitt", mounted.image), ("Ganzer Abzug in Prüfung", holder.image), ("Sichtbares Bildmotiv", holder.image_only),
        )),
        ("Blasse Beschriftung", (
            ("Albumausschnitt", page), ("Papierrand vergrössert", cv2.resize(detail, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)),
            (result, motif),
        )),
    )
    canvas = Image.new("RGB", (1168, 842), "#fbfaf7")
    drawing = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(bold, 34)
    heading = ImageFont.truetype(bold, 21)
    label = ImageFont.truetype(regular, 17)
    drawing.text((30, 20), "Zuschnittbeispiele aus der Testpipeline", fill="#172e2a", font=title)
    for index, (name, panels) in enumerate(rows):
        top = 74 + index * 251
        drawing.text((30, top), name, fill="#172e2a", font=heading)
        for column, (caption, image) in enumerate(panels):
            left = 30 + 382 * column
            drawing.rectangle((left, top + 33, left + 350, top + 223), fill="#e9eeeb", outline="#a5b3ac")
            place_image(canvas, image, left, top + 33)
            drawing.text((left, top + 228), caption, fill="#293e36", font=label)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/images/crop-examples.png"))
    parser.add_argument("--font", default="DejaVuSans.ttf")
    parser.add_argument("--bold-font", default="DejaVuSans-Bold.ttf")
    arguments = parser.parse_args()
    main(arguments.output, arguments.font, arguments.bold_font)
