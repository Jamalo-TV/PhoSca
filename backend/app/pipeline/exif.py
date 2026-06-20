from pathlib import Path

import piexif


DEVICE_TAGS = {
    "0th": [
        piexif.ImageIFD.Make,
        piexif.ImageIFD.Model,
        piexif.ImageIFD.Software,
        piexif.ImageIFD.Artist,
        piexif.ImageIFD.Copyright,
    ],
    "Exif": [
        piexif.ExifIFD.LensMake,
        piexif.ExifIFD.LensModel,
        piexif.ExifIFD.BodySerialNumber,
        piexif.ExifIFD.LensSerialNumber,
    ],
}


def write_caption_exif(image_path: Path, caption: str | None) -> dict:
    try:
        exif_dict = piexif.load(str(image_path))
    except Exception:  # noqa: BLE001 - corrupt or absent EXIF should be sanitized by replacing it.
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    exif_dict["GPS"] = {}
    for ifd_name, tags in DEVICE_TAGS.items():
        ifd = exif_dict.setdefault(ifd_name, {})
        for tag in tags:
            ifd.pop(tag, None)

    if caption:
        exif_dict.setdefault("0th", {})[piexif.ImageIFD.ImageDescription] = caption.encode("utf-8", errors="ignore")

    piexif.insert(piexif.dump(exif_dict), str(image_path))
    return {"gps_stripped": True, "device_tags_stripped": True, "caption_written": bool(caption)}

