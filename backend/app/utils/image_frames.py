from xml.etree import ElementTree  # nosec B405

from PIL import Image


def primary_frame_policy(image: Image.Image) -> str:
    """Accept one page image, including the SDR base of a JPEG with an HDR gain map."""
    frames = int(getattr(image, "n_frames", 1))
    if frames == 1:
        return "single_frame"
    # Pillow can expose the auxiliary gain map as the second frame of an MPO.
    if image.format == "MPO" and frames == 2:
        for marker, payload in getattr(image, "applist", ()):
            header, _, xmp = payload.partition(b"\x00")
            if marker != "APP1" or header != b"http://ns.adobe.com/xap/1.0/":
                continue
            try:
                markup = xmp.decode("utf-8")
                # Reject DTDs before parsing XMP to prevent custom entity expansion.
                if "<!DOCTYPE" in markup:
                    continue
                root = ElementTree.fromstring(markup)  # nosec B314
            except (UnicodeDecodeError, ElementTree.ParseError):
                continue
            item = "{http://ns.google.com/photos/1.0/container/}Item"
            semantic = "{http://ns.google.com/photos/1.0/container/item/}Semantic"
            if [entry.get(semantic) for entry in root.iter(item)][:2] == ["Primary", "GainMap"]:
                return "hdr_gain_map_base"
    raise ValueError("multi-frame images are not supported")
