# Cropping, Segmentation, And Orientation Strategy

Date: 2026-06-22

This note captures the failure analysis and the selected upgrade path for PhoSca's album-page photo extraction pipeline. It is intentionally paired with the runtime work in `backend/app/pipeline/segmentation.py`, `backend/app/pipeline/perspective.py`, and `backend/app/pipeline/orientation.py`.

## Why The Original Outputs Failed

### 1. Whole album page or album border is cropped as a photo

The old failure mode is a detector-contract mismatch. A bounding-box-only detector can say where a plausible object is, but it cannot describe the true quadrilateral or mask of a printed photo. If the model is absent or weak, the classical fallback sees large high-contrast contours: album-page edges, page mats, scanned borders, and groups of multiple photos can all look like a strong rectangular candidate. Once that candidate is accepted, perspective correction faithfully warps the selected rectangle, even if the selected rectangle is the album page instead of a print.

The fix is not just a sharper crop. The detector must produce instance-level masks or polygons, and the fallback must reject page-sized/container candidates when several smaller photo-like regions exist inside them.

### 2. A tall sky/cloud sliver is detected as a photo

This is a false-positive scoring issue. A saturated vertical strip can have enough contrast or texture to resemble a standalone print when the heuristic only looks at contours, local color transitions, and rough area/aspect thresholds. Without semantic validation, interior texture checks, edge support, and stricter aspect logic, a smooth scenic sliver can pass as a photo.

The fix is a combination of stricter candidate assessment and reviewable metadata: edge support, border contrast, rectangularity, interior texture, aspect sanity, detector source, and rejection reasons.

### 3. A correctly cropped photo remains rotated 90 degrees

Perspective correction solves geometry, not semantic orientation. It can make the print rectangular and remove uniform borders, but it does not know which way human faces, text, horizon cues, or scene composition should face. EXIF orientation is not enough for album scans because the extracted photo is a region inside a larger page image.

The fix is a post-crop orientation stage with a conservative fallback heuristic and an optional trained 0/90/180/270 classifier.

## Research Takeaways

Instance segmentation is the right model contract for this problem because it returns per-object masks or contours, not only upright boxes. Ultralytics' segmentation task documentation describes instance segmentation as producing masks or contours for each object plus class/confidence, and its result API exposes normalized polygon masks via `result.masks.xyn`. It also documents ONNX export for segmentation models, which matches PhoSca's deployment target.

SAM-style models are valuable, but they should not replace the production detector by themselves. The original Segment Anything project introduced a promptable foundation model trained with a very large mask dataset, useful for zero-shot and interactive segmentation. SAM 2 improves the promptable image/video segmentation family and reports faster, more accurate image segmentation than the original SAM. For PhoSca, that makes SAM/SAM 2 best as an annotation accelerator or mask refiner: detector boxes or rough polygons can prompt SAM, reviewers correct masks, and the reviewed labels can train a smaller dedicated YOLO segmentation model.

OCR orientation tools are not sufficient for all extracted photos. PaddleOCR-style angle classification is helpful for text snippets and captions, but many album photos have no readable text. PhoSca therefore needs its own photo orientation path, with OCR/text cues treated as extra evidence when available rather than as the only signal.

## Top Two Designs

### Option 1: Dedicated YOLO Segmentation Model With Review-Gated Training

This is the primary production path now wired into the repo.

Architecture:

1. Prepare labels from Label Studio or PhoSca's own review canvas.
2. Validate YOLO polygon labels before training.
3. Train a one-class YOLO segmentation model for `photo` instances.
4. Export to ONNX for backend inference.
5. Parse masks/prototypes at runtime and store polygons on `ExtractedPhoto.segmentation_mask`.
6. Fall back to classical polygon extraction only when the model is missing or produces no usable detections.
7. Crop/perspective-correct from the polygon, trim uniform borders, then run orientation correction.
8. Route low confidence, weak geometry, or unverified OCR to review.

Why this is the recommended default:

- Fast enough for local backend processing.
- Produces exact masks/polygons instead of upright boxes.
- Easy to export to ONNX and run without a GPU.
- Human corrections feed directly back into training data.
- Review and validation are measurable with mean polygon IoU and OCR CER.

Tradeoffs:

- Requires real album-page polygon labels.
- Needs enough negative examples and difficult pages to suppress page-border and sliver false positives.
- Needs retraining when album styles shift substantially.

Repo support currently present:

- YOLO-seg ONNX parser with prototype mask support.
- Candidate rejection in the classical fallback, including nested partial-photo suppression after corner refinement.
- `data/segmentation_training/` plus `scripts/setup_segmentation_training.py` for a low-friction manual image, Label Studio, dataset, validation, and training loop.
- Deterministic train/val/golden split preparation.
- Dataset validator for polygon labels and golden leakage.
- Label Studio task export, Label Studio JSON conversion, and in-app reviewed-mask export.
- Segmentation IoU validation from saved polygons.
- Optional photo orientation classifier trainer and runtime hook.

### Option 2: SAM-Assisted Active Learning And Mask Refinement

This is the higher-accuracy annotation acceleration path, not the simplest production runtime.

Architecture:

1. Use the current classical detector or a weak YOLO model to propose photo boxes/polygons.
2. Prompt SAM/SAM 2 with those boxes and points to refine masks.
3. Present refined masks in Label Studio or PhoSca's review canvas.
4. Export reviewed masks to YOLO labels.
5. Train/distill the dedicated YOLO segmentation model used by production.
6. Keep SAM as an offline annotation/refinement tool, not a required runtime dependency.

Why this is attractive:

- Can reduce manual polygon work on irregular, overlapping, or low-contrast prints.
- Handles exact boundaries better than rectangle-only methods when prompted well.
- Fits active learning: repeatedly annotate failures, retrain, and re-run smoke/regression gates.

Tradeoffs:

- Heavier dependency and model footprint.
- Prompt quality matters; automatic "segment everything" can oversegment album backgrounds.
- More complex to package for a local-first app.
- Still needs human review for historical albums with handwriting, frames, reflections, and overlapping prints.

## Completion Criteria For This Subsystem

PhoSca should not be considered production-complete for cropping/orientation until current evidence proves all of the following:

- `data/label_exports/yolo/*.txt` contains reviewed polygon labels for all train/val pages.
- `data/golden_fixtures/labels/*.txt` contains locked labels for all golden pages.
- `python scripts/validate_yolo_dataset.py --data data/data.yaml` passes.
- `models/yolov8-seg-album.onnx` exists and is produced by the documented training path.
- `models/photo-orientation.onnx` exists or the heuristic fallback is explicitly accepted for the target albums.
- `python scripts/validate_segmentation.py --album-id <album-id>` reports mean IoU greater than `0.85` on golden pages.
- `python scripts/validate_ocr.py --album-id <album-id>` reports mean CER below `0.10` once OCR ground truth is filled.
- `python -m pytest -q`, frontend build, and the real-data smoke path pass in the current environment.

## Sources

- Ultralytics instance segmentation docs: https://docs.ultralytics.com/tasks/segment/
- Segment Anything paper: https://arxiv.org/abs/2304.02643
- SAM 2 paper: https://arxiv.org/abs/2408.00714
- Segment Anything GitHub repository: https://github.com/facebookresearch/segment-anything
