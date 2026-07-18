# Cropping and Enhancement Master Plan

Status: implementation plan and execution record started 18 July 2026
Baseline commit: `46e94be`
Safety posture: immutable uploads, reversible variants, CPU-capable path,
derived-release kill switch on by default

## 1. Objective and release standard

PhoSca must extract the complete physical print, correct only geometry supported
by visible evidence, and improve readability without inventing content. A crop
or enhancement is not accepted because it looks plausible. It is accepted only
when its boundary, transform, content preservation, and fidelity gates pass.

The implementation is staged so that an expensive method is used only when a
cheap stage reports risk:

```text
Canonical image contract
  -> page/capture analysis
  -> model proposals
  -> risk-triggered classical evidence
  -> polygon-aware candidate fusion
  -> full-resolution boundary hypotheses
  -> transform and content-safety assessment
  -> source-corrected immutable crop
  -> degradation profile
  -> deterministic conservative plan
  -> independent quality critic
  -> optional labelled restoration candidates
  -> component confidence + reason codes
  -> source release or review
```

## 2. Reconstructed current pipeline

1. `routers/pages.py` and `services/uploads.py` validate and permanently store
   original bytes. `utils/file_security.py` enforces type, frame, size, and
   pixel limits.
2. `pipeline/image_ops.py::load_image` applies EXIF orientation, converts ICC
   data to sRGB when possible, and exposes BGR `uint8` working pixels.
3. `services/pipeline.py::ingestion_task_logic` verifies SHA-256 and records a
   blur warning.
4. `pipeline/segmentation.py::segment_album_page` runs registered YOLO ONNX
   proposals. The optional classical path uses contours, contrast regions, and
   line-supported quadrilaterals.
5. `services/pipeline.py::segmentation_task_logic` persists normalized boxes,
   polygons, confidence, calibration evidence, and initial review reasons.
6. `pipeline/perspective.py` reduces masks to a quadrilateral, validates basic
   geometry and homography conditioning, warps at source resolution, optionally
   trims a border, and optionally rotates orientation.
7. OCR is run on the page and spatially associated with prints.
8. The old enhancement route analyzes degradation, optionally denoises and
   corrects colour, and contains disabled model paths for LaMa, Real-ESRGAN, and
   GFPGAN. Variants and deterministic release gates are stored in JSON metadata.
9. Deduplication, EXIF persistence, asset lineage, and review queues complete
   the run.

The foundation correctly preserves upload bytes, uses registered model hashes,
keeps derived candidates reversible, and defaults to source release. Those
contracts must remain.

## 3. Baseline evidence

### Tests

At baseline, `python -m pytest -q` completed with **76 passed in 25.00 s**.
The suite did not run real ONNX inference and did not gate boundary loss,
corner error, retained background, orientation, or enhancement fidelity.

### Locked crop holdout

The existing ten tracked pages contain 29 manually verified print polygons.
At IoU 0.85, model-only runtime inference measured:

| Metric | Baseline |
|---|---:|
| Detection precision | 0.9655 |
| Detection recall | 0.9655 |
| Mean independent best-match IoU | 0.9730 |
| Page-perfect pages | 9 / 10 |
| False positives / false negatives | 1 / 1 |

The old always-fused model/classical mode was worse: precision 0.8485, recall
0.9655, mean best-match IoU 0.9668, and 6/10 page-perfect. On `page_31`, however,
the classical boundary for the bad print scored 0.876 IoU versus the model
mask's 0.652. The defect was therefore fusion/selection, not missing evidence.

All pages are from one narrow source context and were split by page, not album.
They are useful regression fixtures but not independent consumer evidence.

### Enhancement baseline limitations

No real paired clean/reference dataset exists. The prior analyzer changes
already-good, high-key monochrome, warm, and sepia-like synthetic inputs because
global range and Lab-b statistics are treated as fading/yellowing. The new
evaluation harness therefore separates:

- generated paired degradations with exact references;
- preservation fixtures where no-op is the expected result;
- future private real crops with descriptive no-reference safeguards and human
  inspection.

Synthetic improvements will not be reported as real historical-photo quality.

## 4. Root causes and planned corrections

### 4.1 Geometry and coordinate contracts

**Root cause:** normalized polygon vertices, half-open boxes, model input
coordinates, and source pixels are represented by untyped dictionaries. Edge
coordinate `1.0` can map outside the last valid vertex, and segmentation still
contains sum/difference corner ordering that duplicates a point near 45 degrees.

**Correction:** one typed geometry module defines coordinate spaces, reversible
normalization, cyclic ordering, winding, uniqueness, convexity, area, angles,
and boundary-safe clamping. Boxes remain half-open; polygon vertices map to
`[0, width - 1] x [0, height - 1]`.

### 4.2 Candidate generation and fusion

**Root cause:** box-IoU NMS can delete real overlapping photos, while classical
and model proposals with different boxes can survive as duplicate crops. Model
proposals receive less boundary scrutiny than classical proposals.

**Correction:** preserve model instances with mask/polygon-aware duplicate
checks. Invoke classical analysis for empty or risky model sets, cluster
cross-source proposals by polygon overlap/containment and corner compatibility,
and use classical geometry to refine a low-confidence model instance only when
independent boundary evidence is strong. Unmatched classical proposals require
strict four-sided evidence. Record every source, replacement, and rejection.

### 4.3 Boundary and perspective safety

**Root cause:** masks are forced into one quad. The existing reprojection test
fits and tests the same four points, so a semantically bad crop can obtain zero
reprojection error. Transform success does not measure content loss.

**Correction:** retain the dense visible polygon, prefer a supported convex
four-corner approximation, and otherwise use its conservative minimum-area
rectangle. Assessment covers source edge support, side consistency, output
stretch, normalized condition, Jacobian variation, corner perturbation
stability, source pixels per output pixel, and protected-region intersections.
Unsafe transforms fail closed and carry review reasons; they do not return
empty or distorted pixels. Polygon-masked and conservative-box automatic
fallbacks remain roadmap work because the repository has no labelled irregular-
print crop references with which to validate their release policy.

The untrimmed rectified crop remains the faithful `source_corrected` asset.

### 4.4 Border handling

**Root cause:** one global median frame colour plus sparse Canny evidence can
remove quiet skies, walls, snow, night scenes, or physical print borders. Exact
trim offsets and confidence are not persisted.

**Correction:** keep trimming opt-in. Classify four edges independently using
uniformity, colour distance, gradient transition, texture continuity, maximum
trim, protected content, and neighbour/caption vetoes. Save offsets, confidence,
reason, and the untrimmed variant. Ambiguous edges are preserved.

### 4.5 Orientation

**Root cause:** no orientation model is installed, yet an uncalibrated
detail-gravity heuristic can auto-rotate a correct landscape 180 degrees and
report a value greater than one as confidence.

**Correction:** heuristic evidence is advisory only. A registered calibrated
model may auto-rotate only above high confidence and margin gates. Medium and
low confidence preserve pixels and emit `ORIENTATION_UNCERTAIN`. Future OCR,
face, pose, and horizon cues remain separate evidence components, not one
unbounded score.

### 4.6 Degradation analysis and deterministic enhancement

**Root cause:** a few global statistics conflate photographic intent with
damage. Conservative and deep feature flags currently enter the same function.
Colour correction stretches channels independently; the critic lacks local
colour, texture, halo, face, and text safeguards.

**Correction:** a typed degradation profile records signal score, confidence,
evidence, and ambiguity for exposure, clipping, contrast, cast, saturation,
monochrome/sepia intent, luma/chroma noise, blur, blockiness, texture, faces,
and text. The planner selects `noop` when evidence is weak.

The deterministic route contains only conditionally selected operations:

1. luminance tone/exposure correction with clipping caps;
2. white balance only with reliable neutral evidence and colour-gain limits;
3. separate edge/texture-aware luminance and chroma denoise;
4. thresholded, scale-aware sharpening only for confirmed blur;
5. immediate per-step reversion and an independent final safety assessment.

It contains no inpainting, super-resolution, face synthesis, or generative
model imports.

### 4.7 Restoration candidates

Deep processing is a distinct optional route. Model artifacts must be locally
registered with SHA-256, license, colour/range contract, scale, tiling, device,
and version. The route is selected from degradation type and resolution; it is
never universal. Face restoration remains opt-in and never auto-releases.

Large-image inference must use overlap, context padding, and blended seams.
Model cache identity includes artifact hash, device, tile size, and parameters.
Failure returns the source/conservative asset and cannot block persistence.

### 4.8 Quality selection

The independent critic vetoes new per-channel clipping, excessive Lab/cast or
saturation change, worst-tile structural change, texture loss, noise growth,
halo overshoot, unexpected dimensions/aspect, tile seams, and protected
face/text mutation. Metrics describe risk; no one score authorizes release.

### 4.9 Confidence and review

Store separate components:

```text
detection_confidence
boundary_confidence
geometry_confidence
perspective_confidence
orientation_confidence
border_trim_confidence
enhancement_safety_confidence
overall_release_confidence
```

Stable uppercase reason codes include `PHOTO_COUNT_UNCERTAIN`,
`OVERLAPPING_CANDIDATES`, `BOUNDARY_LOW_CONTRAST`,
`POSSIBLE_CONTENT_CUTOFF`, `PAGE_BACKGROUND_INCLUDED`,
`QUADRILATERAL_UNSTABLE`, `PARTIAL_PHOTO`, `ORIENTATION_UNCERTAIN`,
`BORDER_TRIM_UNCERTAIN`, `ENHANCEMENT_ARTIFACT_RISK`,
`FACE_RESTORATION_RISK`, and `TEXT_DISTORTION_RISK`.

Review payloads show the source page and visible mask separately from the warp
quad, untrimmed/trimmed crops, all enhancement candidates, component confidence,
operations, gates, and reasons. A review event must validate its task, preserve
the original prediction, and update the current projection. `unsure` remains
in the queue.

### 4.10 Learning data and leakage

Export a manifest first, then derive YOLO labels. Each record contains page and
album/source-group IDs, image hash, original-resolution coordinate contract,
original prediction, corrected/rejected/added instances, orientation, selected
variant, reasons/tags, model/run version, and reviewer evidence. Duplicate
filenames never form identity.

Train/validation/test splitting is by album/source group. The splitter rejects
any group appearing in more than one split. The present one-group dataset
cannot produce an independent split and is explicitly reported as such.

## 5. Evaluation plan and gates

### Cropping

`app.evaluation.cropping` and `scripts/evaluate_onnx_segmentation.py` use one-to-
one assignment and report precision, recall, F1, source-resolution mask and
polygon IoU, boundary F-score, normalized corner error, missing/duplicate/false
positive rates, content cutoff, retained background, perspective stability,
automatic acceptance, review rate, and serious failures.

The initial regression target is:

- exactly two detections on `page_31`;
- replace rather than duplicate the malformed low-confidence model boundary;
- zero serious failures on the locked holdout;
- no regression in model-only page-perfect results;
- expensive classical analysis only on empty/risky model sets.

### Enhancement

`scripts/evaluate_enhancement.py` compares source, frozen July-2026 behavior,
and the new deterministic candidate. It writes reference/source/legacy/new/
difference images, JSON metrics, parameters, hashes, runtime, and an HTML
contact sheet.

Paired fixtures report PSNR, SSIM, Lab error, clipping, noise, sharpness, halo,
texture, and structure. Preservation cases require near-no-op behavior for
already-good, monochrome, and intentional sepia content. Real unpaired examples
will require blinded human comparison and may not be approved from a no-
reference score.

## 6. Test plan

- Geometry: 45-degree ordering, coordinate round trips, border vertices,
  concave/self-intersecting/degenerate quads, perturbation and stretch gates.
- Fusion: cross-source duplicates, true overlap, containment, touching prints,
  page containers, small prints, and locked `page_31`.
- Crop: fallback selection, output density, protected regions, untrimmed asset,
  and independent border decisions.
- Orientation: sideways model evidence, ambiguous cases, upright textured-ground
  landscapes, no-face/no-text images, and manual overrides.
- Enhancement: BGR/RGB, grayscale, alpha, uint16/float contracts; clean/warm/
  sepia/monochrome/high-key/low-key cases; clipping, colour gain, texture,
  halo, local mutation, tile seams, odd sizes, CPU failure, and provenance.
- Integration: upload through persistence, immutable originals, distinct
  variants, status/reason accumulation, idempotent replay, and review export.
- Frontend: masks with more than four vertices remain intact; only a validated
  server quad is edited.

## 7. Performance and reliability

Record wall time by stage and model, peak memory where the runtime supports it,
and candidate counts/rejections. The standard route is:

1. one ONNX pass;
2. classical evidence only for risk;
3. source-resolution refinement only around candidate bands;
4. deterministic enhancement only when a signal is confident;
5. restoration only by explicit policy.

Cache keys include input hash, pipeline/config version, model artifact hashes,
and plan parameters. A failed optional stage cannot mutate or delete the source-
corrected asset.

## 8. Delivery sequence

1. Lock baseline tests and reports.
2. Add independent metric primitives and the manifest format.
3. Fix shared geometry, candidate fusion, and orientation policy.
4. Add transform/fallback/border provenance and confidence components.
5. Separate and strengthen deterministic enhancement and the quality critic.
6. Make review events stateful, auditable, and training-ready.
7. Run the locked holdout and synthetic paired enhancement comparison.
8. Run unit/integration/regression tests and frontend build.
9. Review the complete diff and exclude private/generated artifacts.
10. Document measured results, limitations, launch verdict, and commit.

## 9. Claims explicitly out of scope for this repository evidence

- Consumer readiness across arbitrary albums, cameras, scanners, cultures, and
  print types cannot be established from one album context.
- Page curvature, severe overlap/occlusion, lens calibration, pose-aware
  orientation, and face/text identity need labelled independent data.
- No generative restoration can be described as historically faithful merely
  because it improves a perceptual score.
- The tracked personal golden fixtures require an explicit privacy/consent
  decision; no additional private album pages will be committed by this work.
