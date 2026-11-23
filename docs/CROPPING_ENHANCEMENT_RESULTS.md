# Cropping and Enhancement Results

Date: 18 July 2026

Baseline commit: `46e94be`
Implemented pipeline version: `2026-07-crop-enhance-v2`

## Executive summary

The dominant crop failure was not a lack of candidate evidence. PhoSca's ONNX
model found the right instance count on the difficult page, and the classical
detector found a much better boundary, but the old always-on concatenation/NMS
policy retained a malformed model crop and introduced extra detections. The
pipeline also mixed coordinate conventions, used a fragile corner-ordering
heuristic, and considered a homography valid after testing it against the same
four points used to fit it.

The new staged detector keeps the fast ONNX path for normal pages and invokes
classical boundary evidence only for an empty or risky model set. Polygon-aware
fusion can replace a weak model boundary without duplicating the print. On the
existing locked ten-page/29-photo holdout this changed 28 TP / 1 FP / 1 FN into
29 TP / 0 FP / 0 FN and raised polygon IoU from 0.9730 to 0.9813. The formerly
bad boundary on `page_31` improved from 0.6518 to 0.8754 IoU. That remaining
crop still retains 12.46% background and is correctly below the quality
acceptance gate.

Enhancement was redesigned as a Torch-free, condition-specific conservative
route. It preserves already-good, monochrome, and intentional-sepia fixtures
exactly, improves the paired noisy/soft fixture from 29.50 to 30.73 dB PSNR,
and makes bounded luminance-only changes for ambiguous faded or underexposed
inputs. Model restoration is a separate, registered, opt-in candidate and can
never auto-release. Because the repository has no real paired restoration
dataset, these enhancement results are regression evidence, not proof of
historical-photo quality.

## Current-system findings

| Severity | Confidence | Relevant code | Evidence and failure mode | User consequence | Implemented correction |
|---|---|---|---|---|---|
| Critical | High | `pipeline/segmentation.py`: proposal suppression and `segment_album_page` | Model-only produced 1 FP/1 FN; blindly adding classical candidates produced 5 FP/1 FN. On `page_31`, independent classical evidence was 0.876 IoU while the selected model boundary was 0.652. | Missing print content, duplicate crops, and page decoration/background crops. | Risk-triggered classical analysis plus mask/containment-aware cross-source clustering; replace a weak boundary only when independent evidence is better. |
| High | High | `pipeline/segmentation.py`: box NMS | Axis-aligned box overlap could suppress real overlapping prints and could fail to merge differently shaped proposals for the same print. | Legitimate overlaps disappear or one print is exported twice. | Mask-aware duplicate tests distinguish high mutual containment from real partial overlap. |
| Critical | High | `pipeline/perspective.py`: corner order and transform validation | Sum/difference ordering can assign one point twice near 45 degrees. Reprojecting the four fit points made the old semantic-validity check tautological. | Empty, mirrored, severely stretched, or incorrectly rectified output. | Shared cyclic ordering, uniqueness/convexity/angle/area checks, normalized conditioning, Jacobian stretch, and corner-perturbation stability gates. |
| High | High | `pipeline/image_ops.py`, `pipeline/geometry.py` | Boxes used half-open maximum edges while normalized polygon `1.0` could become an out-of-range pixel; alpha and high-bit-depth conversion were implicit. | Off-by-one crops, lost transparent content, inconsistent full-resolution mapping. | Explicit vertex/box coordinate contracts, reversible source-resolution mapping, EXIF/ICC metadata, white alpha compositing, and deterministic grayscale/uint16/float normalization. |
| High | High | `pipeline/perspective.py`, `pipeline/border_removal.py` | A global uniform-frame heuristic could mistake sky, snow, walls, night scenes, or studio backgrounds for a border. Only the trimmed pixels survived. | Meaningful edge content is silently deleted. | Independent edge classification requiring a physical transition, protected-region vetoes, maximum trim, exact offsets/confidence/reasons, and a permanently addressable untrimmed rectified asset. |
| High | High | `pipeline/orientation.py` | An uncalibrated face/detail heuristic could rotate an upright landscape by 180 degrees and emit confidence above one. | Correct photos are rotated incorrectly. | Only a high-confidence/margin-gated ONNX classifier may mutate pixels. Heuristics are bounded advisory evidence; ambiguous cases preserve orientation and emit `ORIENTATION_UNCERTAIN`. |
| High | High | `pipeline/enhancement.py`, `pipeline/dl_enhancement.py`, feature flags | Conservative and deep flags entered the same model-capable function; the supposed conservative route was not independently model-free. | Unexpected model use and unclear fidelity guarantees. | Separate deterministic conservative, model-restoration, and premium routes with fail-closed registration and variant labels. |
| High | High | `pipeline/degradation.py`, `pipeline/color_correction.py` | Global range/Lab statistics confused monochrome, sepia, high-key, and warm photographic intent with fading or damage. | Historical colour is modernized or monochrome content is tinted. | Typed signals with evidence/confidence/ambiguity; ambiguous colour disables chroma correction while strongly supported luminance damage may receive a bounded Lab-L-only adjustment. |
| High | High | `pipeline/quality_assessment.py`, `pipeline/quality_critic.py`, `pipeline/variants.py` | A local or aggressive mutation could pass weak aggregate checks; applied/reverted provenance and selected paths diverged. | Halos, texture loss, clipping, colour drift, or a stored path that does not match the selected variant. | Per-step and final reversion gates for clipping, chroma, local structure, texture, noise, halos, dimensions, and hashes; one canonical selected-path contract. |
| High | High | `routers/review.py`, `routers/photos.py` | Review events were previously append-only and corrections overwrote prediction context; `unsure` could disappear from the queue. | Feedback was not operational or auditably reusable. | Validated/idempotent state transitions, immutable original-prediction snapshots, before/after correction revisions, latest-event queue semantics, and explicit semantic enhancement selection. |
| High | High | `scripts/export_reviewed_yolo_labels.py` | Legacy export did not reliably require a current approval and filename identity could merge unrelated pages. | Noisy training labels and train/test leakage. | A canonical page-ID/hash-keyed reviewed JSONL manifest, strict approved-geometry YOLO derivation, and deterministic album/source-group leakage-safe splitting. |
| Medium | High | `frontend/src/App.jsx`, `frontend/src/MobileApp.jsx` | Dense instance masks were truncated to four points; mobile corner inference repeated the unstable extrema heuristic. | Review could destroy a good visible boundary while editing its warp quad. | Preserve the dense visible polygon and edit a separate cyclic/minimum-area four-corner warp quad. |
| High | High | `services/pipeline.py`: deduplication query | Perceptual duplicate search crossed tenant boundaries. | One tenant's photo could be linked to another tenant's record. | Candidate search is now tenant-scoped and has a cross-tenant regression test. |

## Cropping changes

### Detection, evidence, and fusion

- ONNX instance segmentation remains the primary CPU path and still requires a
  registered SHA-256 model artifact.
- Every proposal carries detector source, dense polygon, boundary evidence,
  geometry evidence, and fusion alternatives.
- Classical contour/contrast/line analysis runs only when the model is absent,
  empty, or reports a risky instance set. It ran on one of ten holdout pages.
- Polygon IoU, mutual containment, family, boundary support, and geometry are
  used to merge duplicates while retaining legitimate overlapping instances.
- Page-sized containers, implausible strips, nested proposals, and weak
  unmatched classical candidates remain independently rejectable.

### Boundaries, corners, and perspective safety

- Normalized polygon vertices map to inclusive source pixel centres; boxes keep
  their half-open API contract.
- Dense visible masks remain available for review/export. A supported convex
  quad is preferred for warping; otherwise the conservative minimum-area
  rectangle encloses the visible hull.
- Quad validation checks uniqueness, cyclic order, convexity, area, side length,
  corner angle, output dimensions, normalized condition, local singular-value
  stretch, scale variation, and response to small corner perturbations.
- Output size follows source edge lengths and records effective source pixels
  per output pixel. Unsafe transforms fail closed to review.
- The current implementation does not yet auto-release a polygon-masked or
  conservative-box fallback after a failed homography; that is safer than
  returning an unvalidated transform and is listed in the roadmap.

### Border and orientation handling

- Border refinement is enabled by default. It fits the four paper-to-picture
  transitions and intersects them to obtain the print corners. For automatic
  detections it can use neighboring page pixels to recover a missed corner.
  The final image is sampled directly from the original page. The crop metadata
  records the fitted corners in both rectified and original page coordinates.
- `source_corrected` is the crop used for display and enhancement, including
  any supported trim. The full rectified print is available for review as
  `source_corrected_untrimmed`.
- A calibrated orientation model can auto-rotate only above confidence and
  margin thresholds. Missing/ambiguous model evidence preserves pixels and
  records a suggested rotation and reason code.

### Confidence and review

PhoSca now persists `detection_confidence`, `boundary_confidence`,
`geometry_confidence`, `perspective_confidence`, `orientation_confidence`,
`border_trim_confidence`, `enhancement_safety_confidence`, and a conservative
aggregate. A high detector value cannot hide a weak boundary or transform.
Stable uppercase reasons are exposed in desktop/mobile review payloads together
with the source page, visible polygon, warp quad, untrimmed/trimmed crops,
orientation evidence, all variants, metrics, and gates.

## Enhancement changes

### Degradation analysis and deterministic planning

- Array inputs are normalized to a documented BGR `uint8` working contract;
  grayscale, alpha, uint16, and unit-float inputs are covered by tests.
- The degradation profile records evidence and confidence for exposure tails,
  clipping, contrast, saturation, monochrome/sepia intent, cast, luma/chroma
  noise, blur, blockiness, faces, and text where evidence is available.
- The planner selects only supported operations: flat-region/edge-aware denoise,
  neutral-evidence white balance, bounded luminance tone, and local-bound
  thresholded sharpening. Weak evidence selects an exact no-op.
- Warm/sepia ambiguity vetoes chroma correction. Strong independent luminance
  damage may still receive a small Lab-L-only change.

### Denoising, colour, sharpening, and safeguards

- Denoising blends only flat/weak-texture regions and must reduce measured noise
  without excessive texture or saturation loss.
- White balance requires reliable neutral-region evidence and bounded Lab
  movement; monochrome and intentional warm-tone paths preserve source colour.
- Sharpening is scale-aware, noise-thresholded, and clamped to local extrema plus
  a two-level overshoot allowance.
- Every operation is independently compared with its input. New clipping,
  chroma motion, weak local structure, texture loss, noise growth, halos, or an
  unexpected shape reverts the operation; a final critic can revert the whole
  candidate.

### Model routing and variants

- `source_corrected`, `conservative_enhanced`, and `restoration_candidate` are
  distinct canonical variants. The untrimmed/trimmed source derivatives remain
  addressable as review variants.
- Real-ESRGAN/GFPGAN-style processing is disabled by default and requires
  registered local artifacts. Registration or runtime failure falls back to the
  faithful source and cannot block persistence.
- Model/generative results are labelled restoration candidates and never
  auto-release. An explicit human comparison can select one and the semantic
  selection is stored for learning.
- Parameters, degradation profile, step decisions, gates, runtime, model chain,
  input/output pixel hashes, stage/run identity, and asset lineage are retained.

## Evaluation

### Cropping dataset

`data/evaluation/cropping_dataset.json` references the ten existing, manually
verified golden pages without duplicating them. There are 29 labelled prints.
Available tags cover camera capture, visible page border, multiple columns,
caption near photo, rotated photo, and partial photo. All pages belong to one
legacy source group. This is a same-domain regression set, not an independent
album test and not evidence for arbitrary consumer albums.

The evaluator uses one-to-one assignment at IoU 0.85 and rasterizes polygons at
original page resolution. The baseline reports were captured before the
implementation from commit `46e94be`; the final report was generated with:

```powershell
python scripts/evaluate_onnx_segmentation.py --include-classical-fallback --output runs/cropping_final.json
```

| Crop metric | Model-only baseline | Old always-fused baseline | New staged fusion |
|---|---:|---:|---:|
| Precision | 0.9655 | 0.8485 | **1.0000** |
| Recall | 0.9655 | 0.9655 | **1.0000** |
| F1 | 0.9655 | 0.9057 | **1.0000** |
| TP / FP / FN | 28 / 1 / 1 | 28 / 5 / 1 | **29 / 0 / 0** |
| Mean polygon/mask IoU | 0.9730 | 0.9668 | **0.9813** |
| Page-perfect count at IoU 0.85 | 9 / 10 | 6 / 10 | **10 / 10** |
| Duplicate rate | not recorded | not recorded | **0.0000** |
| Serious-failure rate | not recorded | not recorded | **0.0000** |

Additional final-only metrics are boundary F-score 0.9286, corner error 0.00217
of the page diagonal, mean content cutoff 0.636%, mean retained background
1.253%, and perspective stability 29/29. Baseline code did not calculate these,
so no before/after value is invented.

The raw quality gates accept 28/29 detections (96.55%); the second `page_31`
print emits `PHOTO_COUNT_UNCERTAIN` and `BOUNDARY_LOW_CONTRAST`. Production
automatic acceptance is **0%** and review is **100%** because the repository has
no segmentation calibration artifact and `REQUIRE_CONFIDENCE_CALIBRATION=true`
fails closed. This is intentional. Orientation accuracy is unmeasured because
the fixtures contain no verified crop-orientation labels.

CPU timing for `segment_album_page` (image I/O and evaluator scoring excluded)
was mean 1,473.0 ms, median 1,129.2 ms, and p95 3,433.7 ms. ONNX work averaged
408.9 ms. Classical analysis ran on one page (10%) and took 3,954.2 ms there.
The old reports did not measure equivalent end-to-end wall time, so a runtime
speedup/regression versus baseline cannot be calculated. Peak memory, CPU
utilization, queue time, and worker throughput were not instrumented.

### Enhancement dataset

There is no suitable paired real-photo reference set in the repository. The
new harness therefore uses seven generated paired fixtures: faded/yellow cast,
underexposed, noisy/soft, JPEG artifacts, already-good colour, intentional
monochrome, and intentional sepia. It compares the source degradation, a frozen
July-2026 legacy algorithm, and the production conservative route, and writes
reference/source/legacy/new/difference PNGs, full metrics/provenance JSON, and
an HTML contact sheet under ignored `runs/`:

```powershell
python scripts/evaluate_enhancement.py --output-dir runs/enhancement_final
```

| Enhancement metric (7 generated cases) | Frozen legacy | New conservative |
|---|---:|---:|
| Mean PSNR | 23.0915 dB | **57.6125 dB**¹ |
| Mean SSIM | **0.92291** | 0.92271 |
| Mean Lab error (lower is better) | 19.8905 | **11.6659** |
| Mean texture error (lower is better) | 0.07596 | **0.04309** |
| Mean halo score (lower is better) | 0.01383 | **0.00965** |
| Preservation PSNR: good/mono/sepia | 24.5568 dB | **100 dB (exact no-op)** |
| Preservation SSIM | 0.96926 | **1.00000** |

¹ The aggregate PSNR is intentionally dominated by three exact no-ops and
must not be interpreted as a 34 dB restoration gain.

On the noisy/soft case, the new route improved source PSNR 29.50 to 30.73 dB
(legacy 24.84), Lab error to 6.63 (legacy 14.97), and SSIM to 0.679. Legacy SSIM
was higher at 0.787 because it smoothed more aggressively; PSNR, texture, and
visual output are needed to interpret that tradeoff. The faded case improved
source PSNR 22.88 to 23.32 and Lab error to 14.71 with zero new clipping. The
underexposed case received a deliberately small adjustment: PSNR 12.92 to
13.18 and SSIM 0.8641 to 0.8668. Legacy was closer to the synthetic reference
in luminance, but its generic route is not accepted as evidence of safer real
historical restoration.

The conservative evaluator averaged 760 ms per 520x360 fixture (457-1,429 ms),
versus 164 ms for the frozen legacy path. The cost comes from degradation
analysis and independent local quality checks; production no-op cases still
run the analyzer/critic. Peak memory was not measured. No real face identity,
OCR consistency, skin-tone ground truth, dust/scratch reference, or hallucinated
detail labels exist, so those requested metrics remain unmeasured. Generative
restoration was not evaluated or enabled.

### Automated verification

The final validation covers unit, synthetic geometry, API/review, complete
pipeline, regression fixtures, exporter/split behavior, and frontend production
build: **123 backend tests passed**; Ruff lint and format checks passed for all
45 changed Python files; MyPy passed across 96 backend source and test files with known
third-party OpenCV/NumPy stub categories disabled; and the Vite production build
completed successfully. Evaluation artifacts remain ignored because they
include local paths and derivatives; no new album images are added by this
change.

## Remaining failures and limitations

- `page_31` is no longer a missed/false detection, but its difficult second
  boundary has boundary F-score 0.241 and retains 12.46% background. It is a
  review case, not a clean automatic crop.
- The holdout represents one source group. It does not cover independent
  albums, scanners, clean flatbed scans, dark/patterned pages, borderless or
  touching/strongly overlapping prints, glossy reflections, shadows, torn
  edges, hands, severe page curvature, or calibrated lens distortion.
- Page/table/background localization and page-level lens/perspective correction
  are not yet a separate production stage.
- Irregular/torn/partly occluded masks are retained for review/export, but an
  unsupported quad currently fails closed; a validated polygon-masked fallback
  is not yet auto-produced.
- OCR runs after segmentation. Protected-region support prevents known face/
  subject cuts, but robust page-text-versus-photographed-text classification is
  not yet available during candidate generation.
- Orientation has no installed classifier or labelled benchmark. No automatic
  rotation claim is made.
- Confidence components are structurally separated but not statistically
  calibrated. Until an independent calibration artifact is registered, normal
  production policy sends detections to review.
- The enhancement set is synthetic and small. It cannot establish authenticity,
  face consistency, or preference on real historical photos.
- Detection refinement is CPU-capable but the one difficult-page classical path
  is too slow for high-throughput local use without ROI/caching optimization.
- No peak-memory, worker-throughput, queue-delay, GPU-memory, or failure-rate
  benchmark across a sustained workload was captured in this repository run.

## Remaining roadmap

| Priority | Improvement | Expected benefit | Effort | Data required | Runtime impact | Risk / dependencies |
|---:|---|---|---|---|---|---|
| 1 | Build an independent multi-album crop/calibration benchmark and lock privacy/consent | Makes confidence, review rate, and consumer-readiness claims statistically meaningful; improves domain robustness | High | Hundreds of fully reviewed pages grouped by album, including negatives and all listed difficulties; orientation labels | Evaluation/training only; inference unchanged | Privacy, licensing, annotation consistency, album leakage controls |
| 2 | Add full page localization, calibrated lens correction, curvature cues, and polygon-masked/conservative-box crop fallbacks | Largest gain for camera captures, bent pages, torn prints, and unsupported quads | High | Page corners/camera intrinsics where available, irregular masks, expected final crops | Moderate to high on risky pages only | Must prove content preservation; unsafe fallback could hide errors |
| 3 | Replace full-page classical refinement with cached, model-prompted ROIs and line/edge snapping | Retains the `page_31` quality gain while reducing its ~4.0 s CPU cost | Medium | Existing failure pages plus more low-boundary-confidence cases | Large reduction on risky pages | ROI can miss distant evidence; cache keys need model/config hashes |
| 4 | Create a real, blinded enhancement benchmark with faithful references/preferences and face/OCR consistency labels | Enables safe auto-release calibration and degradation-specific routing | High | Licensed scans plus cleaner references where possible; expert A/B votes; identity/text annotations | Evaluation high; selected inference checks moderate | Reference registration, subjective preference bias, sensitive faces/text |
| 5 | Train/register a conservative orientation classifier and add OCR/horizon/pose evidence calibration | Reduces needless review without rotating correct historical images | Medium | Upright crops from independent albums, deliberate rotations, lying people, no-face/no-text negatives | Low to moderate CPU | Dataset bias; faces and horizons are not universal orientation truth |

The largest next model gain should come from fine-tuning the existing one-class
YOLO segmentation model on a diverse, album-grouped failure dataset, not from a
larger generic detector. SAM/SAM 2 can help reviewers refine irregular masks
offline but should not become a required runtime dependency. The largest
orientation gain requires a small registered four-way classifier trained on
PhoSca-specific hard negatives. Any denoise/deblur model should be introduced
only after a real paired benchmark and model-specific face/text identity gates;
generative face restoration should remain an explicitly labelled candidate.

## Final verdict

Cropping is ready for guarded use on albums similar to the locked holdout, with
the current fail-closed calibration and review policy. It is **not yet proven
ready for unrestricted normal consumer use** across arbitrary albums. The most
important known difficult boundary now preserves the complete print and is
routed for review, but still contains excessive page background.

`source_corrected` is ready as the normal faithful release. The deterministic
conservative enhancer has substantially better preservation behavior and is
ready for opt-in/shadow comparison, but **enhanced pixels are not yet ready for
unattended automatic release** on real historical collections. Restoration and
face-restoration candidates always require explicit human choice.

Review remains required for uncalibrated detection, low-contrast or partial
boundaries, possible photo-count errors, unsupported/unstable quadrilaterals,
ambiguous orientation, any applied low-confidence border trim, failed
enhancement gates, and every model/generative restoration candidate. The five
highest-priority improvements are the five roadmap items above: independent
data/calibration, page/irregular geometry, faster ROI refinement, a real
enhancement benchmark, and a calibrated orientation model.
