# Cropping and enhancement evaluation data

`cropping_dataset.json` is the manifest for the repository's existing locked
segmentation fixtures. It references, but does not duplicate, the tracked page
images and manually verified YOLO polygons.

Each page record supports:

- an original page image and polygon/mask annotations;
- expected instance count, ignore regions, page-level text regions, category,
  difficulty tags, expected orientation, and optional expected final crops;
- annotation provenance and a `source_group` used to prevent album leakage.

The current ten-page holdout is **not album-independent**. All records belong
to one legacy source group, so it measures regressions in the known capture
domain only. A consumer claim requires private or appropriately licensed pages
from many independent albums, scanners, cameras, layouts, and photographers.

Do not create labels by guessing boundaries. Set `annotation_provenance` to
`manually_verified` only after a person has checked every photo on the full
resolution page. Synthetic fixtures must use `generated_fixture` and must not
be reported as real-page accuracy.

Run the production crop benchmark and create an HTML contact sheet.

```powershell
python scripts/evaluate_onnx_segmentation.py --include-classical-fallback --visuals --output runs/crop-quality/golden.json
python scripts/evaluate_onnx_segmentation.py --include-classical-fallback --legacy-crop --output runs/crop-quality/rollback.json
```

To diagnose the fallback on a checkout without the registered detector, run
`--classical-only`. Use `--model` to point to an absent file if the registered
weights are present. The report has no model digest and cannot supply release
or detector calibration evidence.

The report assigns detections to visible instances once and scores the final
source-to-output transform of each crop. It reports the sampled footprint,
missing content, foreign samples, samples outside the capture, and corner
error in source pixels when annotated corners exist. Aspect and landmark
errors require paired reference geometry. For banded cutoff and contamination,
one source pixel around the annotation boundary accounts for annotation and
raster uncertainty. Reference crop polygons are sampled at the original pixel
centers using their subpixel coordinates. The evaluator replays each mode's
sampler against the source ownership mask. It counts every output sample with a foreign
contribution and requires zero such samples on the synthetic references.
The footprint uses the inverse-mapped output pixel centers, which avoids an
extra half-output-pixel margin on downsized crops. The gate also checks output
sampling along both axes and the correspondence between the stored quad and
the actual homography. Real release requires at most two source pixels of
error at every reference corner.
The `runtime.image_to_jpeg` duration includes decode, preparation, detection,
protected regions, crop rendering and JPEG writes. It excludes metric
computation, database access, OCR and later enhancement.

The per-page `instance_diagnostics` lists missed prints, weak masks,
combined prints, distinct fragments of one print, duplicates and unexplained
proposals. These are overlap-based leads for source review. They leave
one-to-one precision and recall unchanged. For each mode, the report counts
measured geometry failures after a valid detection and records passing
geometry withheld for review.

The seeded synthetic split isolates boundary fitting behind deliberately
imperfect proposals. Run without `--geometry-only` to measure detection as
well. Use `development` during tuning and `locked` when comparing frozen
candidates. The generator's exact camera intrinsics are unavailable to crop
rendering in both runs. The benchmark measures how the same geometry used on
ordinary uploads estimates the aspect ratio of a tilted print.

```powershell
python scripts/evaluate_onnx_segmentation.py --synthetic locked --geometry-only --require-final-quality --visuals --output runs/crop-quality/synthetic-locked.json
```

The development split also includes `dense_gutters`, with 30 closely mounted
prints, narrow exposed paper, shadows and a mix of bare and bordered prints.
All 60 crop references are recoverable and have exact source masks, corners,
aspect ratios and landmarks. With the registered detector and the v4 local
refiner, the development baseline has 15 of 30 matches at mask IoU 0.85.
Source geometry brings all 30 final instance footprints above 0.85, while the
strict crop check still sends both modes to review. Missing metric aspect
evidence and the local refiner's conflicting ownership claims account for
those reviews. See `gutter-split/dense-development-baseline.json` under
`runs/crop-quality/`.

A four-epoch warm fine-tune of the local refiner mixed 536 synthetic dense
patches with 480 existing training patches. Mean IoU on 166 separate dense
validation patches rose from 0.562 to 0.907, and their mean foreign fraction
fell from 41.8% to 4.6%. On `dense_gutters` the candidate resolved the pixel
ownership conflict and passed four of 30 cases in each crop mode. The native
Met development scan fell from 44 to 42 matches, and the frozen synthetic
detection check failed. The deployed configuration retains its current
refiner. The paired reports are `gutter-split/local-mask-dense-training.json`
and `gutter-split/dense-model-{development,locked,golden,met}.json` under
`runs/crop-quality/`.

For a real, independently annotated benchmark pass a separate
`--dataset-manifest`. Page paths may be relative to the manifest. A page can
declare `expected_count: 0`. Pin each page with `image_sha256` and each YOLO
label file with `annotations_sha256`. A reference mode that uses a mask PNG
also needs `mask_sha256`. Inline polygons are covered by the manifest hash.
Keep physical albums or archive volume identifiers in
`source_group` and verify train, calibration and test provenance before
recording `split_policy.album_independent: true`. Reserve the final manifest
with `split_policy.role` set to `test` or `locked`. The evaluator accepts that
split for inference only with `--require-final-quality`. The release check
requires at least two distinct physical album groups, unique pinned image
hashes and verified annotations for every page.

Each registered detector checkpoint and any active learned mask refiner need
a reviewed training provenance inventory. Add a `training_provenance` object
with the inventory's `path` relative to the model registry and the inventory
file's `sha256` to that checkpoint's registry entry. The inventory uses
`schema_version: 1`, the checkpoint's `model_sha256` and a `model_origin` that
identifies the weight source and inherited checkpoints. Its `review`
contains a `reviewer` identifier and `evidence` pointing to the documented
source trail. Each entry in `sources` records an `image_sha256`, the canonical
physical album ID as `source_group`, the `source` reference and a `split` of
`train`, `validation`, `selection` or `development`.

Include pages used to fit the weights, choose between candidates or tune the
deployed configuration. Document inherited weights and their known datasets
in the review evidence. When album provenance cannot be established, leave
the inventory unregistered and keep results in review. Use the same canonical
album IDs across inventories, calibration and final references. Before
inference, the evaluator rejects any image hash or album ID shared by the
reviewed training/development inventory, calibration split or final split.

`expected_final_crops` follows the order of the YOLO instances. Each item
contains a `full_print` and `image_only` record. Each mode has a normalized
`polygon` or a source-resolution mask PNG and a boolean `recoverable`.
When `full_print` supplies a PNG, instance matching and boundary scores
use that native visible mask, including any disconnected regions and holes.
The YOLO label supplies the instance index and annotation order.
A recoverable reference needs `warp_quad` with four verified normalized source
corners in upright output order (top left, top right, bottom right, bottom
left). Supply metric geometry as `shape` (height and width of an independently
rectified raster) or `physical_size_mm` (measured width and height in
millimetres for that mode). With an independent exact crop scan, provide
`reference_image` and `reference_image_sha256`. Its decoded shape can supply
the reference raster dimensions, and the contact sheet shows it beside the
generated crop. To use an independently captured scan that still includes
mounting or scanner margins, provide `reference_image_quad` in its decoded
pixel-centre coordinate system, normalized between 0 and 1. Order the corners
top left, top right, bottom right, bottom left and supply the independently
measured rectified `shape`. The preflight checks the original scan hash and
rectifies that scan for the review sheet. Where both types of dimensions exist,
their aspect ratios must agree within the 1% benchmark tolerance. Optional
`source_landmarks` and `landmarks` must be supplied together in source and
rectified pixel coordinates. Set `annotation_provenance` to
`manually_verified` on a mode after verifying its reference. The release gate
requires references for both modes on every print and at least 90% passing,
automatically eligible results per mode among recoverable prints. At least two
of the same physical album groups must have recoverable references in both
modes. The 90% threshold also applies to each group with recoverable examples
in the mode being scored. Report schema 32 binds
per-mode eligibility to native instance-mask scoring, the source-stock
occlusion check, album-window detection, dense-page tiled and deskewed inference,
the classical line-pair fallback, separated contrast recovery, shared print
stock grouping, empty drawn frame rejection, print edge recovery, page-plane
estimation, curved motif
checks, the configured source or learned mask refiner and detector-bound
confidence calibration. The report pins the reviewed training inventories and
records the final page hashes and album IDs. The runtime checks disjointness
against the registered inventories and fitted calibration and verifies the
overall and per-album counts for both modes.
A verified motif can qualify separately when its print corners require review.
Detector calibration and the other release gates still apply.

Validate the real reference files before running the detector on an independent
split. `--check-references` checks hashes, geometry and review coverage without
inference. It can prepare a draft before the model or calibration is available.
With `--require-final-quality` it returns a nonzero exit code while the reference
split is still incomplete. The report lists dataset, page and
mode issues, including missing measurements and unverified annotations.
Instance indices start at zero and follow the page's YOLO labels. The report
also lists the groups with verified recoverable examples in both modes.
Missing source files are listed at their expected paths. Pages and modes
with the required files remain available in the review sheet.

```powershell
python scripts/evaluate_onnx_segmentation.py --dataset-manifest path/to/locked-album-manifest.json --check-references --require-final-quality
```

Add `--visuals` to reference preflight for a source-only review sheet. It
links to each original page and shows the indexed instance labels, the draft
mask and warp quad for each mode, native corner tiles and any independent
reference image. When the reference image and quad are present, the sheet
also shows how the draft quad rectifies the page image for direct comparison.
Open `index.html` in the report's sibling `-visuals` directory. The SHA-256
at the top identifies the exact manifest under review. These visuals use
the annotations and source files only and leave the release checks unchanged.

An independent reviewer can confirm the page's photo count and instance
assignments, then assess **Ganzer Abzug** and **Nur Bildmotiv** separately.
Choose **Korrektur nötig** for a reference that needs changes. A mode with
missing measurements, corners or pinned files cannot be confirmed. Enter
the reviewer identifier and save the decisions as JSON. Partial reviews
can be applied to a new manifest and reviewed further in a fresh sheet.

```powershell
python scripts/evaluate_onnx_segmentation.py --dataset-manifest draft.json --check-references --apply-reference-review phosca-referenzpruefung.json --reviewed-manifest reviewed.json --visuals --output runs/crop-quality/reviewed-reference-check.json
```

The import checks the draft's SHA-256, reruns the source checks, and records
the reviewer, timestamp and receipt hash with each decision in the new
manifest. Album independence and real release quality still require their
existing evidence.

Export calibration scores from a separate manifest with `split_policy.role`
set to `calibration` and `split_policy.album_independent` set to `true`.
Use verified page annotations, page and label SHA-256 hashes and physical
album `source_group` identifiers. Reserve the final release test albums
before exporting scores. The export requires the registered training inventories
and checks calibration albums and images against them. Each eligible prediction gets its raw score and
a correctness label from one-to-one matching at IoU 0.85 with the reviewed
instance boundaries. Each fitted detector source needs at least 20 predictions
with both correct and incorrect examples. With calibration required, sources
without enough examples stay in review.

```powershell
python scripts/evaluate_onnx_segmentation.py --dataset-manifest path/to/calibration-albums.json --include-classical-fallback --calibration-output runs/crop-quality/calibration-scores.json --output runs/crop-quality/calibration-diagnostics.json
python scripts/fit_confidence_calibration.py runs/crop-quality/calibration-scores.json --holdout-id calibration-albums-v1 --output models/segmentation-calibration.json
```

Keep `REQUIRE_CONFIDENCE_CALIBRATION` enabled for automatic release. Use the
deployed detector, registry, optional refiner, OpenCV version and
detector settings for the fit and the release report. Calibration schema 2
binds the fit to that configuration. The release evaluator checks both album
identifiers and page hashes for overlap with the calibration sources.
If orientation is enabled, the configured orientation classifier must provide
a decision for an automatically eligible crop. Include
`--include-classical-fallback` for the default production configuration.

The orientation development check used 63 upright archival photos for
training and 35 photos from different photographers for evaluation. Four
rotations per evaluation original produced 140 cases. The repository CNN
got 34 cases correct. An EfficientNetV2 ONNX candidate using PhoSca's runtime
interface got 140 clean cases correct. Under simulated low resolution,
shading, glare, contrast loss and noise it got 92 correct and applied 27
wrong rotations, including six to already upright photos. A 0.90 confidence
threshold still applied 10 wrong rotations. The reports are under
`runs/crop-quality/orientation-archival-development/`. The candidates remain
development artifacts. [Pretrained model](https://github.com/duartebarbosadev/deep-image-orientation-detection/releases/tag/v2)

With `onnx` and `onnxruntime` installed, prepare the upstream
`orientation_model_v2_0.9882.onnx` for the PhoSca evaluator. The preparation
checks the upstream SHA-256, adds the upstream center crop and normalization to
the graph, and checks the converted logits against the original graph. The
runtime supplies RGB resized with OpenCV's area interpolation.

```powershell
python scripts/prepare_orientation_candidate.py --source path/to/orientation_model_v2_0.9882.onnx --output runs/crop-quality/orientation-candidate/phosca-v2.onnx
```

`orientation-archival-holdout.json` pins 61 oriented crops from 47 Rijksmuseum
scans across six album groups. It records source URLs, archive and crop
hashes, crop rectangles and visual evidence for each upright label. The set
includes faded, shadowed, low-contrast and mounted photos. To restore the
samples, retrieve each `source_image_url` to its `source_image` path, verify
the digest, apply Pillow's `ImageOps.exif_transpose`, convert to RGB, and
save `source_rect_pixels_exif_canonical` as a lossless PNG at `image`.

The orientation evaluator uses the production ONNX path. It checks image
hashes and reported pixel rotations, scores each input as acquired and
after all four right-angle turns, and reports correct outputs, accepted
errors, and cases preserved for review. Pass a training manifest with the
same sample fields to check for overlapping image hashes and source groups.

```powershell
python scripts/evaluate_photo_orientation.py --manifest data/evaluation/orientation-archival-holdout.json --model path/to/orientation.onnx --output runs/crop-quality/orientation-holdout.json
```

On 244 rotations from this frozen set, the converted upstream EfficientNetV2
scored 242 correct, with one wrong rotation and one review. All 61
as-acquired photos stayed upright. The eight- and eighteen-epoch fine-tunes
from 63 development originals scored 237 and 238 correct, each with four
wrong rotations. The eight-epoch model increased correct outputs from 91
to 96 of 140 on the fixed synthetic stress diagnostic. The real-scan pilot
and synthetic stress results support keeping these models under evaluation.

The local intake under `runs/crop-quality/calibration-intake-sources/`
contains eight views with 66 draft print outlines from four source groups,
including the [Allston album](https://nmaahc.si.edu/object/nmaahc_2009.31)
and the [Baxter leaf](https://nmaahc.si.edu/object/nmaahc_2021.13.13a-f).
The duplicate screen compares those views with the available training,
Golden and reserved pages. Both detector profiles produced 66 YOLO and five
classical edge-contour predictions. A single crowded page contributes 49 of
the 71 predictions. The draft remains pending independent review before fit.

`calibration-archive-expansion/dataset-draft.json` adds 18 views and 37
draft outlines from the Getty album, Kessler family album and the Met
Disdéri proof album. Two cover views contain no prints. The manifest pins
the archive images and label files. Retrieve each `image_url` to the
matching `image` path to render the reference review locally.

```powershell
python scripts/evaluate_onnx_segmentation.py --dataset-manifest data/evaluation/calibration-archive-expansion/dataset-draft.json --check-references --visuals --output runs/crop-quality/calibration-archive-expansion-preflight.json
```

Across the prior and added views, the Registered detector with OpenCV 4.10
produced 144 predictions. The provisional matching rule uses IoU 0.85 and
excludes serious failures. Correctness still depends on independent review
of the source outlines and counts.

| Detector source | Provisional matches | Provisional nonmatches |
| --- | ---: | ---: |
| `yolo_onnx` | 39 | 47 |
| `edge_contour` | 5 | 28 |
| `yolo_onnx_tiled` | 23 | 1 |
| `contrast_region` | 0 | 1 |

The 33 edge-contour predictions span five albums. All 24 tiled YOLO
predictions in that census come from the same dense page. The only contrast prediction
does not support a fit. The draft digest, profiles, and per-album counts
are in `runs/crop-quality/calibration-intake-expansion/registered-expansion-source-census.json`.

`ibcc-denton-calibration-draft.json` adds a second physical album for tiled
YOLO calibration. The spread shows 16 separately mounted prints. The fixed
v22 Registered diagnostic returned 20 base proposals with a median area of
1.47% of the page. Tiling contributed 13 of the final 16 proposals in both
the v22 and v23 censuses. The verified hashes pin the scan and draft outlines
for independent review.
The archive provides the source under CC BY-NC 4.0. The scan stays local.
[Denton archive item](https://ibccdigitalarchive.lincoln.ac.uk/collections/document/26027)

```powershell
python scripts/evaluate_onnx_segmentation.py --dataset-manifest data/evaluation/ibcc-denton-calibration-draft.json --check-references --visuals --output runs/crop-quality/dense-real-source-intake/ibcc-denton-tracked-preflight.json
```

```powershell
python scripts/evaluate_onnx_segmentation.py --dataset-manifest path/to/locked-album-manifest.json --include-classical-fallback --require-final-quality --visuals --output runs/crop-quality/real-release.json
```

Configure `CROP_VALIDATION_REPORT_PATH` with the passing report. A report for
a different checkpoint, settings or synthetic split cannot enable automatic
downloads. Manual confirmation can make a reviewed crop available. Removing a
report also revokes access to the automatic downloads that used it.

Generated reports and visual comparisons belong under `runs/`, which is
ignored because reports can contain local paths and private derivatives.

The Library of Congress catalog links stand-alone scans of mounted prints to
the source album pages in the Prokudin-Gorskii collection. The archive
documents separate acquisitions of 430 mounted pages from 14 volumes and
714 individual print scans at 3,000 pixels on the longer side. The earlier
`runs/crop-quality/independent-intake-prokudin/intake.json` and its three
staged print scans are absent in the current workspace. Catalogued
candidates with exact leaf links include LOT 10333 no. 128
(`LC-DIG-prok-02023`, leaf `proka.0004a.22`) and LOT 10334 no. 54
(`LC-DIG-prok-02032`, leaf `proka.0005a.08`). Native raster requests
returned HTTP 403 from the network proxy. The catalog records nominal dimensions and the
rasters may include material beyond the print. Both boundaries and usage
rights still need verification before inclusion in the final split.
[Digitization notes](https://www.loc.gov/pictures/collection/prok/digitizing.html) ·
[LOT 10333 no. 128](https://www.loc.gov/pictures/item/2018679139/) ·
[LOT 10334 no. 54](https://www.loc.gov/pictures/collection/prok/item/2018679194/)

The source-only preflight at `eccc049` rechecked seven tracked drafts.
On this workspace it reported 26 missing pinned page rasters and 12
missing independent scan assets. The drafts account for 15 final-split
print candidates and 53 calibration candidates, with zero verified crops
per mode and zero shared verified physical groups. Renewed requests for
the LoC print and mounted-page paths and the pinned Getty image returned
HTTP 403 from the workspace proxy. The source labels and existing draft
measurements remain available for inspection.

The calibration export also requires at least two verified physical
album groups. Those groups must be disjoint from training and the
release groups. The three reserved calibration sources in the tracked
draft remain candidates. Each deployed detector source needs at least
20 scored predictions and both positive and negative cases for the
matching configuration. The exploratory transfer ONNX and PT assets
and all seven legacy Git originals match their inventory hashes. The
inherited COCO checkpoint also matches its hash. Its album-level
disjointness and the additional samples behind the historical registered
ONNX remain unverified. The preflight performs no inference on the
release drafts.

All 17 archived legacy JPEGs retain the same capture signature. The
relative `DateTimeOriginal` values follow the numbered page sequence
across the original train, validation and Golden sets within 242 seconds.
Treat these rows as one conservative source group. The EXIF supports
a shared capture session. The exact volume and 23 unarchived originals
still lack source evidence. A local census of 9,544 unique image and
model files found none of the 39 missing pinned file hashes.

## Precise crop validation

The frozen end-to-end test passes the rendered pixel, corner and aspect
gates for all eleven recoverable full prints and all twelve recoverable
motifs. All seven obstructed full prints and six obstructed motifs stay in
review. The final visible segmentation matches all eighteen instances at
IoU 0.85, with mean IoU 0.9953. The raw detector
matches six instances at the same threshold. Both metrics are reported
under `validation_gates` and in the raw and final instance metric sections
of `runs/crop-quality/observed-stock-locked.json`. On the tilted fixture the
relative aspect error is below 0.3% for both modes without camera intrinsics.
Across the 23 recoverable outputs, the largest corner error is 0.76 source
pixels. The renderer samples no foreign source pixels and loses no content
beyond the one-pixel boundary tolerance.
Two additional single-axis checks measured 4.25% and 2.75% aspect error in the
unmeasured previews. Both now require review. Supplying the measured print
ratio brought all four mode outputs within 0.31%. See
`runs/crop-quality/mild-single-axis-visuals/metrics.json`.

The precise renderer keeps the fitted quad, raster dimensions and interior
samples. At the perimeter it renormalizes the bilinear weights over source
pixels inside that mode's quad. A subpixel corner with no owned bilinear tap
uses the nearest owned source pixel within two pixels. Insufficient source
support routes that mode to review. In a controlled color fixture, foreign
contributions fell from 0.61% to zero for the full print and from 0.71% to zero
for the motif. On a 12 MP input the paired isolated warp medians were 15.4 ms
for OpenCV and 29.1 ms with the boundary pass. The images and timing samples
are under `runs/crop-quality/owned-sampling-visuals/`.

The stored raw proposal supports reproducible reprocessing. The final
visible outline uses verified physical edges for complete prints and
observed paper boundaries for appropriate review cases. A curved or
occluded full print stays in review. A motif can be rendered separately
when the corner holder crosses the print outline, its material is distinct
from the adjoining backgrounds, and four supported motif edges enclose an
uninterrupted paper rim. Evidence of a neighboring print causing the
occlusion keeps the proposed crop in review. On the paper-corner fixture the
motif has zero foreign samples and a largest corner error of 0.038 source
pixels. Automatic downloads still require the independent validation report.
A separate detected print fully inside another print's outline routes the
enclosing crop to review. An explicit wrong-object review removes that
detection from neighboring occlusion checks until its geometry is edited.

The confidence fitter pools tied raw scores before isotonic regression and
keeps the endpoints of merged blocks. The loader rejects ambiguous duplicate
knots. This prevents an interpolator from reading the last of several
contradictory labels at the same score as the calibrated confidence.

Two YOLO11n fine-tunes on disjoint generated seeds and existing training
labels improved development detection. Both regressed on the ten golden
pages. The registered model therefore remains in production. Training
outputs and the SAM 2.1 prompt comparison are in
`runs/crop-quality/training/` and `runs/crop-quality/sam2-comparison/`.
A third run used pale sections in varied generated motifs, textures from
existing training prints, and empty album pages. It reached 16 of 17
development matches at IoU 0.85 and produced four false positives. On the
golden pages it missed one of 29 prints and produced ten false positives.
See `runs/crop-quality/training/pale-motifs-development.json` and
`pale-motifs-golden.json`.

Another run trained box and mask refinement on matched plain and pale views
of the same generated print, keeping the shared features and class branch
frozen. It improved development matches from 7 to 15 of 17 at IoU 0.85.
The difficult pale motif still fell short. On the golden pages it matched 26
of 29 and generated eight false positives, so the candidate remains a training
artifact. The training, visual comparisons and golden report are under
`runs/crop-quality/training/edge-pairs-*`.

A proposal-conditioned mask model was trained on labelled patches from the
development dataset and tested with the registered detector's instances.
The second run added seeds that follow only the pigmented part of a print.
It reached 18 of 18 IoU matches on both synthetic splits. On the frozen
white-motif fixture the mask IoU rose from 0.5579 to 0.9843. Replacing all
masks lowered the golden mean IoU from 0.9786 to 0.9545 and left one serious
white-on-white failure in the frozen split. Selecting replacements against
the surrounding native paper edge reached 16 of 18 frozen matches and
preserved all 29 golden masks. Both experimental paths miss release criteria.
The training provenance, metrics and comparison image are under
`runs/crop-quality/training/local-mask-v2-*`.

A third run used separately generated pairs of pale album pages and prints
with opposite contrast at the paper edge. On the frozen split the white-on-white
mask improved from 0.8287 to 0.9877 IoU. The local masks matched all 18 prints,
with boundary F-score 0.8264 and zero serious detector failures. Feeding the
local masks directly into the renderer yielded 8 of 11 recoverable full prints
and 8 of 12 recoverable motifs. Using the original geometry with the learned
visibility restored the corner-holder motif but still failed cases with a
neighboring print. The registered path passed all 11 and 12 rendered cases.
In development, assigning overlapping pixels to the nearest original mask
improved the IoU on touching and overlapping prints. The renderer still sent
two otherwise recoverable prints to review.
The offline training and per-page render evidence are under
`runs/crop-quality/training/local-mask-v3-*`.

A development-only inference sweep compared full-page scales, flips and a
second pass on local proposal windows. Local views reached up to 9 of 17
matches at IoU 0.85. The best fixed full-page view by match count reached
10 of 17, with mean IoU 0.8464 and 15.0% mean content cutoff. The registered
inference path is unchanged. See `runs/crop-quality/probe-local-reinference.json`.
An ensemble probe kept the registered instances and compared one-to-one mask
alternatives from the existing fine-tunes against independent paper-edge
observations. It raised development matches from 7 to 16 at IoU 0.85, with
mean mask IoU 0.9534. Even the best candidate for the pale-motif failure reached
only IoU 0.653, and that case retains substantial missing print area. See
`runs/crop-quality/probe-mask-ensemble.json`.

The native decoder preserves small inward sections of a mask when the
fine contour stays inside the simplified silhouette within one source
pixel. A global switch to fine contours increased background retention on
the golden pages. The selective change preserves all 29 golden matches
and the existing mean IoU. See `runs/crop-quality/concavity-golden.json`.
An alternating two-pass run on the same ten pages measured p95 at 2.49 s
for precise crops and 2.66 s for the rollback path. Its paired samples are
in `runs/crop-quality/paired-concavity-runtime.json`.

The experimental local mask uses 320-pixel patches and adds registered
detector proposals from the training split. With exclusive pixel ownership
between neighbors it matches 18 of 18 visible instances at IoU 0.85 in the
frozen synthetic split, with mean IoU 0.9750 and boundary F-score 0.8767.
Keeping the registered geometry proposals yields 11 of 11 passing whole
prints and 12 of 12 passing motifs. The same-album golden comparison keeps
all 29 matches. Mean IoU is 0.9788 and boundary F-score is 0.9185, compared
with 0.9786 and 0.8926 for the registered model. On page 26, IoU drops
from 0.9885 to 0.9698 as the candidate includes more album stock.

The ONNX export produces identical thresholded masks to PyTorch for all
1,843,200 pixels in 18 development patches. On the test machine, CPU
inference alone takes 28.2 ms median and 32.3 ms p95 per patch with two
inference threads. The measurements are in the `local-mask-v4-*.json`
reports under `runs/crop-quality/training/`.

Set `VISIBLE_MASK_MODEL_PATH` to use a local ONNX refiner with precise
crops. Register its filename, SHA-256 and license in the same model
manifest as the primary detector. The refiner sees the original source
pixels, preserves the primary geometry proposal and supplies a separate
visible polygon. Nearby proposals compete for disputed pixels using
distance to their primary masks. The exported contour is rasterized again
to check that it leaves disputed pixels excluded. If the ownership decision
cannot be represented by one polygon, the worker keeps the original
proposals for review. Evaluation records that review outcome separately
from model execution errors. The release evidence binds the
resolved checkpoint, its registry and the checkpoint actually run on
the page.

Evaluate the integrated path with the same configuration as the worker.

```powershell
python scripts/evaluate_onnx_segmentation.py --synthetic locked --include-classical-fallback --visible-mask-model runs/crop-quality/training/local-mask-v4.onnx --manifest runs/crop-quality/local-mask-v4-manifest.json --require-final-quality --visuals --output runs/crop-quality/integrated-source-rim-v4-locked.json
```

The report preserves the primary detector results under
`primary_detector_metrics`. The detection gate measures the masks handed
to cropping, before fitting physical edges. The integrated frozen test
passes all 18 detections and all 23 recoverable crops. Source-based rim
recovery brings final segmentation back to mean IoU 0.9953. The Golden
comparison keeps all 29 detections. Independent album references remain
required for automatic release.

The optional refiner also checks for a sleeve reflection when its mask
extends past the primary mask. It removes the added band only when the
bright ridge continues outside the prints and the strip matches exposed
album backing at the same distance from the ridge. The tests cover glare
confined to a print and pale print stock directly beneath a fold. On
Golden page 26 the mask IoU rises from 0.9698 to 0.9832. The frozen
synthetic detection and both crop-mode gates still pass. The reports
are `sleeve-reflections-golden.json` and `sleeve-reflections-locked.json`
under `runs/crop-quality/`. With both masks already available, the
reflection check adds a median of 3.6 ms per Golden page across 50 runs.

In 20 paired Golden runs, median added time from image decode to JPEG was
115 ms per page. The p95 for that path was 2.39 s with the primary detector
and 2.92 s with the refiner. The paired samples are in
`runs/crop-quality/integrated-v4-paired-golden-runtime.json`.

The Smithsonian and Met album examples are diagnostic pilots with draft
annotations. An independent release benchmark still needs verified outer and
inner references from multiple physical albums and capture devices.

`runs/crop-quality/independent-intake/reserved-intake.json` pins 18 native
archive pages across the Whiting and Howland source groups for later review.
It excludes the three Whiting pages already run as diagnostics. The paired
archive crops register to the photographed texture, and their frame bounds
also include album stock or captions. `reserved-reference-drafts.json` has
provisional annotations for six reserved pages, including two negatives and
one obstructed print. The adjacent `reference-audit/` overlays compare both
boundaries at native resolution. `reserved-reference-drafts-pinned.json`
also pins the six draft label files. The preflight report at
`runs/crop-quality/reserved-reference-issues-pinned.json` lists 18 unverified
crop-mode annotations for nine prints and nine missing independent aspect
references. The reserved pages remain unused for model evaluation or tuning.

`runs/crop-quality/independent-intake-stringer/` adds three native album pages
from the State Library Victoria Stringer family collection. The intake has
15 visible prints and no exact or near whole-page duplicates among the
1,989 local comparison files. Ten prints on two pages have draft mode
references and native-resolution overlays. `stringer-pinned-native-review.json`
and `stringer-pinned-derivative-review.json` pin the image and label files.
The derivative manifest records the upload-compatible images and their
pixel-center mappings back to the archived pages. The native files remain
preserved. These archival scans provide a
second collection with complete paper edges for review. The references and
physical aspect ratios still need independent verification. The library gives
dimensions for the bound album and describes prints of various sizes. The
JPEG density supplies no metric rectification for individual prints. [Library record](https://find.slv.vic.gov.au/primaws/rest/pub/sourceRecord?docId=alma9933203733607636&vid=61SLV_INST:SLV)

The combined draft manifest at
`runs/crop-quality/combined-archive-reference-drafts.json` brings together
eight native pages with 19 prints. Its reference preflight confirms the file
hashes and lists 38 unverified mode annotations and 13 missing independent
aspect references. These drafts are reserved for reference review.

The Nelson album intake adds page GEM 1978.1292.0072 and four separately
photographed prints from the George Eastman Museum conservation study.
The publication-embedded JPEGs are pinned in
`runs/crop-quality/independent-intake-nelson/publication-assets.json`.
The catalog and image files preserve the distinction between the page
and separate close-up exposures. [Stephanie Becker's 2017 University of Rochester study](https://urresearch.rochester.edu/fileDownloadForInstitutionalItem.action?itemFileId=184123&itemId=33603)

For print 0072d, fitting the page and independent close-up separately
puts the maximum source-corner disagreement at 0.85 px for the complete
print and 0.78 px for the motif. The measured overall aspect agrees
with the catalog's 3 11/16 by 2 11/16 inch dimensions within 0.40%.
The draft has rectified reference images and landmark pairs held out of
the registration fit used to check match accuracy. Print 0072a has corroborating
outer corners, with up to 1.13 px of disagreement. Its observed aspect
differs from the catalog aspect by 1.51%, so its metric reference is
pending. A cross-check using print 0072d to measure the same page plane
agrees with the observed aspect of 0072a, but still conflicts with the
catalogued size. The faint top motif edge also remains unresolved.

`runs/crop-quality/archive-reference-drafts-with-nelson.json` now covers
nine pages with 23 prints. Its preflight pins the input and reference
files and reports the open review and measurement items. The Nelson
page remains reserved from detector evaluation while those reviews
are pending.

The follow-up at `runs/crop-quality/archive-reference-drafts-with-occlusions.json`
also includes visible motif masks for prints 0072b and 0072c, transferred
from the separately photographed details. In both drafts the gold corner
mounts cover printed content. Approximate covered areas are 0.64% and
0.66%, respectively. The current preflight report is
`runs/crop-quality/archive-reference-review-corner-mounts.json`.

A synthetic regression reproduces a light holder with a colored outline
on matching print stock. The corner check requires the same colored
component outside both adjoining sides and inside the print. A holder
that reaches the motif sends both modes to review. A holder confined to
the paper leaves the independently recoverable motif available. The
locked synthetic report is `runs/crop-quality/outlined-mount-locked.json`.

`runs/crop-quality/archive-reference-review-plus-nelson-figures.json`
extends the source-only intake to 11 pages and 30 prints. Figures 7–10
in the Nelson publication add two pages. Separate photographs show
mounts obstructing about 1.19% of the beach print's motif and a front
print with a paper caption obstructing about 85.53% of the rear church
print's motif. The three visible islands of the rear print are in the
native PNG sidecars. The draft YOLO label marks its largest visible
component, which covers 49.84% of the visible print area. The evaluator
reads all three from the PNG. All new mode annotations remain unverified.
The source and separate exposures can be compared in the preflight sheet.

The album-window regression uses pale leaves with a common dark gutter in
front-facing and perspective captures. It distinguishes thin decorations,
rectangular and arched apertures, and prints with their own broad paper rims.
The trace follows visible material. An aperture retains its curved outline
and both crop modes go to review with `ALBUM_WINDOW_OCCLUDED`, including
after manual coordinate edits. A separate pale border remains visible in the
proposal and goes to review with `ALBUM_WINDOW_STOCK_UNCERTAIN` because
an insert can present the same outline. Suppression of spurious plate and
spread hypotheses preserves small independent photo detections.

The integrated comparison with the registered detector and optional refiner
moves from 4 to 12 of 12 instance matches and from 16 to zero false positives
across eight synthetic scenes. The report and overlay are
`runs/crop-quality/album-window-integrated.json` and
`runs/crop-quality/album-window-integrated.jpg`. The unchanged locked
synthetic and same-album Golden splits are checked separately.

When the normal motif search falls short at a broad blank margin, PhoSca
checks that the same pale stock surrounds the motif on all four sides and
fits the four seams again in the original page pixels. A ruled margin,
another print or protected content can veto the suggestion. A verified
`suggested_warp_quad` stays in review. Selecting it in the desktop corner
editor and saving the motif corners is the existing manual confirmation.
The full-print crop stays available under its own approval rules.

The independent synthetic probe in `runs/crop-quality/wide-rim-agent/`
includes four recoverable wide-rim captures and a look-alike where the
same white area belongs to the photograph. Both require explicit motif
confirmation. On the four recoverable examples the maximum confirmed
corner error is 0.231 px with zero foreign source samples and zero cutoff
beyond one source pixel. Double frames, engravings, open sky and a
beveled insert produce no suggestion in the eight corresponding controls.

When the camera geometry alone leaves the aspect uncertain, the correction
can map a nearby print into the measured plane of a reviewed print on the
same capture. It verifies the reference edges at source resolution, checks
the target rectangle by reprojection, and checks sensitivity to half-pixel
corner changes. Conflicting applicable measurements keep the original
uncertain aspect, including when a peer has too little perspective to
estimate the plane on its own. The workflow keeps the rectified full
preview and the applicable geometry of both modes in review until the
target is measured.

The independent test in `runs/crop-quality/plane-reference-agent/` covers
2200 px and 4400 px views with one-axis tilt. A 0.45 px perturbation of one
reference corner changes the estimated target aspect by 0.044%. A hinged
plane, small reference and contradictory measured dimensions all leave the
aspect unresolved. A wrongly measured aligned reference can still produce
a plausible rectangle, so the correction records which measurement was used
and keeps both downloads in review.

A capture with registered camera intrinsics can also use a plane inferred
from multiple distinct, source-verified print boundaries spread over the
page. The fit checks the agreement between the source quads and the plane
and then checks each target again. The correction uses the target's own
corners. Automatic camera-based previews require review.
[Metric rectification of planes](https://www.robots.ox.ac.uk/~vgg/publications/1998/Liebowitz98/)

On the calibrated `dense_gutters` development capture, the registered
detector and crop renderer confirmed enough source evidence for 28 of the
30 prints. For those 28, the largest rendered aspect error fell from 4.27%
to 0.81% for **Ganzer Abzug** and from 4.80% to 0.94% for **Nur Bildmotiv**.
The full crop checks rose from 20 to 26 and from 21 to 27, respectively.
Two further prints needed boundary review. See
`gutter-split/calibrated-page-plane-comparison.json` and
`gutter-split/calibrated-page-plane-visual.jpg` under `runs/crop-quality/`.

Without camera intrinsics, several source-verified prints mounted at
different angles can constrain the shared plane's metric. Two disjoint
groups must support the fit and agree on the target aspect within 1%.
The preview uses the target's own source corners and requires review.
Aligned prints supply insufficient right-angle evidence for this estimate.
[Metric rectification of planes](https://www.robots.ox.ac.uk/~vgg/publications/1998/Liebowitz98/)

For aligned prints, v24 can estimate a shared horizon from at least six
source-verified print quads and set the metric from a measured print on the
same capture. Two disjoint groups must support the horizon. Each fit must
also agree with the applicable reference measurements, and the target's own
corners still have to fit the plane. Both previews require review.

On a synthetic 2050 × 1470 capture with 13° tilt and one measured reference,
all eight whole-print and eight motif previews land within 1% aspect error.
The maximum error is 0.21% for whole prints and 0.59% for motifs. None of the
16 previews samples foreign pixels or cuts into the subject beyond the
one-source-pixel tolerance. See `measured-aligned-page-geometry.json` and
`measured-aligned-page-preview.png` in `runs/crop-quality/curved-geometry-probe/`.

The `rotated_prints` development fixture places 12 bordered prints on one
plane tilted by 31°. With approximate seeded masks, whole-print previews
within 1% aspect error rose from two to 12, and motif previews from one to
12. The 24 candidate previews have zero foreign source samples and zero
cutoff beyond one pixel. Before separated contrast recovery, the registered
detector and fallback matched one of the 12 instances at mask IoU 0.85.
The paired geometry and inference reports
are `gutter-split/uncalibrated-page-{control,candidate,registered}.json` under
`runs/crop-quality/`.

On densely mounted pages with many small masks and detector warnings,
PhoSca reuses the registered ONNX model on four overlapping
windows. It excludes proposals that reach an artificial tile boundary and
compares masks back in source coordinates. Agreement across views helps
resolve competing proposals, while separately observed overlapping prints
can both remain. Tile detections have their own source in confidence
calibration. The approach draws on [Slicing Aided Hyper Inference](https://arxiv.org/abs/2202.06934).

On the development scan of the [Met Album Page of Portraits](https://www.metmuseum.org/art/collection/search/281929),
matches at mask IoU 0.85 rose from 35 to 44 of 48, false positives fell from
14 to 2 and mean matched mask IoU rose from 0.787 to 0.921. The second,
sparsely mounted Met pilot still matches four of five with the registered
detector. The locked synthetic regression passes 18 detections and all 23
recoverable crops. The same-album Golden regression matches all 29 prints.
The reports are `met-pilot/dense-final-registered.json`, `dense-tiles-locked.json`
and `dense-tiles-golden.json` under `runs/crop-quality/`.

Partial tile coverage now contributes local evidence on dense pages,
including when the initial proposals have invalid geometry. Full-page pairs
remain available where no tile replaces either proposal. At 66% resolution
on the dense Met development scan, the paired test improved from 30 to 33
matches and from 16 to 10 false positives. Mean matched mask IoU rose from
0.735 to 0.819. Native results remain 44 of 48 with two false positives.
The paired reports are `dense-cutoff-probe/dense-tracked-{control,candidate}.json`.

Before classical fallback, a risky full-page result can retry on a rotated
inference view when dispersed horizontal and vertical page boundaries agree.
The view includes the entire source raster, runs the registered model and
overlapping tiles, and maps the masks back to source pixel centres. Candidate
masks from padding are discarded. The retry needs more source-supported
small prints than the first pass. Cropping still samples the original.
[OpenCV line detection](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html)

On rotated derivatives of the dense Met development scan, the retry finds
32 of 48 prints at −3° and 25 at +3°, with seven and 13 false positives.
The previous full pipeline exceeded 35 seconds on both cases in the paired
run. Retries for ±5°, ±7° and ±10° also bypassed classical fallback, with
19 to 30 matches. The native scan stays at 44 of 48. Blank and upright
internal-line controls abstain. The cases and mask overlays are under
`runs/crop-quality/deskew-risk-probe/`.

The classical fallback now requires the line segments to end near the four
candidate corners and preserves separate collinear traces from neighbouring
prints. The frameless diagonal lattice previously generated 55,350 line-pair
proposals. The new pipeline returns zero detections on that input in about
0.6 seconds in the local run. On an aligned synthetic pair with broken
corners, the two seed quads reach 0.996 IoU and at most 1 px corner error.
The locked synthetic and same-album Golden metrics match the previous run.
See `classical-line-{locked,golden}.json` and
`classical-fallback-v17/fallback-verification.jpg` under `runs/crop-quality/`.

On risky pages, separated contrast recovery examines components before
closing can bridge the narrow album paper between them. It verifies the
individual stock strip, the surrounding album material and the four source
edges. It retains an enclosing print with an observable exterior outline.
Hollow illustrations and plain labels fail the interior checks.
[OpenCV morphological operations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)

The full pipeline now matches all 12 prints on `rotated_prints` with mean
mask IoU 0.994 and zero false positives. All 24 previews have aspect error
under 1%, no foreign source samples and no cutoff beyond one source pixel.
They remain review previews because their aspect is inferred from the page.
On `dense_gutters`, matches at IoU 0.85 rise from 15 to 21 of 30. The native
Met development page remains at 44 of 48. The paired end-to-end reports are
`gutter-split/uncalibrated-page-registered.json` and
`gutter-split/separated-contrast-development.json` under `runs/crop-quality/`.

An outline can resemble a photo mask even when it is only ink on an album
page. The source check samples both sides of the candidate after rectifying
the quadrilateral. It fits a local paper colour gradient and rejects the
candidate when narrow strokes trace all four sides on the same empty paper.
[OpenCV projective sampling](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)

The registered detector produces 56 false proposals on the drawn-frame
development page. The source check removes all 56 in the original and in
the rotated, recompressed, gradient-lit and grain controls. On the mixed
synthetic page it removes 53 drawings and retains all three faint or sparse
print examples.
The comparisons are `deskew-risk-probe/ink-frame-end-to-end.json`,
`deskew-risk-probe/ink-frame-native-comparison.jpg` and
`deskew-risk-probe/ink-frame-mixed-comparison.jpg` under `runs/crop-quality/`.
A separate ruled-label experiment also flags striped photographs and a
faint Met portrait. Its scores remain exploratory in
`gutter-split/label-artifact-probe.json`.

The Emily Howland intake pairs the album page with a separate front scan
of the same Harriet Tubman carte-de-visite. The window covers about 35% of
the complete card and 24% of the printed image. The p95 registration error
on 486 held-out feature pairs is 1.41 px in the album scan. The draft
treats both modes as unrecoverable from the album view. The combined preflight
at `runs/crop-quality/archive-reference-review-plus-howland.json` covers 12
pages and 31 photos and verifies all file hashes. The annotations are still
pending independent review. [Smithsonian dimensions](https://nmaahc.si.edu/object/nmaahc_2017.30.47)
and [Library of Congress album and print record](https://www.loc.gov/pictures/item/2018645050/).

`whiting-paired-crops-draft.json` pins the original TIFFs, the fitted
quads and held-out landmarks for two prints in Whiting's *Diary in photos*,
volume I. The Library of Congress catalog distinguishes the album page
series `17161` from the separately scanned prints in `17162`. The TIFFs
record 200 dpi for the page and 400 dpi for the print scans, with acquisition
timestamps on different days. The independently fitted corners agree within
0.66 native album pixels across both crop modes in the original whole-side
fits. A local recheck of print 81's motif moved the lower-left corner 3.56
native pixels, with all four revised corners agreeing within 0.46 px across
the two scans. The source motif polygon now traces the curved left edge.
Its residual from the corner-to-corner chord is 3.21 px p95. Use the
polygon for pixel coverage and the four corners for aspect and corner error.
[Library of Congress catalog](https://www.loc.gov/pictures/item/2007675295/)

Use the three source URLs in the draft manifest to place the TIFFs at their
pinned paths under `runs/crop-quality/whiting-diary-pair/`. The evaluation
command renders the four reference crops directly from those masters.

```powershell
python scripts/evaluate_onnx_segmentation.py --dataset-manifest data/evaluation/whiting-paired-crops-draft.json --check-references --visuals --output runs/crop-quality/whiting-native-reference-review.json
```

The source-only intake at `runs/crop-quality/archive-reference-review-plus-native-whiting.json`
still needs independent review. It has 13 missing metric references, down
from 17 before adding the Whiting pairs.

`whiting-more-native-crops-draft.json` adds five further whole-print
references from separately digitized TIFF masters. The maximum disagreement
between independent corner fits was 0.73 native album pixels, with up to
0.26% aspect-ratio disagreement. A sixth paired print retains an unresolved
top edge. The combined preflight at
`runs/crop-quality/archive-reference-review-plus-five-whiting-native.json`
checks all pinned files and leaves eight missing metric references and 61
mode annotations awaiting independent review. [Whiting album catalog](https://www.loc.gov/item/2007675298/)
and [American Colony album catalog](https://www.loc.gov/pictures/item/2007675258/).

The physical volumes in these Whiting receipts have three catalog identities.
*Diary in photos*, volume I (LOT 13836, series `17161`/`17162`) has been used
in development diagnostics and belongs outside the final split. *World War I
in Palestine and the Sinai* (LOT 13833, `13708`/`13709`) and *Members and
activities of the American Colony* (`15831`/`15830`) are separate albums.
The drafts retain their conservative source grouping while prior use and
same-print duplication across the albums are checked.

The American Colony pair `15831-00019` / `15830-00057` has a whole-stock
corner disagreement of 0.135 native pixels and held-out landmark p95 of
0.397 pixels. The motif still needs a source-only decision on the shared
paper and image boundary. The TIFFs record 200 and 400 dpi and timestamps
three seconds apart. Separate acquisition still needs a comparison of
the raw pixel grids and capture metadata. [American Colony album catalog](https://www.loc.gov/pictures/item/2007675258/)

The International Bomber Command Centre catalog pairs four mounted Quine
album pages with larger images of all 14 prints. A separate Langworthy
album page has six represented prints and multiple close-ups of one print.
Matching image features and withholding each spatial quadrant from its
homography fit gave a per-pair 95th-percentile reprojection error of
0.75–1.66 native page pixels for the Quine images and 0.96–2.26 for the
Langworthy images. This identifies corresponding prints. Review capture
provenance, mount occlusion, and both sets of crop boundaries before using
the pairs as benchmark truth. The archive publishes these images under
CC BY-NC 4.0. [Quine page 24962](https://ibccdigitalarchive.lincoln.ac.uk/collections/document/24962),
[page 24963](https://ibccdigitalarchive.lincoln.ac.uk/collections/document/24963),
[page 24973](https://ibccdigitalarchive.lincoln.ac.uk/collections/document/24973),
[page 25096](https://ibccdigitalarchive.lincoln.ac.uk/collections/document/25096),
and [Langworthy page 34635](https://ibccdigitalarchive.lincoln.ac.uk/collections/document/34635).

`ibcc-quine-paired-crops-draft.json` pins the mounted page 24963 and its four
individual scans. The page EXIF identifies a Nikon D750 capture. The catalog
identifies the larger images as individual scans and all four show stock
hidden by the red mounts on the page. Each whole-print reference is marked
obstructed. Each motif is visible, with the source corners for print 0015
left open because the upper edge is too faint. The mounted top edge of
print 0016 bows 5.72 native pixels from the chord between its corners,
with 0.94 px 95th-percentile residual across 30 supported edge samples.
The draft traces that bow in the source polygon and measures aspect from
the separate scan. A single homography maps straight lines to straight
lines, so this is a real case for the one-pixel cutoff and contamination
checks. All annotations await independent review.

Download the page's `image_url` and the `reference_image_url` for each print
to the paths pinned in the manifest. The source files stay local. Check
hashes and render the side-by-side review without running the detector.

```powershell
python scripts/evaluate_onnx_segmentation.py --dataset-manifest data/evaluation/ibcc-quine-paired-crops-draft.json --check-references --visuals --output runs/crop-quality/ibcc-quine-24963-preflight.json
```

The synthetic dewarping comparison in `runs/crop-quality/curved-geometry-probe/`
uses exact contours and known aspect for eight shapes. The boundary-guided
remap removes all strict cutoff and contamination samples across both modes.
Interior landmark error still reaches 42.68 px p95 for the strongly curved
cylindrical full print. Two samples with identical contours differ by
13.67 px p95 in interior alignment under that same remap. Score landmark
error as well as the footprint before promoting a curved crop. Boundary
remapping is also used in [document dewarping research](https://arxiv.org/abs/2501.03145).

A separate cylinder fit recovered three synthetic cylinders in both modes
with at most 0.24 px landmark error at the 95th percentile and zero strict
boundary errors under 0.35 px trace noise. It used the true focal length
and independently known aspect. A 5% focal-length error introduced foreign
samples in all six crops. An interior bulge with the same outline still
caused 11.35 px landmark error at the 95th percentile with zero strict
boundary errors. See `curved-geometry-probe/cylinder-results.json` under
`runs/crop-quality/`.

The motif guard measures supported paper-to-picture crossings across the
edge and checks for coherent bowing in rectified pixels. A bow above 2.25 px
stops the straight motif crop and leaves the full-print mode available when
its own geometry passes. The same check applies to the legacy border-trim
plan. Version 22 in the 28-case development probe sends eight previously
ready motifs with strict cutoff or paper samples to review. All four safe
motif crops and all 24 ready full-print crops still pass the strict pixel
checks. The traces and visual comparison are under
`runs/crop-quality/curved-geometry-probe/`.

Version 27 samples the paper-to-picture transition again in native pixels
when four paper margins support the motif estimate. It compares measured
segments against the proposed straight side, including bends that a single
quadratic fails to capture. A clear disagreement keeps **Nur Bildmotiv** in
review. The desktop editor and evaluation contact sheet draw the measured
segments on the original page.

In a geometry-only replay from the pinned TIFFs with known print corners,
the former unmeasured Whiting 76 motif lost 422 source pixels beyond the
one-pixel band against its draft reference. The new check puts it in review.
For Whiting 81 with a measured outer aspect, the left trace is within
0.55 native pixels at p95 of the paired draft contour. The replay isolates
the boundary logic from detector and face-guard behavior. See
`whiting-v27-runtime-native-traces.json` under
`runs/crop-quality/whiting-independent-review/`.

The same archive pair shows why the native outline and the deformation must
be tested separately. A perspective warp that preserves the silhouette on
an expanded transparent canvas keeps all measured source coverage. An
opaque rectangular warp following all four curved edges also clears the
one-pixel coverage check, but its landmark error against the independently
captured print reaches 5.13 output pixels at p95. Both captures contain the
bent edge. See `whiting-81-planar-alpha-vs-curved.json` in the same directory.
For a genuinely uneven surface, interior evidence is needed to validate
flattening. [Grid regularization study](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Revisiting_Document_Image_Dewarping_by_Grid_Regularization_CVPR_2022_paper.pdf)

Version 28 renders a separate PNG for contour review when all four motif
traces admit a simple closed outline and the original pixels confirm both
arms of every extrapolated corner. The renderer translates the oriented
print homography onto a canvas that encloses the contour. It interpolates
premultiplied colors and ownership together, then stores straight colors
with alpha. The preview asset appears under `image_only_preview` while the
motif's download keeps its regular release requirements.

In the 23-case synthetic contour comparison, 11 review previews render.
All 11 have zero cut, foreign ownership and foreign interpolation samples
beyond one source pixel, with a minimum outline IoU of 0.9971. Each of the
three straight controls keeps its ordinary crop. The other nine cases lack
a corroborated contour or print geometry. Whiting 81 still has a 166-pixel
gap in the measured left edge. PhoSca retains its review traces as evidence.
See `v28-production-outline-preview.json` in the contour report directory.

Version 29 recovers tentative measurements between strong native motif
samples. On the paired Whiting scans, the longest left trace interval for
print 81 falls from 166.06 to 62.27 native pixels. Print 76 still has just
two traced sides. Both motifs stay in review. The supplied weak-seam probes
expect a closed outline in ten cases where the current continuity rules
withhold it. See `runs/crop-quality/cloud-v29/`.

Version 30 checks for a second, sustained color step alongside the selected
motif transition. When an incomplete trace still suggests a straight crop,
the native line fit must corroborate its sides and corners before the mode
can be ready. In a nine-case production replay, seven ambiguous motifs stay
in review without a derivative, both control derivatives still render, and
all nine whole-print crops remain ready. The v29 replay exposed up to 356
source pixels cut beyond the one-pixel band and 81 foreign output samples.

Geometry-only, Registered and Local each keep 18/18 frozen detections and
11 passing whole-print and 12 passing motif crops. The Registered and Local
ready crops have at least 0.9905 footprint IoU, at most 0.755 native pixels
of corner error, at most 0.283% aspect error and at most 0.646 output pixels
of landmark p95 error. All have zero cutoff past one source pixel and zero
foreign interpolation samples. Reports and overlays are under
`runs/crop-quality/cloud-v30/`.

The wider shadow probe varies shadow contrast from 22 to 70 gray levels,
weak-run width from 15 to 47 pixels, and edge displacement in both directions
on straight and curved motifs. Version 30 produced a faulty derivative in
24 of 48 cases when the second step was small relative to the shadow.
Version 31 uses local noise and spatial consistency to assess that weaker
step. All 48 motifs stay in review without a derivative, all whole prints
remain ready, and the three frozen runs retain their previous pass counts.
See `runs/crop-quality/cloud-v31/shadow-amplitude.json`.

Version 32 shows connected strong measurements from rejected motif sides
in the review editor and leaves gaps between them. On Whiting 76 it adds
four top segments with 89 samples and five right segments with 109 samples.
Their positions agree with optical transitions in the independently scanned
print to 0.50 and 0.60 native pixels at p95. The upper-right gap still needs
review. A weak one-sided extension was measured separately and remains a
diagnostic. Neither Whiting motif yields a derivative. See the paired audit
and overlay under `runs/crop-quality/cloud-v32/`.

Version 33 checks whether source observations support a straight outer edge.
It sends both modes to review when the fitted print boundary shows a clear
bow on one side or corroborated bowing on adjacent sides. In the 48-case
development replay, this catches five geometrically ready failures across
the two crop modes at moderate or strong radial distortion. All six control
pages and the ready cases from the 18 blur-only pages retain their previous
results. A weakly distorted tilted print still exceeds the 1% aspect
tolerance in both modes (1.11% and 1.24%). The frozen detection and crop
pass counts are unchanged. See `runs/crop-quality/cloud-v33/`.

Sampling the Whiting reference polygons at source pixel centers changes the
strict whole-print cutoff counts from 565 to zero on print 76 and from 501 to
124 on print 81, with the same rendered crops and one-pixel tolerance. Print
81's remaining cutoff is on its ambiguous right edge. These measurements use
draft references. See `runs/crop-quality/cloud-reference-raster/whiting.json`.

Version 34 keeps both crop modes in review when the native print transition
is too broad to certify ownership. The Registered replay on 31 development
pages with blur, distortion, overlap and mounts routes all four previously
incorrect ready whole-print crops to review. A passing blurred cast-shadow
crop is also withheld. Registered and Local both pass every remaining ready
crop in this replay. The detection results still include misses and extra
instances. Switching to original pixels or sharpening the detector input
reduces some mask errors while losing passing crops or adding duplicates.
The frozen runs retain 18 detections and 11 whole-print and 12 motif passes.
See `runs/crop-quality/cloud-v34/`.

Version 35 uses local stock and backing observations from three fitted edges
to guide a nearby ambiguous fourth edge in the review preview. The material
transition must continue on both arms of each affected corner. A neighboring
print can veto the proposal. The ambiguous crop keeps its review status.
In the Whiting 81 Registered replay, whole-print cutoff beyond one native
pixel goes from 124 to zero, foreign interpolation samples go from 972 to
zero, and maximum corner error against the paired draft drops from 1.91 to
0.93 native pixels. Landmark p95 error drops from 1.75 to 1.38 output pixels,
with 0.60% aspect error. A stock-colored exterior strip can still resemble
the same transition in one capture and requires review. The frozen detection
and crop pass counts hold across Geometry-only, Registered and Local.
See `runs/crop-quality/cloud-v35/`.

The frozen seeded-geometry benchmark passes all 23 recoverable mode crops.
The v21 and v22 Registered end-to-end runs match six of 18 raw instances
at IoU 0.85. On the development white-margin and white-on-white pages the
raw masks recovered only 3.58% and 1.97% of the physical rim.

Default precise segmentation in v23 measures original-page contours before
crop rendering. Fully supported stock supplies a quad. Obstructed or curved
stock can contribute an observed visible rim. The renderer uses the raw
polygon for its geometry seed. Two candidates that try to add the same interior pixels
outside both seeds keep their original masks and request review. The report
records the raw proposal scores and the evaluated source masks. Calibration
binds to the source contour revision.

On the synthetic development pages this raises mask matches at IoU 0.85
from 40 to 60 of 60 and lowers serious segmentation failures from 27 to zero.
The frozen run matches 18 of 18 and passes 11 whole-print and 12 motif crops.
The same-album Golden check keeps 29 matches. The two Met development drafts
still give 48 of 53 matches and two false positives. See the `v23-source-contours-*`
reports under `runs/crop-quality/curved-geometry-probe/` and
`runs/crop-quality/met-pilot/`. The optional registered local-mask artifact
also passes the frozen split, with 45 of 53 matches on the same Met drafts.
In four alternating measurements per Met page, the source contour check
adds 108 ms on the sparse scan and 274 ms on the dense scan to median
segmentation time. See `v23-source-contour-timing.json` in the contour reports.

When a contrast component has three strongly supported sides and one
ambiguous side, v25 retries that side with a three-pixel search radius. A pale
paper rim must surround both quads, and the exterior strip at the refitted
edge must agree with the album backing. The neighboring-photo, corner and
content guards still apply. A band with nearly the same color as the album
can be part of the print. Both modes carry `PRINT_EDGE_CONFIRMATION_REQUIRED`
when the retry supplies their geometry. Confirm the corners in the editor.

On the default `dense_gutters` fixture, with the synthetic camera intrinsics
available to the renderer, two additional previews per mode pass the pixel
and aspect checks. Their largest source corner error is 0.285 px. With
intrinsics withheld, one additional preview per mode passes. These results
leave the ready counts unchanged. In a separate near-background surround
stress test with fresh contrast proposals, all 47 affected retries require
review. See `dense-v25-production.json` and
`edge-retry-production-regenerated.json` under
`runs/crop-quality/curved-geometry-probe/`.

Version 26 also checks for a faint step outside an apparently complete
contrast-component boundary. The proposed edge must carry a consistent
luminance step along its span. For an extended review preview, both adjoining
sides must show the same material change up to that edge. Known neighboring
prints can veto the added strip. When a coherent step remains unresolved,
the old quad stays in review. A similar strip can belong to a shadow or an
unrecognized neighbor, so either result requires confirmation at the source.

In 29 synthetic cases with freshly generated contrast proposals, previous
geometry with calibrated intrinsics cut off content while marking all 29
whole prints ready. Version 26 reduces that count to 11. Without camera
intrinsics it reduces the count from eight to four. Fourteen calibrated
previews meet the one-pixel footprint tolerance, with a maximum corner
error of 0.790 px. Twelve of those also have zero foreign interpolation
samples. See `edge-v26-production.json` in the same report directory.

`nmah-gardner-paired-crops-draft.json` pins two independent Smithsonian scans
of the same Marshall House accession. The fitted aspect ratios differ by
0.24%. After registration, the largest corresponding corner discrepancy is
3.29 native page pixels. Review the faint stock boundary and whether it
coincides with the motif before confirming either mode. The museum records
an overall size of 9 by 7 inches. The draft uses the separate scan as its
aspect reference. The source images stay local under the museum's usage
conditions. [Smithsonian catalog](https://www.si.edu/es/object/plate-1-marshal-house-alexandria%3Anmah_1293152)

`getty-talbot-measured-review-draft.json` pins two native Getty images with
catalogued print dimensions of 181 × 170 mm and 203 × 166 mm. The provisional
source fits differ from these aspect ratios by 0.26% and 0.06%, respectively.
The draft requires independent verification of whether the outer stock and
motif boundaries coincide on each print. Download the two `image_url` assets
to the paths in the manifest before running `--check-references --visuals`.
[The Ladder](https://www.getty.edu/art/collection/object/107R3B) ·
[Part of Queen’s College, Oxford](https://www.getty.edu/art/collection/object/108GG5)

The Brady/Pedro II volume published by the Library of Congress repackages
individual card scans. Four paired volume and catalog image requests
returned identical rasters and point to the same TIFF assets. Its catalog
dimensions are rounded to millimetres, giving up to 1.55% aspect uncertainty
under nearest-millimetre rounding. The source needs a separate mounted
album view to establish the two crop modes. [Volume catalog](https://www.loc.gov/item/2021670872/)

```powershell
python scripts/evaluate_onnx_segmentation.py --dataset-manifest data/evaluation/nmah-gardner-paired-crops-draft.json --check-references --visuals --output runs/crop-quality/nmah-gardner-marshall-preflight.json
```

A development experiment zoomed into the bounded gap on the
[sparsely mounted Met page](https://www.metmuseum.org/art/collection/search/289066).
It found the central fifth print in the original scan at 0.904 mask IoU.
Its uncertain top edge left 9.6% of the draft footprint outside the proposal.
The new branch added no detections in 14 modified negative cases, but missed
all 13 resized, recompressed or geometrically transformed positives in the
robustness check. The diagnostic reports are in
`runs/crop-quality/dense-tiles-probe/gap-negatives-v2/`. The runtime keeps the
validated detection path while this case remains unresolved.

## Multipanel development and reproducibility

The shared-stock check measures the perimeter on the original page when
the proposals include an enclosing print and at least two adjacent motifs.
Without an enclosing proposal, it looks for connected pale stock around
rows and grids of up to six neighboring motifs. It checks every full paper
separator and the portions between crossing separators on the original
pixels. A join in one row can then veto grouping even when the other rows
share continuous stock. At a narrow gutter, stock observed inside a recovered
print can distinguish its outer edge from the facing edge of its neighbor.
The bounding geometry of the recovered print must still be valid.

The search tries the entire connected proposal group, groups with one
neighbor removed, and individual pairs. Irregular layouts without full
separating strips and competing outlines keep their original proposals.
Pages with more than twelve plausible pairs keep their proposals at this
stage. Every recovered print requires review in both crop modes.

`backend/tests/test_pipeline/test_multipanel_prints.py` covers rotated
pages, narrow gutters, partial joins in a grid, a separate neighboring print,
cross-detector fusion, and the upload, review and export flow with and
without an enclosing proposal. A separate replay on the ten Golden originals, with native,
resized and JPEG75 views and exact or inset inputs, found no false merges
in 198 separate-print pair checks. The Golden pages come from one album
and provide separate-print controls.

Rerunning the frozen seeded geometry gives 11 passing full prints and 12
passing motifs. The model-free fallback before and after the change matches
12 of 29 on the same-album pages with 16 false proposals, and 13 of 18
on the frozen synthetic pages with six false proposals. These fallback
runs have no registered detector digest.

The model registry pins the expected ONNX binary with SHA-256
`4d8235880ea1868a0ce6c77fd313522ae979329eed1441385cb2c256e2cf960f`.
The source split lists the legacy train, validation and Golden filenames,
and the training summary gives aggregate counts for the additional drafts.
Those records leave the additional per-image hashes, archive IDs and
assignments unresolved. The Cloud runs, imported receipts and ONNX binary
are needed to replay the reported nine-view comparison. Restore the
exact artifacts or document the provenance of a replacement detector and
reserve independently sourced albums for its final evaluation.

The Commons catalogue of Adam Harangozó's Hungarian family album is a
candidate for one physical group. It links individual files 1 and 2 to
page 1 and file 10 to page 5. The same scanner's metadata records the
individual files three to four minutes after their respective page files.
The source describes each as a scan of an owned copy. Separate capture,
usable geometry and usage rights still require verification. The original
file requests returned HTTP 403 from the network proxy, so no file hashes
or crop references from this album enter the final manifest.
[Album catalogue](https://commons.wikimedia.org/wiki/Category:Photograph_album_of_an_unknown_Hungarian_family_(1920s,_30s)) ·
[Page 1](https://commons.wikimedia.org/wiki/File:Hungarian_family_photo_album,_page_1.jpg) ·
[Print 1](https://commons.wikimedia.org/wiki/File:Hungarian_family_photo_album,_photo_1.jpg) ·
[Print 10](https://commons.wikimedia.org/wiki/File:Hungarian_family_photo_album,_photo_10.jpg)

### Paired album source intake

Three Red Album captures match separate scans of the corresponding physical
prints at the pinned commit below. The 2048 × 1536 album JPEGs record an
Olympus capture. The print scans show card stock or printed lettering beyond
the album windows. The registration median is 0.42–0.55 native page pixels
across the three pairs. The windows obscure the full-print boundary in all
three. The window also covers about 6.4% of the `REDA3.jpg` motif. The
`REDA2.jpg` and `REDA4.jpg` oval motifs appear fully visible and would
need masked motif references. Image reuse rights remain unverified. These
observations remain outside calibration and the final manifest.
[Album pages](https://rudged.github.io/home/redalbum.html) ·
[Separate scans](https://rudged.github.io/home/redalbumis.html) ·
[Pinned files](https://github.com/rudged/home/tree/9422452820380ca089db3e0baf0cfaf3dc6c7cdc)

| Page and SHA-256 | Print and SHA-256 |
| --- | --- |
| `3REDA.JPG` `9439d117430e219010834292e0dacce34e2531a4000b369363fa6b279adc2c69` | `REDA2.jpg` `b0b08afb5ac28bc991468a6dec4e4dc80843a3c610d7371a5bf791782abd47de` |
| `4REDA.JPG` `02fcf56c076ed5fd8ca6c9c0c933469c3903e5966e2d41694b55772e19d7b2a8` | `REDA3.jpg` `4443fd86e4a9bb134a739e66ca6a11d711eb0f0bf982a69448bf9926a714ce1d` |
| `5REDA.JPG` `7250cfeb2e6343f7eeba6ff08cba60b9ccbea7ced4da3bc2415eea13e9fdeca3` | `REDA4.jpg` `356bdcd589771f8565aa73c14b3ead5e1c4887bd4a5aea43908bd1f64969e91f` |

Two exposed cards in the same album provide further source-only checks.
The separate exposure of `REDA6.jpg` shows the same taped corner and
album backing. Matching the card and extrapolating to its four corners
leaves up to 9.31 native page pixels of disagreement. The small card on
`7REDA.JPG` lies loose across the gutter. The 245 × 397 individual scan
clips its stock by about 2–7 pixels per edge in that scan. The rectangular
motif is visible in both captures. Separate local fits of the ink boundary
give width-to-height ratios of 0.56782–0.56846 and agree within 1.84 native
page pixels after registration. This motif geometry remains a draft pending
review of the ink boundary and usage rights.

Independent review of the loose card distinguishes its right ink line from
the paper edge another 5.7–7.5 native page pixels to the right. The corrected
paper outline projects all four corners outside the separate front scan.
Minimizing the worst corner discrepancy still leaves 2.48 native pixels
of disagreement in the scan plane. The separate 245 × 397 back
scan also clips the card stock. Its SHA-256 is
`914375d19407f25fd10188b9a2ae62bc6376477a4b51bf501113bc05d3cd2b62`.
The whole-print reference still needs an unclipped acquisition or physical
measurements and independent confirmation.

| Page and SHA-256 | Print and SHA-256 |
| --- | --- |
| `6REDA.JPG` `fa66ebc5a49f18f37f3bf731ac020a16edc9ad862d2d0b570cee0adb228941d2` | `REDA6.jpg` `5f69b0e9bd7b0d7b1e8d690e1b5b6f10d282a6086207927b1c7cf6030ee72ce9` |
| `7REDA.JPG` `2883f364a0640f4bbb215b3495e9875b21c3642fe1f55db335be63698dfca3e8` | `REDA9.jpg` `edfdd4627fa131a4d6d99ae7735b78785befb6e39b82e2dc0f97f0b41373df84` |

The Coble Album is a separate physical album in the same repository.
`calbum3.JPG` matches `ca12twogirls.jpg`. The available page is 360 × 270
pixels and the mount obscures the print edges. The Willett album archive
provides 107 native JPEGs with inventory descriptions, photo corners,
crowded mounts and pages without photos. Its published usage is educational and
research use, subject to copyright restrictions. Verified paired print
scans, metric dimensions and usage for product calibration remain open.
[Coble Album](https://rudged.github.io/home/coblealbum.html) ·
[Willett archive](https://historycenter.mld.org/willett_album/) ·
[Willett files](https://github.com/historycenter-mld/willett_album/tree/f02315fbb8ab62f491c1003ce06614daf30cf617)

Museums Victoria lists two other physical albums with public-domain catalogue
images and measured prints. The Scott album record lists a 180 × 135 mm
print on a 280 × 360 mm page. The Hill presentation album record lists a
291 × 234 mm print on a 440 × 360 mm page. The native media requests returned
HTTP 403. The page coverage and separate motif dimensions remain unverified.
[Scott record](https://collections.museumsvictoria.com.au/items/1563024) ·
[Hill record](https://collections.museumsvictoria.com.au/items/1562410)

The Library of Congress Free to Use and Reuse bulk package and Harvard's
Smithsonian Open Access mirror give two further documented archive paths.
The package metadata, two candidate Gardner assets and the mirror's
documented control asset all returned HTTP 403. The reachable Library of
Congress companion repository provides the starter notebook. Whether
the exact pinned Whiting and Gardner rasters occur in the archives remains
unverified. [Library of Congress package](https://data.labs.loc.gov/free-to-use/) ·
[Smithsonian mirror](https://source.coop/harvard-lil/smithsonian-open-access)

## Clipped prints in the NerdScan development scan

NerdScan's original upload has
SHA-256 `6e9b834cea27386399d8beb58bb242098ee02ffadf0f6d866f2387cde733e92f`.
The photos continue past the capture edge. The photo rights and physical
album ID remain unconfirmed. The manifest and visible outlines are
development drafts. Keep the source raster local. [Pinned development source](https://github.com/klimentij/NerdScan/blob/0cb6cc25e14a023e2c55daf4698cd1c4be59ab99/input/1979/2025-04-29-0022.jpg)

With Pillow 11.3.0, apply EXIF orientation to the pinned JPEG, convert to
RGB, resize with `thumbnail((3000, 3000), Image.Resampling.LANCZOS)`, and
save the PNG as `runs/crop-quality/nerdscan-clipped-source.png`. The
manifest pins the resulting 2179 × 3000 raster and the draft label file.

```powershell
python scripts/evaluate_onnx_segmentation.py --classical-only --dataset-manifest data/evaluation/nerdscan-clipped-development.json --visuals --output runs/crop-quality/nerdscan-clipped-report.json
```

The original fallback matched 1 of 3 prints at IoU 0.85. Two proposals cut
away 35% and 45% of the respective visible prints. The new recovery requires
at least two matching detector proposals on the same capture edge, continuous
material changes along the visible sides, native pixels reaching the edge,
and distributed interior texture. It matches all three draft footprints
with mean IoU 0.999. Both crop modes stay in review, including after a
manual corner or print-size edit. The previews retain the visibly captured
portion.

The scanner check removed the remaining false contour on the upright
development scan at IoU 0.85, giving three matches and zero false proposals.
It applies to standalone classical contours crossing at least three
source-confirmed clipped prints. Two perpendicular scanner rails must
continue across two different unsupported sides of the candidate. A
coherent material change on a side prevents rejection. Compact tone
changes or bidirectional texture inside the candidate keep it in review.
The original print contours and the full-print preview metrics stayed
the same. The visibly incomplete source provides no motif release reference.

The upload flow returned review previews and withheld both downloads for
all three clipped prints in every quarter turn. At 90° and 270° a different
classical proposal remains inside the bottom print, covering about 13% of
its area. The candidate shares the parent's measured paper edge, and its
remaining sides lack enough support for a separate print. The right side
cuts a tree branch. A nearly registered laid-on copy can still reproduce
a continuous branch, so the candidate stays in review with an overlap
reason. Marking it as a wrong object through the desktop editor or the
mobile review removes it from the photo-count task. The API replay
returned three remaining proposals after that correction in each quarter
turn. A low-contrast laid-on print was preserved in the test and both
variants exported after a manual corner and size confirmation. Twenty
warmed measurements of the overlap check gave about 5 ms median.

JPEG70 in the original orientation produced three clipped prints with no
other proposal. Five independently probed dark counterexamples kept the
ambiguous area in review. Those inserted prints still need manual
localization. Twelve warmed measurements of the scanner check on the
2179 × 3000 raster gave about 57 ms median and 60 ms p95.

The frozen geometry still passes 11 full prints and 12 motifs. The
model-free Golden and frozen synthetic comparisons retain their prior
instance counts and crop measurements.

### Recover clipped prints with a unique match on the same capture edge

On the pinned NerdScan source the mixed detector retains both a near-full
model mask and a classical fragment for the lower clipped print. Both
matched the same native outline. The earlier assignment cancelled the
entire edge recovery while resolving that pair, including the unrelated
upper and middle prints.

Capture-edge revision 4 matches each native outline to at most one
proposal and each proposal to at most one outline. All matches count when
checking uniqueness, including the contenders for a disputed outline.
At least two separate matches on the same capture side must qualify.
The competing lower proposals keep their masks and crops. Scanner
backing checks use only the recovered outlines.

With the mixed candidate the pinned scan moves from 2/3 to 3/3 matches
at instance IoU 0.85. The remaining two false proposals are still in
review. For the upper print, against the visible draft footprint, mask
IoU moves from 0.6475 to 0.9990, the rendered footprint IoU from 0.6463
to 0.9987 and area cut off from 35.37% to 0.104%. The new crop has 555
outputs with some foreign interpolation contribution, all within one
native source pixel of the draft boundary. The photo extends outside
the capture. Verified print aspect and motif landmarks are unavailable.
Maximum corner error against the visible draft falls from 347.52 to
1.85 native pixels for the upper print. It changes from 2.01 to 2.09
pixels for the middle print.

The real upload reproduced all five review entries and withheld both
named downloads and the legacy trimmed alias. Geometry confirmation was
refused for the recovered upper print and the two ambiguous lower entries.
Reprocessing kept those holds. Rotating the source and its existing
proposals together preserved the two matches through all four quarter
turns. Two independent qualifying capture sides leave the page in review.
The classical run on the pinned image remains 3/3 with no false proposals.
Over 40 paired warm runs, the recovery stage took 81.98 ms then 80.85 ms
at the median and 117.83 ms then 130.30 ms at p95, on one OpenCV thread
with prepared proposals and source pixels already available.
The mixed seven-view development comparison now reaches 24/24 matches
and two false proposals. Other real development views and the frozen
synthetic detection and crop metrics reproduce the previous results.

### Explain the remaining clipped-scan proposals in review

The lower measured footprint has a near-complete model proposal and a
contained classical proposal. The classical contour cuts through branches,
steps and the coat in the original. Its three visible sides have weak
support for an independent paper edge. The existing overlap review now
points to this particular proposal. A separate contour with a supported
paper edge stays independent. Both lower masks and crop matrices stay
unchanged.

The lower footprint can also supply part of the scanner-backing check.
The native outline must be disjoint, its contenders must link only to that
outline and an existing proposal must cover almost all of it. With that
footprint excluded, two perpendicular scanner rails continue through the
unrelated broad candidate. Small, dark inserts near the lower print can
produce the same signal. This additional evidence gives the broad candidate
a scanner-backing review reason and keeps it counted. The rule based on
three individually recovered prints still runs first. The revision is 5.

In the normal upload flow, desktop and mobile explain both disputed rows
and offer **Falsches Objekt**. After two explicit rejections, the three
underlying prints remain in the current album and print-count task with
their previous corners and preview JPEGs. Both crop exports and the old
trimmed alias stayed blocked. Reopening mobile geometry review shows only
the three current tasks. The desktop can still open previous rejections
and correct their geometry. The same mixed upload with a 165 × 150 px
low-contrast insert crossing the lower footprint kept the broad candidate
in review. The dark-material tests stay outside release evidence.

The pinned mixed result remains 3/3 with two extra proposals before review.
The full measured regressions for Golden, Domoritz, the contact print, the
stock challenge and the frozen synthetic sets reproduce the preceding
instance and crop metrics. On the pinned five-proposal scan, the added
source checks took the recovery step from 94.76 to 200.96 ms median and
from 216.90 to 312.65 ms p95 over 40 paired warm calls with one OpenCV
thread. Image preparation, ONNX inference and rendering were excluded.

### Keep small dark regions for scanner review

The classical detector previously discarded the broad candidate on the
pinned scan even after dark photographic material was added near the lower
print. Revision 6 checks the exposed source pixels for local tonal corners
before that discard. It compares four complete windows of Lab lightness and
excludes any window touching a measured neighboring print. The test can
hold a candidate for review when part of a dark print may be visible.
Review uses the existing candidate outline. The rail check with the
disputed third footprint feeds review directly.

Two example inserts use rectangles `(870, 2185, 168, 150)` and
`(950, 2490, 88, 90)` in native `x, y, width, height` units on the pinned
evaluation raster. To reproduce them, resize the upper source patch
`[20:871, 1060:2170]` to the input dimensions, take the pixels at the insert
coordinates and set their BGR values to `round(.038 * BGR + 40.4)`. Blend
with a rectangular alpha mask blurred at sigma 1.2 px and restore every
pixel inside the measured lower outline. Both cases now keep the broad
candidate in review and recover the three known clipped prints. The clean
scan still removes the scanner candidate. A 54 × 46 px dark insert also
covers the regression in the upload tests for both crop modes.

Paired uploads for `(865, 2185, 163, 150)` and `(950, 2490, 88, 90)`
checked the previous and current scanner logic through the normal worker.
Each changed from three rows to four. For the three known prints, the
stored masks, crop-mode metadata and original preview JPEGs matched in each
pair. Both inserted cases show the scanner hint in the mobile editor. The
two named downloads and the old trimmed alias were held.

A simulated matte scanner feature also moved to review. A narrow strip
almost entirely covered by the lower print can leave too little exposed
evidence for the new check. The pinned scan's isolated scanner check took
67.62 ms before and 92.15 ms after at the median of 24 paired warm runs with
one OpenCV thread. Instance and both crop-mode metrics reproduced across
the frozen, seeded, Golden, stock, flatbed and contact-sheet controls.

### Retain dark chromatic stock on scanner backing

Revision 7 keeps the luminance checks and also measures local chroma
at the four existing corner windows. Both arms of an exposed corner
must show a coherent change in the Lab a/b vector. Its magnitude must
exceed the variation among the other three quadrants by at least three
Lab units. All sample windows must clear neighbouring prints. An
isolated rail or a global colour cast usually changes several quadrants
together. The same material veto also applies to fallback ownership
review in fusion.

Six locations on the pinned NerdScan raster were challenged with a
dark chromatic print insert in the original image, JPEG70 and a blurred
variant. All 18 retain the scanner candidate for review. The previous
luminance-only rule would have discarded it in 16. The unmodified scan,
its JPEG70 and blurred derivatives, and the global casts agree with
the previous rule. A local chromatic scanner stain can also warrant
review. The narrow rule preserves ambiguous material, and the separate
mixed-detector ownership conflict remains counted. Very faint neutral
inserts still overlap the scanner colour evidence.

In the classical upload path, a 167 × 112 px insert at native
`(872, 1100)` differs by just one Lab lightness level at the median
and 13.56 units across Lab. The unmodified scan has three rows.
The inserted scan had three rows before the change and has four after,
with the added row flagged `SCANNER_BACKING_UNCERTAIN`. The three known
prints keep their masks, render matrices and both mode statuses.
The ambiguous row returns 404 for both named downloads and the old
trimmed alias. Its source-corrected review image returns 200. A
general approval and reprocessing preserve the hold.

The fixed-geometry scan comparison used 12 paired warm measurements
on one OpenCV thread. Median time was 112.74 to 162.84 ms and p95 was
172.46 to 232.76 ms with other regression checks active. The mixed path
with unresolved ownership already retains the candidate and bypasses
the material veto. Frozen detection and geometry, Golden, NerdScan,
Domoritz, the contact and stock controls reproduce previous detection
and crop metrics.

### Show the visible paper sides in a clipped overlap

The lower NerdScan crop already has a measured outer footprint at the
source. Revision 8 uses it when the existing conflict test associates
one nearly complete model proposal and one unsupported internal
classical contour with the same outline. It passes the measured
physical sides to the parent print's `full_print.suggested_edge_traces`.
The frame interval is excluded on every side of the raster. The native
2179 × 3000 test has two physical sides, the upper and left edges.
The right and bottom are capture limits. The child retains its
separate overlap reason and wrong-object action.

The geometry editor draws the trace over the original page. The
detected masks and existing warp serve as the active geometry. Saving
an explicit edit uses the existing manual capture-outline guard. Rejecting the
fragment leaves the parent's physical clipping and its review
requirement in place.

The source comparison shows why the two count contenders still matter.
An upright model-only replay matches 3/3 with no extra from proposal
thresholds 0.20 to 0.60. Running capture-edge measurement on those
model proposals in isolation produces roughly 0.999 draft-mask IoU on
all three. Two of the quarter-turn layouts still produce an extra
proposal. The unmodified hybrid can recover an additional generated
overlay with a two-level change in the source channels on the lower
print. The model-only source-measurement path misses that layer.
Retaining the count contenders preserves that detection. In the
180° hybrid replay the expected upper print is split among three
proposals and the result is 2/3 matched with four extras. Each piece
occupies a share of the same source-measured footprint.
The union of the two model fragments covers 91.2% of that
footprint with only 0.27% shared area. The classical fragment
adds about 0.3 percentage points to the union.
The current recovery requires a unique owner of that whole footprint.

Paired multipart uploads of the upright source produce five active
rows in both revisions. All masks, crop matrices, mode status and
source-corrected preview JPEG bytes match after excluding the new
trace metadata. Both named downloads and the legacy alias return 404.
The measured trace survives reprocessing after the fragment is
explicitly rejected. A subsequent manual rectangle and test size still
hold the clipped print and both named variants. Desktop and 412 × 890
mobile workflow checks
display the same two source sides to within 0.000001 normalized
coordinates. Both show the recapture hint and the fragment's separate
wrong-object action. Mobile disables the completeness approval while
a capture-edge or capture-outline review reason is active.

The mixed native capture step took 309.73 to 294.13 ms median and
468.31 to 508.92 ms at p95 across 24 paired warm one-thread calls while
other regression jobs were active. It uses the already measured
outline for the two coordinate traces.

### Review a shared capture corner and preserve separate prints

Revision 9 adds two safeguards to the same native source path. Contour
discovery closes gaps in a reduced raster. Before accepting the fitted
outline, PhoSca samples the original along the measured physical side
and towards the opposite capture edge. A narrow dark anchor must begin
at the paper edge. The traced groove must stay darker than both
shoulders over at least 66% of the path and reach both end sections.
Flank offsets of 4 and 7 or 6 and 10 native pixels test narrow and
wider gaps. The comparison uses Lab lightness. A supported groove
disqualifies the combined extent. The reduced search can still find
an individual sheet on its own side of a wider gap.

For a remaining capture-corner outline the recovery compares contenders
at both frame edges. The review path requires a disjoint source outline
with two well-supported visible sides, two contained model pieces with
at most 0.04 mutual IoU, at least 85% combined coverage and independent
recovery of two other prints on this capture edge. It retains a
classical proposal in the group when it crosses both model pieces and
has the same frame-side support. The stored `capture_split_review_traces`
contains only the two physical lines. Desktop and mobile count review
draw the group once while at least two proposals remain active. Both
crop modes hold `PRINT_AT_CAPTURE_EDGE`. The source traces belong to
`full_print` review. The existing mask, warp and review preview stay
available for each proposal. The ordinary named download routes and
legacy trimmed alias enforce the hold even if older metadata says ready.

The original 180° case stays at 2/3 detections with four extra proposals.
Two learned pieces cover 91.2% of the visible fitted footprint. A third
classical proposal crosses both. The measured outline has 0.9963 native
IoU against the development draft. Its two observed sides meet near
`(1149.32, 2109.29)` at 2179 × 3000. Frame limits at `x=0` and `y=2999`
are excluded from the display. Re-running the registered model in the
opposite orientation on the same source yields one plausible proposal
with 0.9521 IoU to that footprint. The same retry also proposes one
sheet for a physical two-sheet composite. A visually near-registered
simulated layer can preserve the original measured outline as well.
The deployed path keeps the contenders for review because retry
confidence and scene continuity leave the paper count ambiguous.

The paired separation controls insert a slanted scanner gap 0.5 to 8
native pixels wide in the same source, using continuous artwork,
subpixel displacement and a different photograph. The old recovery
extended one learned sheet across its neighbour. The new check keeps
both learned sheets in the seven controls, including JPEG75 and mild
blur. The measured sheet IoU that had degraded to 0.512–0.515 recovers
to 0.856–0.969. A dark printed streak that stops before the frame keeps
the paper outline. These are generated counterexamples built from a
single development scan. They measure a false-merge regression and
support the review policy for this capture pattern.

Paired modes on the three linked original proposals retain their
rectified review pixels. The standard upload populates both editors
with six active suggestions and the same two source edges. Geometry
approval returns HTTP 409, both mode exports and the old trimmed alias
return HTTP 404 and the review JPEG returns HTTP 200. The desktop and
mobile wrong-object action removes one active proposal at a time. Mobile
removes the shared count overlay when only one linked suggestion remains.
That suggestion keeps its own source overlay and crop hold.

The original 0°, 90° and 270° replays retain their previous counts and
native recoveries. Two aligned separation controls at 0.5 and 1 pixel
receive a linked review. In the blurred two-pixel control, both learned
sheet masks survive, but the minimum of two independently recovered
outlines prevents the capture pass from keeping its one remaining
unrelated fit or issuing the linked hint. A full-bleed printed line
with the exact appearance of scanner ground can also remain ambiguous.

The 180° capture pass took 1096.95 to 331.18 ms median and 1949.81 to
634.86 ms at p95 over 15 warm paired one-thread calls with other tests
running. Converting the rotated view to a contiguous raster before the
native measurements removes repeated full-image copies in OpenCV. The
frozen detector control retains 18/18 detections. The seeded geometry
run retains 11/11 whole-print and 12/12 motif passes. Aggregate
detection, geometry and crop-mode metrics match the previous revision
also on Golden, the upright NerdScan runs, Domoritz, both contact-sheet
JPEGs and the stock challenge.

`backend/tests/test_pipeline/test_capture_edge_prints.py` builds both
native gutter widths, the early-ending line and the competing corner
proposals without an external raster. To replay the source review,
rotate the SHA-256-pinned 2179 × 3000 development PNG by two quarter
turns without interpolation. Upload the resulting PNG with the same
registered mixed detector and pass both the count and geometry review
queues. The rotated raster stays local.

## Four neighboring loose prints on a flatbed

The published 600 × 851 JPEG shows four separate prints that the author
placed on the scanner. The outer polygons are development drafts. The
capture has no independent motif or metric references. Keep the image
local and out of calibration and release evidence. [Source description](https://www.domoritz.de/blog/2021/01/photo-scans)

Save the pinned JPEG as `runs/crop-quality/domoritz-flatbed-development.jpg`
and check its manifest hash before running the diagnostic.
[Pinned source](https://github.com/domoritz/domoritz.github.io/blob/1b850fd3a58519d3fa896b9da1f4d121e55434f7/images/scan.jpg)

```powershell
python scripts/evaluate_onnx_segmentation.py --classical-only --dataset-manifest data/evaluation/domoritz-flatbed-development.json --visuals --output runs/crop-quality/domoritz-flatbed-development-report.json
```

At IoU 0.85 the fallback matches all four separate prints with zero false
or merged proposals. Before this change the mean mask IoU was 0.960.
The bottom-left proposal omitted 7.8% of the draft source footprint, with
one corner 36 native pixels from the draft. An independent native-pixel
fit, anchored to the visible transitions, found 17–19 px of pale backing
between the lower prints. The source JPEG leaves a possible one-pixel
paper rim unresolved.

The new review fit uses the observed opposite side to seed a search for
one missing side. The final line must have a native material transition
with pale backing outside, and both adjacent paper edges must terminate
at its corners. The fit checks for crossing covers and for visible content
in any removed strip. A competing outline or ownership claim leaves the
original geometry in review.

Mean mask IoU on the pinned scan is 0.985. The bottom-left and bottom-right
full-print preview losses against the drafts fell from 6.4% and 4.5% to
1.8% and 1.9%. The two corrected previews have zero measured foreign
interpolation mass against those drafts. All four retain review status in
both modes. The normal upload replay confirmed the new previews and
withheld both downloads. The largest corner discrepancy for the two
corrected previews was 2.65 native pixels. Independent motif dimensions
remain unverified.

The lower-left side was recovered in all four quarter turns, JPEG75, and
a one-pixel blur probe. The blur also split the original upper-left
proposal in the baseline and candidate runs. Separate tests preserve the
original geometry for detectable pale covers, continuing pale paper,
small dark detail outside the proposed edge, and a conflicting neighbor.
Edges with virtually identical backing and paper remain ambiguous. Forty
warmed measurements of the recovery on this 600 × 851 scan gave 26 ms
median and 28 ms p95 per considered print. The model-free Golden and
frozen synthetic measures are unchanged. Seeded geometry still passes
all 23 recoverable crops.

### Split print under mild blur

The three pinned blur derivatives use the same 600 × 851 JPEG and the
same draft outlines. From the repository root, use OpenCV 4.10.0 and the
normal PhoSca decoder to create the local fixtures.

```bash
PYTHONPATH=backend python - <<'PY'
import cv2
from app.pipeline.image_ops import load_image

prefix = "runs/crop-quality/domoritz-flatbed-"
source = load_image(prefix + "development.jpg")
blurred = cv2.GaussianBlur(source, (0, 0), 1)
cv2.imwrite(prefix + "blur-development.png", blurred)
cv2.imwrite(prefix + "blur-sigma1-2-development.png", cv2.GaussianBlur(source, (0, 0), 1.2))
cv2.imwrite(prefix + "blur-jpeg75-development.jpg", blurred, [cv2.IMWRITE_JPEG_QUALITY, 75])
PY
python scripts/evaluate_onnx_segmentation.py --classical-only --dataset-manifest data/evaluation/domoritz-flatbed-blur-development.json --visuals --output runs/crop-quality/domoritz-fragments-blurred.json
```

At sigma 1 the original ranking rejected the upper-left print as a container,
although the classical detector had proposed its complete outline.
Two partial contrast proposals survived instead. At IoU 0.85 the
development replay moves from three of four matched prints and two
false proposals to four matched prints and zero false proposals. The
three other masks and full-print preview measures are identical.

The ranking retains the complete contour for review only when its
perimeter is visible against pale backing in the source. A supported
inner edge, an independent rival outline, or a step along the outer
edges vetoes the change. Several separated scene structures must cross
the proposed division at two of three near-native sampling widths.
The recovered preview for the upper-left print improved from 0.590 to
0.982 footprint IoU, with zero measured foreign interpolation mass
against the draft. The source has no independent motif reference.
Both variants remain in review. The upload and browser replay showed
the review reason and verified that neither download was available.

With sigma 1.2 or sigma 1 followed by JPEG75, the lower-right print has a weak
outer edge. Three exterior transitions and both corner arms support one
candidate outline. Connected scene features and substantial overlap among
competing fragments support offering it in the editor. A broad inner edge or
a separate print perimeter vetoes the suggestion. This includes a narrow
gutter that a bad fragment mask crosses. The suggested outlines reach 0.997
and 0.995 IoU against the full-print draft, respectively. These values score
the editing outlines. Detection and the rendered crops retain their previous
geometry. No newly held crops were counted in the two uploads. The desktop
and mobile editors can load the outline on demand and highlight its uncertain
edge. Both downloads still require review.

An independent stress test accepted no false joins in 1,116 transformed
controls with narrow gutters, touching equal stock and overlapping
prints. The real development case recovered at blur sigma 0.8 and 1.
Some stronger blur and recompressed variants still split. The same
comparison on the unchanged JPEG, the model-free Golden pages and the
frozen synthetic fixtures retained their prior results. The frozen
seeded geometry passes 23 of 23 recoverable crops. Twenty warmed
measurements of the native check on the blur fixture gave 27 ms median
and 43 ms p95 per considered parent.

#### Compare the fragments while checking print count

The current model-free replay finds all four prints at sigma 1. It finds
three and adds four false proposals at sigma 1.2 and at sigma 1 followed
by JPEG75. On those two derivatives, the lower-right print has three
proposals sharing the same outer outline. The provisional whole-print crop
cuts off about 65% of the draft footprint. The mixed development detector
finds all 12 prints over the three derivatives with zero false proposals
at IoU 0.85. The draft has no verified physical aspect or motif reference.

Desktop and mobile count review show the shared outline once while all
seven individual proposals remain visible. The dotted side identifies the
weak paper edge. Only current detections with the same four corners and
weak side can share this outline. A corrected or rejected row stops
contributing. The whole-print and motif exports still require review.
The browser replay saved the proposed corners on one row and separately
rejected the other two fragments. Count review then shows five active
rows and retires the shared outline. The unrelated extra proposal remains
available for its own review.

The original pixels support three outer sides. A matched-step probe in the
weak section of the right side found a material transition displaced by
4.0 and 4.5 native pixels from the proposed straight line. The weak segment
can be influenced by the pale sky and scanner backing. The outline remains
an editing suggestion. In 28 of 190 source-derived controls with three
supported outer sides, the same outline test crossed a known physical
separator. A fresh JPEG75 detection on a version with two unrelated
overlapping prints also gave both rows one common suggestion. Both rows
remain visible and count separately on that input.

#### Replay the available real-image development views

The original nine-view Cloud report and its exact inputs remain unavailable
in the checked Git history, local artefacts and GitHub workflow runs.
The `8fd221c` replay uses four tracked manifests with seven locally
reproducible views as a separate development comparison. The three Domoritz
derivatives share the original scan and its draft labels. They count as
stress variants of the same source.

| Development views at instance IoU 0.85 | Classical matches | Classical false proposals | Mixed matches | Mixed false proposals |
| --- | ---: | ---: | ---: | ---: |
| Domoritz original and three blur variants | 14/16 | 8 | 16/16 | 0 |
| ScanCropper original and PDF-embedded scan | 5/5 | 0 | 5/5 | 0 |
| NerdScan clipped source | 3/3 | 0 | 2/3 | 3 |
| Total | 22/24 | 8 | 23/24 | 3 |

Use the existing `domoritz-flatbed-development.json`,
`domoritz-flatbed-blur-development.json`, `scancropper-flatbed-development.json`
and `nerdscan-clipped-development.json` manifests with the evaluation script.
The classical arm uses `--classical-only` and an absent `--model`. The mixed
arm uses its registered scratch weights from the transfer receipt, with
`--include-classical-fallback`. Weights and release configuration stay under
the existing acceptance criteria. The mixed NerdScan replay cuts off 35.3%
of the draft footprint at one clipped print, and both modes remain in review.

### Scalloped paper on white scanner backing

The ScanCropper development fixture pins two original rasters with five loose
prints. The earlier scan is a PNG. The later PDF embeds one 4816 × 6870 JPEG.
The published cropped images derive from those same scans. Album membership,
individual print rights and physical dimensions need independent review.
The fixture records a single development source group. The label polygons
trace the connected paper against locally sampled scanner white at a separate
16 Lab threshold. Approximation to one native pixel keeps the labels compact.

```bash
git clone https://github.com/murniox/ScanCropper.git /tmp/phosca-scancropper-source
mkdir -p runs/crop-quality/scancropper-development
git -C /tmp/phosca-scancropper-source show 5020a66c05795940657160c12ab033e24bceb42e:example.png > runs/crop-quality/scancropper-development/earlier.png
git -C /tmp/phosca-scancropper-source show 9b6cfe39b172c350900c563b043fd37972c010b3:scans/example.PDF > runs/crop-quality/scancropper-development/original.PDF
pdfimages -j runs/crop-quality/scancropper-development/original.PDF runs/crop-quality/scancropper-development/embedded
python scripts/evaluate_onnx_segmentation.py --classical-only --dataset-manifest data/evaluation/scancropper-flatbed-development.json --visuals --output runs/crop-quality/scancropper-development-report.json
```

The earlier scan produced two candidates for one print. The inner candidate
left out 15.18% of the visible paper and 0.41% of the motif. The outer candidate
enclosed the paper with 6.73% scanner backing in its footprint. The lower
print's geometrically ready full-print crop lost 11.13% of the paper.
The new native measurements leave five matched instances and no false or
missing suggestions at IoU 0.85. The four scalloped PNG review outlines reach
0.9977–0.9986 IoU against the draft physical contours, with up to 0.072% source
paper loss and 0.162% measured foreign interpolation mass. The lower print in
the earlier scan also has a straight motif preview with 0.9988 draft IoU and
zero measured foreign interpolation mass. The remaining three scalloped
motifs still require boundary review.

PhoSca measures the repeated cuts in the source contour and confirms a
continuous pale stock band against the surrounding scanner backing. Distinct
stock and an observed layer shadow hold competing crops with their original
outlines. A single-pixel layer shadow is covered by a regression test. The
diagnostic contact sheet includes the actual transparent PNG review assets.
The upper print in the older scan retains the inner proposal as editable review
evidence. The desktop and mobile upload replay verified its marked contour,
the transparent full-print preview and the withheld downloads. A rectangle
around these four prints would contain 2.78–5.03% scanner backing against
the draft silhouettes.

### Cast shadows outside a straight paper edge

The straight-rim regressions generate measured paper and motif footprints
with neutral or warm cast shadows. The earlier outward search could extend
a print by roughly 10 native pixels by treating the dark part of a smooth
shadow as paper. The source profile now checks for a gradual fall across the
outer candidate and an independent rise near the inner edge. A repeated
profile keeps the inner fit as a review suggestion. Both modes expose the
shadow reason. The dark neutral paper rim around a bright motif still
recovers when its outer transition is sharp.

In a separate 60-case replay with exact source masks, the neutral, warm and
cold shadow tests include displaced seeds, blur and nearby stock. Nine
cases with measured background in the previous footprint now have zero
foreign pixels. Two previously ready full-print cases keep their measured
geometry and move to review. The neutral case with stronger blur remains
ambiguous and shows up to 0.12% cutoff against its synthetic truth. Median
time per case for image generation, geometry and both crops was about
0.40 s before and after the change. The paired tests check native corners
and interpolation ownership in both modes.

The lower-left straight print on the later ScanCropper raster gains one
full-print review. The measured shadow is on the right. Both warp coordinates
and motif availability stay the same. Retaining the raw instance mask reduces
mean mask IoU from 0.9965 to 0.9953 against the five draft contours. Over 80%
of that straight print's previously measured missing paper falls in the
neutral scanner-shadow transition. The exact boundary still needs review.
All five instances still match.

### Faint paper on pale backing

When the strong lines follow only the motif, search outward for a four-sided
rim. Sample the original pixels across each candidate, including beyond every
corner. A small change in the same colour direction must span the sides and
terminate at the corners. A plausible outline becomes a review suggestion.
A pale patch on album stock can make the same transitions. If strong outward
edge candidates compete on multiple sides, both modes require review and the
editor preserves the measured inner fit.

The controlled 60-case replay covers simulated paper and shadow profiles with
independent pixel masks and exact corners. The faint neutral case with
a motif seed goes from 24.74% paper excluded to 100% paper included at
raster resolution. In the upload replay both suggestions were within one
native pixel of the fixture corners. Both mode previews have zero measured
foreign samples and zero cutoff beyond one pixel in the regression test.
Stronger blur with JPEG70 and a dark stock with a neutral shadow each move
one faulty full-print result to review pending correction. Neither of those
cases obtains a verified outer boundary. A continued backing guide vetoes a
faint rim suggestion.

```bash
python -m pytest -q backend/tests/test_pipeline/test_print_rim_shadows.py
```

The normal page upload emits both faint-rim suggestions and blocks both
downloads. Desktop and mobile show the source-pixel correction hint. Saving
both measured quadrilaterals in manual review enables the two normal JPEG
downloads at the expected dimensions. The frozen seeded benchmark again
matches 18 instances and passes all 23
recoverable crops. The model-free frozen, Golden, ScanCropper and blurred
Domoritz replays use the same scoring and availability as the shadow revision.

### Line-pair search after rejected contours

Golden page 11 previously yielded no print. Three raw contours described
the same thin interior fragment and the fourth reached the page frame.
All failed assessment. The existing line-pair search now runs when the
precise classical path has no surviving detection. It uses the same
candidate checks and the already computed edge map.

The full model-free Golden replay gains one match, from 12 to 13 of 29,
with 16 false proposals in both runs. The recovered print scores 0.911
mask IoU. Its neighbour stays separate and remains undetected. The normal
upload puts both crop modes in review for the unsupported boundary and
uncertain aspect ratio. Both editors show how to correct the visible
paper edge. Neither export is available. The median classical proposal time
on this affected page was 0.98 s before and 1.30 s after in six warmed
measurements at two OpenCV threads. The frozen model-free and seeded
benchmarks and the two flatbed development sources retain their previous
results.

### Nested classical contours on the same print

On Golden page 37, a diagonal through the image produced a high-confidence
fragment. The detector suppressed a more complete contrast contour because
the two bounding boxes nearly coincided. A replacement now requires three
fitted sides shared within two native pixels, four supported outer sides, and
scene detail crossing the unsupported fragment side. A colour step found
in rectified coordinates can veto the replacement after confirmation on
the original raster. An observed neighbouring proposal can also veto it
regardless of the neighbour's relative size.

The model-free Golden replay improves from 13 to 14 matched prints out
of 29 and from 16 to 15 false proposals. The recovered instance reaches
0.994 mask IoU against the draft. Its full-print review footprint has
0.995 IoU, 0.19% cutoff and 0.50% foreign source sample mass against
the draft. A measured bow of 1.47 native pixels and the uncertain aspect
ratio keep both modes in review. The motif still needs its own boundary.
Normal upload and processing withhold both downloads. Desktop and mobile
point to the extended paper outline for correction.

The replacement improves five of 12 page derivatives. Two derivatives
already find a print above 0.90 IoU. Several resized, rotated and recompressed
cases still return the partial contour. In 120 generated touching, gapped
and stacked controls with slanted joins, light blur, JPEG and quarter-turn
rotations, the rule does not accept a combined contour. Eight continuous
scene controls pass. On the affected original, the median classical
proposal time was 1.28 s before and 1.39 s after in six warmed measurements
with two OpenCV threads. The nine other Golden pages, frozen model-free
baseline, 23 seeded crops, and the ScanCropper, ScanSplitter and blurred
Domoritz diagnostics retain their previous results.

### Review suggestion beside the album edge

On Golden page 37, three sides of the adjacent print are strongly supported.
The fourth follows the narrow strip between the print and the album edge.
Three connected source lines seed a proposal when short segments support
the fourth side and a staggered neighbouring print is already observed.
Original-pixel profiles corroborate a bright band between two transitions.
The suggestion reaches 0.988 polygon IoU against the draft for that print.
On page 11, the missing neighbour still has insufficient evidence for this
proposal.

Generated controls with a white print border or scenes printed on shared
paper can reproduce the same evidence. The suggestion adds an outline to
the existing review item. It leaves the observed instance, its warp and the
page's detected count unchanged. Desktop and mobile show the dashed contour
against the original and ask whether it is a separate piece of paper.
On desktop, using the suggestion starts a manual addition with four editable
corners. The normal upload replay kept one detection until the edited corners
were explicitly saved. The added photo still required aspect review in both
modes and neither download was available.

The proposal adds a review reason to the already-held Golden case. The
frozen model-free and seeded benchmarks and the ScanCropper, ScanSplitter
and blurred Domoritz checks keep their prior instance and crop metrics.
The new search took 176 ms median on page 37 and 137 ms on page 11 over
six warmed repetitions with two OpenCV threads.

### Confirming a suggested print exactly once

Before confirmation was bound to the displayed suggestion, a second editor
could add another manual record using a fresh request identifier. The
duplicate could then reopen the first print's exports for overlap review.
The workflow now uses the source photo and displayed outline as a shared
identity. A successful save removes the suggestion in the same transaction
that creates the print. If rendering fails, the suggestion and photo count
stay as they were. An exact retry returns the saved record. A competing
or outdated edit refreshes the review view and shows a conflict. Manually
marked overlapping prints with distinct outlines still use overlap review.
Matching corners from an older editor also retire the stale suggestion.

The desktop addition form can record the measured width and height of the
numbered edges at first save. The upload test ran two desktop editors on
Golden page 37. Their saves returned 201 and 409, the exact replay returned
200, and the page ended with two records. Both crop downloads still need
the missing metric evidence for this real page. A separate controlled API
fixture supplied measured dimensions, saved the motif boundary and verified
both downloads, including protection against stale and conflicting requests.

### Crossing contours on a pale album leaf

On Golden page 04, a weak edge proposal crosses the middle of a separate
print and extends over neighbouring prints. A pale-page search models the
lighting on smooth backing, opens narrow gaps between foreground regions,
and checks four sides at the original resolution. The original also shows
pale backing on opposite sides and a distributed shadow outside one edge.
The resulting editable contour reaches 0.994 IoU against the Golden outline.
Its largest corner error is 2.75 native pixels. A diagnostic whole-print
render has 0.997 footprint IoU, 0.087% source cutoff and 0.327% foreign
source sample mass. No cutoff exceeds one pixel from the reference edge.
The source has no independent metric aspect or motif landmark measurements.

A printed gradient on shared stock can imitate a real shadow. Two adjacent
prints with flush edges can also appear to be one piece, and a pale cover can
hide part of a print behind an apparently complete boundary. The proposal
is a dashed correction in review. It does not replace a detected mask or
add an instance. A reviewer can load and adjust the corners, or reject the
existing detection. Submitting only motif corners, submitting a box, or
moving the old outline by at most two native pixels leaves review open.
Material overlap after a correction holds the edited crop and any already
released overlapping crop for review.

Normal upload and processing showed the same outline on desktop and mobile.
Saving the whole-print outline kept the existing photo ID and count. Both
download endpoints still returned 404 without a measured aspect. A
controlled API fixture supplied measured dimensions, then saved the
motif, downloaded both JPEGs, and verified that a later overlapping edit
withheld both exports. The frozen geometry remains 18/18 instances and
23/23 recoverable crops. The frozen classical, Golden, ScanCropper, blurred
Domoritz and mounted ScanSplitter replays retain their instance and crop
metrics. Golden still matches 14 of 29 prints with 15 false proposals.
These replays had zero newly held usable crops. The native search on the
affected original took about 350 ms median over six warmed runs with two
OpenCV threads.

Two additional Rijksmuseum album records identify different physical
groups. The Roelofs album lists 89 prints, including an 80 × 110 mm print.
The second album lists 81 photos, including a 51 × 83 mm portrait of
Wilhelmina. The catalogue marks the albums and example photos public
domain. The official Linked Art and IIIF image requests returned HTTP 403
in this environment. Separate capture and motif geometry are still
unverified, so neither group contributes to the release split.
[Roelofs album](https://id.rijksmuseum.nl/200322643) ·
[Dutch album](https://id.rijksmuseum.nl/200322353)

### Measured paper curves in review

The print-edge fit now retains short runs of supported original-pixel
measurements when it reports a curved paper edge. Desktop and mobile show
these runs as dashed lines next to the editable straight corners. A missing
run or a jump between measurements stays open. No fitted curve is extended
to the corners or used as a crop mask.

Golden page 37 shows two measured segments in the ordinary upload flow.
Both crop downloads are still withheld. Deliberately distorted camera
controls place the displayed points within one native pixel of the
independent synthetic boundary. The frozen seeded run passes all 23
recoverable crop checks. Model-free Golden and the frozen detection run
retain their prior instance, warp and mode-availability metrics. A
paired native boundary benchmark on page 37 took 79.5 ms without tracing
and 79.8 ms with tracing over ten warmed runs per version at two OpenCV
threads. A control with a 4-pixel outward bow loses 1,598 source pixels
when its four corners alone are saved as a
manual rectangular crop. Rendering a complete closed contour would require
independent observations of all sides and corners. The segments in this
change provide review evidence for that work.

The flank prints on Golden page 04 remain separate investigations. The
left outline sampled at 32 vertices comes within 0.81 native pixels of the
independently measured outer edge. The straight outer chord deviates by
up to 5.82 pixels. Adaptive pale-corridor measurements support 96.4% of
the outer edge and reject flush album edges and continuous covers in the
controls. Printed frames, scene panels and some shadow and glare patterns
can still reproduce the corridor. On the right, the top join shows a
4.0–4.9 pixel gap between the prints. Part of the adjoining sky has no
resolved second edge. The right outer edge also departs from the corner
chord by about 7.5 native pixels. These findings are source diagnostics
for page 04. The detector has insufficient evidence for a closed review
outline for either flank.

A one-parameter radial fit with a fixed image centre reduced held-out
line error on six of nine measured side segments. Error increased on
the other three, including from 0.82 to 1.12 pixels RMS on the lower
left side. With no independent proof of optical distortion or plane
geometry, these measurements are insufficient to apply the same warp
across the page. Published document dewarping methods can also use
text-line constraints that album prints lack. [Grid regularisation study](https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_Revisiting_Document_Image_Dewarping_by_Grid_Regularization_CVPR_2022_paper.html)

### On-demand source edge measurement

Desktop and mobile review let the operator mark three points on a visible
edge, magnify the original, and move a point one original pixel at a time.
The server looks for a sustained source transition around the marks. The
blue dashes show the observed sections and leave gaps at competing or
unsupported transitions. The measurement is a temporary overlay. Paper
ownership still needs visual confirmation. Strong borders printed onto the
stock and hard lighting transitions can produce the same contrast.

On Golden page 04 the left outside edge is measurable over more than 1,000
native pixels. Its trace departs by more than four pixels from the corner
chord. Sections of the right top, outside and bottom edges are measurable.
The outside trace leaves about 240 native pixels open, and the bottom
trace leaves about 182. The adjoining sky on the right inner edge has
insufficient evidence for a trace. The measurement is also tested on rotated
controls with an obscured interval, thin ink, glare and competing parallel
edges. Across twelve warmed repetitions on page 04, median original decode
took 193 ms and tracing took 1.7 ms with two OpenCV threads.

Through the ordinary upload and processing path, both existing page-04
proposals stayed in review. The whole-print and motif downloads and the
older motif alias all returned 404 before and after using both review
interfaces. A separate manual right-side correction with an entered size
can still yield a rectangular whole-print export despite unresolved
curvature. The measurement does not change that path. Frozen seeded
geometry passed 18/18 instances and 23/23 recoverable crops. The model-free
frozen detection and Golden replays retained their detection, geometry
and mode-availability metrics with zero newly held crops.

### Guard for a manual print rectangle that would cut a paper curve

For a manual correction of the same print, PhoSca checks an outward
curve in the original server-stored visible contour against the source
pixels. The saved proposal's corners anchor the measurement. Replacing
the submitted polygon or moving the editable corners a few pixels cannot
erase that evidence. A distinct nearby transition can indicate actual
paper outside a printed arc. The strongest source-supported portions
are shown as separate traces and both crop modes stay in review. Four
corner geometry still supplies the interactive perspective draft.

The 4-pixel control returned 404 for both crop downloads and the legacy
motif alias after two normal manual saves. Its previously generated ready
metadata also failed the read-side check until reprocessing. A straight
outline with extra points remained downloadable. The faint straight-paper
control with curved ink allowed both modes after correcting the actual
outer corners. Deliberately generated 3–5 pixel outward curves retained
the hold, including the 4-pixel cases with light blur and JPEG quality 75
and 85. This contour check requires a dense outline and source-supported
curve. An isolated trace uses the separate check below.

### Keeping a measured paper curve in review after a manual edit

Before this change, entering a measured aspect on Golden page 37 with
the original corrected corners exported a 142,367-byte whole-print
JPEG, despite the outstanding native curve. The manual correction now
retains the earlier server geometry and curve evidence. A measured
aspect or subpixel change alone leaves both modes and the older motif
alias in review. Older completed edits of an extended classical outline
can reconstruct the curve diagnosis from the original proposal and
source pixels. Their existing ready flags are checked before export.

A corrected straight alternative must move outside the disputed edge,
agree with four independently supported straight sides, preserve the
association with unaffected sides, and terminate at the print corners.
A long album rail and a straight mount around the curved print are
counterexamples. The actual paper boundary beside a curved printed
region passed the check with a five-level pale step. The check also
accepted that boundary after light blur and JPEG quality 75 or 85.
An additional native sweep accepted all 36 straight-paper alternatives
with dense curved ink outlines from 3.5 to 5 pixels and exterior
steps of three to five grayscale levels.
A weak straight-edge check also has to locate the transition at a
consistent depth along the side. In a separate sweep, all 180 tested
outward edits on 20 verified curved controls stayed in review. The
normal API also withheld a genuinely curved pale JPEG75 control after
its corners moved outward by 2.25 pixels.

The normal page upload and browser edit on page 37 returned 404 for
both downloads and the older motif alias after saving the aspect.
Desktop and mobile still show the measured segments. The frozen
geometric run passes all 23 recoverable crop checks. The automatic
mode-availability and accuracy metrics in Golden and in the frozen
classical benchmark match the preceding revision.
On page 37 the manual boundary check took a median of 0.26 ms with
the stored curve and 87.2 ms when reconstructed from the older
extended outline, over seven warmed calls at two OpenCV threads.

The Hegan album, The Rooms VA 103, has documented independent captures of
removed prints. The publication uses the images with permission and
offers no paired native masters. The Memorial University Grenfell album
PF-325 has measured individual images with reserved reproduction rights.
No matched whole-page rasters were located for those prints. Neither
album contributes verified reference cases at this stage.
[Hegan conservation study](https://digitalcommons.uri.edu/jfs/vol18/iss18/5/) ·
[Grenfell archive record](https://collections.mun.ca/digital/collection/m_grenfell/id/1023/)

### Manual correction after an uncertain capture edge

Golden page 04 reproduced an export error through the ordinary upload,
manual correction and reprocessing APIs. The right proposal extended to the
frame with unsupported paper edges. After entering 68 × 100 mm and saving
four plausible corners, the previous worker downloaded the unverified
rectangular whole print. The motif and the old motif alias were withheld.

The original capture-edge finding now survives geometry edits. The corrected
outline needs a source-verified side shared with the prior proposal, four
straight sides that end at the paper corners, and a coherent stock edge
against the surrounding backing. A nested pale paper layer or substantial
unexplained material in the old footprint holds the result for review.
Measured dimensions alone still establish only the aspect. A source with
indistinguishable stock, insufficient backing or curved paper can require a
new capture. For pre-fix rows with no saved reasons, explicit original-frame
contact in the saved rendered quad also requires this check.

Both modes and the legacy motif alias return 404 on page 04 after repeat
edits and reprocessing. Desktop and mobile show the outline guidance.
Mobile keeps the approval action disabled when the saved edit is held.
The old ready-row shape is checked at download time.
A controlled straight inset paper print with a nearby thin album rail
produces both JPEGs at the expected dimensions through the normal save and
download APIs. Further controls cover JPEG and perspective variation,
inboard ink, dark rails, lighter and darker mats, smooth backing gradients,
and faint extra photo content beyond the immediate paper edge. The frozen
seeded 18/18 instance checks and 23/23 recoverable crop checks pass. The
model-free Golden and frozen detection metrics for both modes are unchanged.
The native check took a median of 24 ms on the verified straight control and
under 0.1 ms on the rejected page-04 candidate over twelve warmed runs, with
two OpenCV threads, excluding source decoding.

The Phelps album at Missouri State Archives, RG 998.506, documents in-situ
scans and separately scanned rectos and versos for 68 removed prints. The
finding aid lists no access or publication restrictions and locates the
scans on archive media. No downloadable same-stock pair was recovered.
The public-domain Crowe album has documented 300 ppi capture and TIFF
masters, but the online requests returned 403 and the separately scanned
loose prints lack an unambiguous mounted match. These are provenance leads.
Neither contributes to the independent release groups.
[Phelps finding aid](https://www.sos.mo.gov/CMSImages/Archives/manuscripts/RG998_MS506.pdf) ·
[Crowe capture and rights](https://da.mdah.ms.gov/series/crowe/gentech)

### Uncounted print beside the pale leaf edge

The left print on Golden page 04 is lost during proposal construction.
Three Canny passes and the parallel-line search produce no closed
candidate. One contrast contour surrounds the print and part of the leaf.
Its 0.402 IoU and frame contact explain why the existing rejection is
appropriate. The conservative pale-leaf inset clips the candidate at the
left and bottom. Reducing the proposal inset to two pixels at 2048 × 1536
gives 0.981 IoU on the original and 0.996 on the prepared raster.
Feeding that contour into normal ranking changes existing survivors.
A painted shadow also passes the contour and local material tests.

After a crossing-outline finding, a separate search near the leaf edge
can show native measurements beside that outline. The adjacent region
must be staggered, unclaimed by other proposals, and supported by
measured transitions on at least three sides. On page 04, review shows
nine open segments, including over 1,000 pixels along the curved left
edge. The displayed span bows by more than four pixels. The gaps at
corners and competing transitions stay open. The count, predicted masks,
and download state use the existing detections.
The additional search took 248 ms median over eight warmed page-04
repetitions with two OpenCV threads, excluding image preparation.

Desktop review and mobile print-count review offer the measurements as
an overlay. Saving a matching manual outline retires the page hint in
the same transaction. A long, measured curved segment is retained on
that photo and keeps both crop modes in review. On page 04, a Haar face
proposal straddles the true print boundary. In this trace-backed review
path, the existing content-cutoff review reason preserves that caution
while allowing the editor to store the outline. Both crop downloads and
the legacy motif alias remain withheld after reprocessing.

### Synthetic stock detector experiment

The legacy checkpoint and registered ONNX could not be reconstructed.
This experiment starts from Ultralytics YOLO11n-seg v8.3.0 with published
SHA-256 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`.
The base is COCO-Seg pretrained under AGPL-3.0. The model and its ONNX
export are scratch artifacts. The production model registry is unchanged.
[Publisher checkpoint](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt) ·
[Publisher hash](https://huggingface.co/Ultralytics/YOLO11/blob/97c61428146e92d06e5cde0c70308ed8d36a6c34/yolo11n-seg.pt)

`scripts/prepare_detector_experiment.py` generates 96 train, 24 validation,
and 24 challenge pages. Each carries a synthetic source-group ID,
generation seed, split, image hash and label hash. Shared paper has one
label regardless of the number of image panels. Separately mounted prints
have separate labels. The generator uses no physical album pages. The
internal validation and challenge share layout families with training
and use separate seeds. The CSV records every artifact and the JSON pins
the run inputs, output hashes and paired evaluation metrics.
[Source inventory](detector-stock-experiment-sources.csv) ·
[Experiment receipt](detector-stock-experiment.json)

```bash
PYTHONPATH=backend python scripts/prepare_detector_experiment.py /tmp/phosca-stock-data
python scripts/train_segmentation_model.py --data /tmp/phosca-stock-data/data.yaml \
  --base-model /tmp/yolo11n-seg.pt --project /tmp/phosca-stock-run --name stock-512 \
  --device cpu --imgsz 512 --batch 8 --epochs 8 --export /tmp/phosca-stock-run/stock-512.onnx
```

The challenge has 12 one-stock multipanel cases and 34 separately
mounted prints. Model-only inference finds 12/12 and 32/34 at IoU 0.85
with zero false proposals. The hybrid path finds 11/12 and 32/34, with
21 false proposals. The classical fallback finds 7/12 and 4/34, also with
21 false proposals. On the true Golden pages, the hybrid retains 14/29
matches with 16 false proposals, versus 14/29 and 15 for the fallback.
On page 04 it matches none of the three prints at IoU 0.85. On the
Domoritz draft flatbed input, both hybrid and fallback match all four
prints. The hybrid finds 17/18 on the locked synthetic split and passes
6/11 whole-print and 7/12 motif checks. The frozen seeded geometry
control passes all 23. The experiment is rejected for deployment.
For a replay, supply `--model` and a scratch `--manifest` containing the
experiment ONNX digest and its 512 × 512 NCHW input to
`scripts/evaluate_onnx_segmentation.py`. Use `images/challenge` and
`labels/challenge` for the stock cases and `--synthetic locked` for the
frozen comparison. Run the model with and without
`--include-classical-fallback` and compare with `--classical-only`.

### Isolate invalid model proposals at the image boundary

An exported stock model reproduced a whole-page parse failure on challenge
seed `931260` at proposal threshold `0.025`. Prediction row `5134` extended
only 0.195 input pixels past the top letterbox padding. Undoing the padding
collapsed the proposal to a line of native pixels. Its corner-ordering
exception also discarded the valid proposals from the same inference.

The parser now omits collapsed proposals and reports their count in
`rejected_model_proposals`. A positive-area visible mask with unusable
perspective corners retains its contour for review. The upload regression uses
the reproduced prediction, verifies that the surviving proposal reaches both
crop modes, and exports both at their expected dimensions after confirmation.
The pipeline revision is `2026-11-precise-crops-v37` so validation applies to
the corrected behavior. Automatic crop release still requires verified
configuration-specific evidence.

### Keep an enclosed contact grid in count review

A 4080 × 3072 camera image includes one 35-frame paper contact sheet and a
separate loose negative-film sleeve. The fallback proposed the paper, its
enclosed image grid and the sleeve. The current detector keeps the paper and
shows the grid as a dashed, uncounted candidate. Both crop modes stay in
review until the user confirms the outer paper and the motif. The user can also
add the dashed contour as a separate print. A flush print on similar paper can
look identical in one exposure. One supported interpretation must never become
an automatic count confirmation.

The native check requires near-parallel edges in the outer print's plane,
continuous pale stock around the inner perimeter, and no distributed exterior
shadow that would support raised paper. It linked neither touching nor
staggered independent prints in the 40 paired controls. The independently
replayed helper took 1.71 ms median to test all three pairs in the camera scene.
On the real development image, false proposals fell from two to one at IoU
0.85. The remaining false proposal is the film sleeve. The paper mask IoU
stayed at 0.995 and its crop footprint IoU stayed at 0.9945 against the draft.
The published digital layout omits 132–147 native pixels at the nearest
physical paper corners, so the manifest has no verified crop-mode reference.

Save the pinned camera file as `runs/crop-quality/contact-sheet-camera-development.jpg`.
The images stay local. [Source capture and printing process](https://github.com/stevekm/contact-sheets/blob/d30662533467f0542e23ee4f8ed19484a54e282d/README.md#L313-L323)

```bash
python scripts/evaluate_onnx_segmentation.py --classical-only \
  --dataset-manifest data/evaluation/contact-sheet-camera-development.json \
  --visuals --output runs/crop-quality/contact-sheet-camera-development.json
```

### Recover detector provenance from the Git archive

The 28/29 Golden result in `46e94be:data/onnx_golden_segmentation_report.json`
records `e4168ae…` in every fixture. The 29/29 summary in
`4fcfec2:models/yolov8-seg-album.training.json` pins the later `4d823588…`
artifact. The warm model's original per-page report and registered bytes remain
unavailable in the inspected archive. Neither result counts as a current
independent crop benchmark. The [historical inventory](detector-registered-historical-evidence.json)
records 40 historical label hashes and 17 linked archived images. Physical
album IDs for the legacy training and validation pages remain unresolved.
The later model summary provides only aggregate counts for its additional
50 training, 11 validation and 34 held-out views.

### Compare real camera pages against a paired synthetic control

Git yielded five labelled camera originals from the old training split and
two from the validation split. Treat them as one legacy source group with
Golden until distinct physical album IDs can be verified. The mixed experiment
replaces 40 of 96 synthetic training images with rotated and gamma-adjusted
derivatives of the five training pages. Both arms share the same 24 synthetic
and two native validation pages. The stock challenge, frozen synthetic cases
and external draft scenes stay outside the training and checkpoint-selection
splits. The [source inventory](detector-transfer-experiment-sources.json) pins
the original file and label bytes, each derived image and label, and the
precise cohort memberships. Two independent rebuilds yielded identical
source inventories. Both datasets passed label and image validation.

```bash
PYTHONPATH=backend python scripts/prepare_detector_transfer_experiment.py \
  /tmp/phosca-stock-data /tmp/phosca-transfer-data
for cohort in synthetic-control mixed-legacy; do
  python scripts/train_segmentation_model.py \
    --data "/tmp/phosca-transfer-data/$cohort/data.yaml" \
    --base-model /tmp/phosca-stock-run/stock-512/weights/best.pt \
    --project /tmp/phosca-transfer-run --name "$cohort" \
    --device cpu --imgsz 512 --batch 8 --epochs 6 \
    --export "/tmp/phosca-transfer-run/$cohort.onnx"
done
```

All inference comparisons use a 512 × 512 input, proposal threshold 0.25,
acceptance threshold 0.70, and instance IoU threshold 0.85. Counts below
give matched prints followed by the number of false proposals.
The [comparison receipt](detector-transfer-experiment.json) records model
hashes, mask and crop measurements, review counts, and run times.

| Development or regression input | Synthetic model only | Mixed model only | Synthetic hybrid | Mixed hybrid |
| --- | ---: | ---: | ---: | ---: |
| Legacy Golden cameras | 2/29 + 6 | 28/29 + 1 | 14/29 + 20 | 29/29 + 2 |
| Frozen synthetic | 18/18 + 0 | 17/18 + 2 | 18/18 + 0 | 17/18 + 4 |
| Targeted stock challenge | 46/46 + 2 | 44/46 + 1 | 45/46 + 10 | 43/46 + 17 |
| Native legacy validation | 0/5 + 3 | 5/5 + 0 | 3/5 + 4 | 5/5 + 1 |
| Domoritz draft flatbed | 0/4 + 0 | 4/4 + 0 | 4/4 + 0 | 4/4 + 0 |
| Camera contact sheet | 0/1 + 0 | 0/1 + 4 | 1/1 + 1 | 1/1 + 4 |

The mixed model retains all 12 shared-stock cases in model-only inference.
It misses two separate neighboring prints and proposes one extra instance.
On Golden page 04 it detects all three prints. The hybrid adds two classical
false proposals on that page and recovers the rotated print on page 31.
The Golden whole-print footprints are draft measurements. The available
Golden annotations provide no verified motif geometry or aspect calibration.

On the frozen reference crops, the mixed hybrid passes 9/11 whole prints and
10/12 motifs. Both modes incorrectly report ready geometry for one obstructed
print with similar paper on both sides of a shared edge. The synthetic
control hybrid passes 6/11 and 7/12. The seeded geometry control continues
to pass all 23. The mixed model's improvement on the independent flatbed
applies to model-only detection. The classical fallback already matches
all four prints, with a better boundary score and fewer foreign samples
than the mixed hybrid. On the independent camera contact sheet, the mixed
hybrid adds three false proposals relative to the classical fallback.

Both trained candidates remain development artifacts. Automatic release
still requires a separately calibrated production configuration and verified
crop references in two independent physical album groups.

### Compare sparse and dense artwork on the same physical stock

On the known JPEG75 contact sheet the mixed model proposes a best
raw mask IoU of 0.822 at 512 pixels, below the 0.85 instance threshold.
Inference at 640, 768 and 1024 yields 0.351, 0.350 and 0.208. Masking
the neighboring negative sleeve at each input size yields 0.816,
0.350, 0.345 and 0.200. The lossless and JPEG55 derivatives show
the same direction. The paired experiment keeps the registered input
size and tests whether the training examples improve physical ownership
when the subject has many printed subdivisions.

Both 96-image training sets retain the same 40 archived camera
derivatives and 24 procedural images. The archived train originals
are pages 03, 23, 27, 30 and 38 in `legacy_album_context_01`. The
original image bytes, labels and derivative hashes are in the existing
transfer source inventory. The paired generated material provides
16 joined sheets and 16 two-sheet controls. The controls have matching
film artwork, physical gaps and paper rims. Each joined pair has the
same stock mask, camera, orientation, JPEG encoding, underlying art
and translucent sleeve. The sparse arm prints a 2 × 2 pattern and
the dense arm prints a 5 × 7 pattern. Rebuilding produces exactly
16 different training image hashes and byte-identical labels throughout.
All 34 validation image and label pairs match in the two arms, including
eight fresh layouts and the inherited validation set. The inherited
native validation pages share the legacy album group with training.

| Pinned trial input | SHA-256 |
| --- | --- |
| Mixed-legacy warm checkpoint | `8fa869a3fbafbac6217b8d99ed6e39f8a0c78fdd87d4128716d09851b80a0d86` |
| Mixed-legacy input inventory | `31b7db998453c763a867aeb97d018d981fe4804f273f1367d298a94aed8b79ae` |
| Sparse-control inventory | `9e1de68f4110f97014deb8afdee1081d2212b6b232edfdc58ad03156d3b75181` |
| Dense-contact inventory | `4b386a34dd1455185e8f2d4dcbb1e6f5ec4157e3b18dca345c6609d090123bff` |
| Independent procedural holdout generator | `a7a873fc578ce53be43914f03310b19497ee5c971c0e328feb2d46f0083066d5` |
| Independent procedural holdout manifest | `1119603f7b78fc73891772224755051954b5b1ef9a2de956c8f7c37e52276ac1` |

The independent procedural holdout is frozen before model scoring. It
contains 12 layout and artwork roots with 24 rasters and 36 exact
print ownership masks. Each root pairs one continuous dense print with
two closely spaced papers under the same camera, rotation and JPEG
settings. A separate negative-sleeve graphic is present in both views.
The images, polygon labels, ownership masks and manifest reproduce byte
for byte. Their exact hashes are disjoint from both training and
validation cohorts. The real contact derivatives stay together as one
known source outside this training experiment. The held-out groups are
procedural scenes and qualify for synthetic development evaluation.

```bash
PYTHONPATH=backend python scripts/prepare_contact_detector_trial.py \
  /tmp/phosca-transfer-data/mixed-legacy /tmp/phosca-contact-pair
python scripts/prepare_contact_detector_holdout.py /tmp/phosca-contact-holdout
for arm in sparse-control dense-contact; do
  python scripts/train_segmentation_model.py \
    --data "/tmp/phosca-contact-pair/$arm/data.yaml" \
    --base-model /tmp/phosca-transfer-run/mixed-legacy/weights/best.pt \
    --project /tmp/phosca-contact-pair/runs --name "$arm" \
    --device cpu --imgsz 512 --batch 8 --epochs 3 --patience 3 \
    --export "/tmp/phosca-contact-pair/models/$arm.onnx"
done
```

Register the exported filenames and SHA-256 values at input size 512 ×
512 using the same manifest format as the mixed-legacy transfer. Score
the detector-only and hybrid paths with the same manifest and thresholds.
The hybrid adds `--include-classical-fallback`. Use `--synthetic locked`
in place of the procedural manifest to run the two-mode crop regression.

```bash
PYTHONPATH=backend python scripts/evaluate_onnx_segmentation.py \
  --model /tmp/phosca-contact-pair/models/dense-contact.onnx \
  --manifest /tmp/phosca-contact-pair/registry.json \
  --dataset-manifest /tmp/phosca-contact-holdout/heldout.json \
  --proposal-threshold 0.25 --acceptance-threshold 0.70 --iou-threshold 0.85 \
  --output /tmp/phosca-contact-pair/dense-holdout-raw.json
```

Before scoring candidate weights, the paired experiment requires at
least three more raw matches among the twelve joined sheets at 0.85
IoU. Separate controls must retain their two proposals without more
undercounted pages or extra proposals. The known JPEG75 contact sheet
must match at the same threshold. Frozen and known real development
replays must retain their prior detection and crop safety results. The
proposal and acceptance thresholds stay at 0.25 and 0.70. Candidate
weights also need independent physical album evidence and separate
calibration before any deployment decision.

The training used Ultralytics 8.3.0, PyTorch 2.7.1 CPU, OpenCV 4.10.0
and NumPy 2.2.6. The three epochs took 61 seconds for the sparse arm
and 79 seconds for the dense arm on four CPU threads. The shared
validation mask mAP50–95 at the final epoch was 0.9034 and 0.8965.
The ONNX SHA-256 values tie the evaluation to the exported artifacts.

| ONNX development artifact | SHA-256 |
| --- | --- |
| Mixed-legacy parent | `1e95a1a90ff897df7099ff80675186ec1d8145f7bb07559fe918b3dd3a6db748` |
| Sparse-control | `58cd17651af74997a98d09e27f8c427f0924f5501f2ea4f9d112599fa9a1e595` |
| Dense-contact | `fbfbe014b5cbb69183f746aecae9a03ae2f8d2e0d4061bf45d50a4a89e8369b3` |

| Procedural holdout at IoU ≥ 0.85 | Parent | Sparse | Dense |
| --- | ---: | ---: | ---: |
| Raw joined matches / 12 | 3 | 8 | 11 |
| Raw separate matches / 24 | 5 | 18 | 21 |
| Raw joined extra proposals | 37 | 40 | 35 |
| Raw separate extra proposals | 36 | 29 | 22 |
| Raw precision / recall | 9.9% / 22.2% | 27.4% / 72.2% | 36.0% / 88.9% |
| Hybrid final joined matches / 12 | 2 | 3 | 6 |
| Hybrid final separate matches / 24 | 6 | 18 | 21 |
| Hybrid final extra proposals | 86 | 86 | 68 |

The denser artwork added three whole-sheet matches. It removed twelve
inside-paper fragments compared with sparse across both fixture types.
The loose sleeve still received 30 raw proposals versus 31 for sparse.
On separate page `1449600-separate`, sparse found both sheets. Dense
found one and added a combined proposal covering 97.5% and 98.3% of the
two respective ownership masks at detector confidence 0.896. The dense
arm lost a previously correct nearby-paper case despite the aggregate
gain. On joined page `1449610-joined`, dense still proposed four internal
fragments. The per-case audit compares the raw detector polygon. The
parent's source-derived visible polygon on `1449611-separate` crosses
0.85 IoU after detection, accounting for its one additional match in
the top-level evaluator count for the detector-only run. A second pass
scores the pre-fusion proposals against the native PNG ownership masks.
For all 72 model-page combinations the 0.85 detection classification
agrees with the standard evaluator's polygon-based result.

On the real JPEG75 sheet, sparse has a single proposal and reaches
0.8930 mask IoU in both runs. Dense reaches 0.8937 with three extras
before fusion. A weak proposal triggers the classical fallback. The
classical inner outline overlaps the better whole-sheet proposal at
0.9044 IoU and is ranked higher by the fused score, 0.6985 versus
0.6455. The hybrid falls to 0.8317 with four counted extras. Against
the development outline the sparse hybrid's whole-print preview
excludes 8.76% of the footprint and includes 2,268 output samples with
foreign source contributions, with 1,661 beyond the one-pixel band.
The two modes remain in Review for every proposal in all three JPEG75
replays. An independently verified native motif reference remains open
for this source.

| Other hybrid replays, correct / expected, extra | Parent | Sparse | Dense |
| --- | ---: | ---: | ---: |
| Same physical contact sheet, JPEG55 | 1/1, 3 | 1/1, 3 | 0/1, 3 |
| Same physical contact sheet, lossless PNG | 0/1, 6 | 0/1, 5 | 0/1, 4 |
| Golden camera development pages | 29/29, 0 | 26/29, 11 | 29/29, 8 |
| NerdScan clipped-print development variants | 3/3, 2 | 3/3, 2 | 3/3, 2 |
| Domoritz flatbed development page | 4/4, 0 | 4/4, 0 | 4/4, 0 |
| Earlier synthetic stock challenge | 43/46, 17 | 46/46, 13 | 46/46, 19 |

All Golden proposals still require Review in both modes. The total per
mode grows from 29 to 37 under each trained arm. The Golden pages share
the physical legacy source group with part of the training material.
The three encodings of the contact sheet represent one physical sheet.

| Frozen synthetic hybrid | Parent | Sparse | Dense |
| --- | ---: | ---: | ---: |
| Correct detections / 18 | 17 | 18 | 18 |
| Extra proposals | 4 | 1 | 0 |
| Passing recoverable `full_print` / 11 | 9 | 5 | 6 |
| Passing recoverable `image_only` / 12 | 10 | 6 | 7 |
| Obstructed crops marked ready, per mode | 0 | 1 | 1 |
| Median from image to JPEG in ms | 675 | 659 | 638 |

The ordinary multipart upload and processing path was replayed for all
three registered models in isolated local stores at the default detection
thresholds, with automatic photo rotation disabled for the comparison.
The JPEG75 page has 5, 1 and 4 counted proposals respectively, all held
for Review. Each proposal allows the untrimmed preview. Before correction
the named paper and motif downloads and the legacy download return 404.
After rejecting the extra proposals and saving the measured test paper
outline and 203 × 264 mm dimensions, only the named paper export opens.
Saving the motif outline then opens the motif and legacy downloads with
the same bytes and leaves the paper JPEG byte-identical. Both named
JPEGs match across all three models. The corrected motif samples all
35 image fields, 315 checked landmarks and 1,938 reference text pixels.
These manual exports use the same known development measurements.

Both trained arms mark the obstructed `touching_same_color` fixture
ready in both modes. Their local geometric statuses precede the
separate release policy, which stays locked. Cases such as `portrait`
and `camera_tilt` also enter Review with ambiguous paper borders under
both trained arms. Both exported weights remain development artifacts.
The unchanged detector control still finds 18/18 prints. The separate run
with seeded geometry still passes 11/11 paper crops and 12/12 motif
crops, without marking an obstructed reference ready.

#### Isolate the fusion choice on the native JPEG75 image

An ablation switches only the selected candidate in the existing
dense-contact cluster. The mask IoU rises from 0.8321 to 0.8937, and the
whole-print review-preview IoU rises from 0.8321 to 0.9021 against the
development outline. The two modes still require Review. Measuring the
paper perimeter on the original yields a quad within one native pixel
of the development corners when seeded with the model proposal and the
active neighbours. Relative to this perimeter, the model polygon gains
34,079 paper pixels that the classical polygon omits. It also loses
6,090 paper pixels that the classical polygon includes, mainly at the
lower rim. The perimeter supplies additional information for the
reviewer beyond either individual proposal.

Across the 26 hybrid clusters that selected a classical alternative in
the three-model replay, nine met the exploratory perimeter probe's
pairing and size conditions. The probe measured seven perimeters in
55–63 ms each on one CPU thread. All seven came from different runs of
the same physical contact sheet. It found no new perimeter on the
frozen synthetic, Golden, NerdScan or Domoritz inputs in this replay.
The dense model quad fits its measured physical perimeter better than
the classical quad in all three encodings. The parent model quad fits
the same perimeter worse than the classical quad in all three. A rule
that favours the model family would miss that difference.

With the same seeds and positions, the perimeter search also returns a
quad on 6 of 12 physically separated controls and 16 of 18 controls with
a second paper layer. The smallest gaps can disappear during sampling or
compression. Both candidate polygons and the exterior measurement can
be consistent with separate ownership. A separate seam test needs two
paper flanks, local backing colour in the gap and a continuous path
ending at independently measured sides. In this image local backing and
paper have similar lightness and distinct chroma. Keep the candidates in
Review while that ownership is unresolved. Sleeve proposals merit a
separate negative example experiment because the paired training barely
affected them.

#### Check whether a chroma line identifies a physical gutter

The native-resolution prototype samples both paper flanks and the colour
of the local backing. It registers the path endpoints at observed paper
edges and checks the entire path across the source image. The frozen
prototype has SHA-256
`0bb3efe0c0e78947a34f7a328d10ec59fc1c7d91fa2c115c2d9d47d3d003fe0d`.

On the 70 controlled JPEG75-contact-sheet variants, the native rerendering
of gaps from 0.75 to 4 pixels finds all 70. A further JPEG70 recompression
reduces this to 40, including 11 of 28 horizontal gaps. Fourteen untouched
and overlaid-paper negatives give zero marks. On a separate clipped-scan
control set the existing dark-gutter test finds all 16 complete physical
gaps and gives zero marks in 20 non-through controls. The chroma
prototype gives zero marks in that set.

An independently authored blind set freezes 35 source layouts and 280
variants before the prototype runs. Localized hits need at most 4 native
pixels of p90 cross-track error and 13 native pixels of endpoint error.
The fixture manifest has SHA-256
`bc4900107c84c887ee3596dd952a9ed02256660df18aa369a84934a7663233fd`.

| Blind development case | Localized marks |
| --- | ---: |
| Visible physical gaps | 65/136 |
| Printed lines on continuous paper | 68/96 |
| Printed lines after excluding raster-identical pairs | 56/80 |
| Overlap or subpixel ambiguity | 6/40 |
| Untouched continuous print | 0/8 |

The marked lines track real raster structures. Some printed controls
have the exact pixels of a physical gap, and the prototype marks both
identically. Distinct-artwork neighbors get 0/16 localized marks and
stepped ends get 2/24. The blind median is 169 ms and p95 is 257 ms on
one CPU thread. The results support rejecting colour-only evidence as
a physical-gutter classifier or a new user-visible review signal.

The four native gaps on the pinned Domoritz flatbed are a separate
development stratum. The original and draft-edge hashes match the
manifest. Their median widths are 18.5–24.1 pixels. Sampling the gap
centre and both print flanks shows median minimum chroma differences of
3.2–7.2 encoded Lab levels and lightness differences of 70–173 levels.
These prints have dark or coloured image content along their edges.
The controlled contact sheet supplies a different material signal.
Independent side evidence or a continuous, observable backing texture
is necessary to strengthen the ownership claim.

#### Compare paper-edge previews beside a neighboring proposal

A separate scratch trial holds the detector fixed. It uses locally
consistent paper along each of the four sides, including material
transitions before and beyond the fitted corners. This lets the backing
colour differ between sides. The adjacent proposal must be nearby and
the candidate warp must avoid its polygon. Count and both Review statuses
stay the same. On the frozen `touching` pair the new preview produces the
following measurements against the known native reference.

| Whole-print review preview | First print | Second print |
| --- | ---: | ---: |
| Maximum corner error, px | 5.47 → 0.62 | 3.68 → 0.52 |
| Footprint IoU | 0.9766 → 0.9984 | 0.9795 → 1.0000 |
| Omitted pixels beyond one native pixel | 1,320 → 0 | 674 → 0 |
| Landmark p95, px | 4.51 → 0.52 | 2.84 → 0.52 |

Independent controls show why this is insufficient for the review warp.
A full-bleed printed colour field yields its own tidily terminated
rectangle. With a true neighbor on the opposite side, the trial moves the
preview edge from x579 to x574 while the physical sheet extends to x603.
Visible loss increases from 6.43% to 7.77%. The same false choice also
occurs in 12 of 16 inputs with an actual overlapping rear print on the
*same* disputed side, across the native, rotated and skewed replays.
The production renderer keeps the existing preview. A subsequent trial
must preserve every previously visible sample and test foreign samples
at the expanded edge independently of the nearby proposal.

On the sigma 1.2 Domoritz derivative, the lower-right corner of the
current review preview is 9.48 native pixels outside a source-fitted
right paper edge. A scratch side fit reduced that to 0.84 pixels or
less. In a shadow render, footprint IoU rose from 0.9755 to 0.9874
against the native line model and foreign samples beyond one pixel
fell from 1,486 to zero. Omitted pixels beyond one pixel rose from 14
to 16. A same-colour physical paper extension can produce the same
local line. Both modes remain in review and the production preview is
unchanged.

Even a containment gate with a 0.05-pixel tolerance leaves an ownership
problem. A fresh 384-input control set gives 24 false expansions onto
album backing, a transparent carrier or a separately printed strip. All
24 have a native one-pixel gap. Measured on the separate-strip preview,
foreign sample centres increase from 1,932 to 4,830 with the previous
preview corners preserved. Both modes stay in Review.

#### Show competing edges in Review

For an already-held whole-print crop, retain the two strong alternatives
from the native line-fit evidence. Both need support at 88% of stations
and at least 70% in every quarter of the edge. The marks stop at observed
stations. Display them as possible edges in the original image. The
print and motif warps, previews, release statuses and paper counts keep
the previous values. Existing explicit outline, capture-edge and other
measured trace hints take precedence, as do film and paper grid ownership
groups. The code also skips a side that an existing stock fit has resolved
in the preview.

The locked `touching` control shows a pair of alternatives at source
rows 413.8 and 425.5. The provisional lower warp edge stays at row 420
and the motif still requires review. The reviewer can compare the
printed image edge and the actual outer paper edge on the full page.

The normal multipart replay with the mixed-legacy development model
keeps two counted photos. The original-coordinate alternatives appear in
the full-print editor on desktop and mobile. Both named downloads and
the legacy trimmed path return 404, with the untrimmed preview available.
Paired renders before and after the change produce identical preview
bytes and identical crop geometry for both photos. On mobile the full
approval button points to the paper-edge hint and stays disabled during
this geometry review state. An explicit mobile edit to the four known
paper corners with the 166 × 114 mm test measurement removes the traces
and opens both named downloads. The active count and the neighboring
photo's mask, modes and JPEG bytes reproduce.

The locked detection control retains 18/18 matches, 6/11 whole-print
passes and 7/12 motif passes. The independent seeded replay retains
11/11 and 12/12. Contact JPEG75, flatbed and clipped NerdScan keep the
same per-instance crop metrics and availability in both modes. Across
100 alternating CPU measurements of three ambiguous line fits, retaining
the native traces adds a median 0.06 ms. The JPEG75 and lossless PNG
contact-sheet group hints retain precedence, including the competing
negative-sleeve and paper-grid proposals. All encoded crops and previews
of those 11 rows are byte-identical to the trace-disabled reference.

### Show the visible spans of an interrupted paper edge

On the sigma 1.2 blurred Domoritz flatbed derivative, the bottom-right
print has one unsupported right side. The current preview's lower-right
corner is about 9.48 native pixels outside the source-fitted line. An
opposite-side direction opens a narrow native search window. Three
search radii must agree, nearby print polygons must clear both the
current and measured shape, and the disputed band must match the pale
backing. Both adjacent paper-to-backing transitions must end where the
new line meets their corners. The hint draws source spans with a localized
step and leaves a gap over the low-contrast sky. The overlay is provisional
because a paper extension matching the scanner backing exactly still
produces the same evidence.

The two spans reproduced through all four quarter turns with at most
0.54 native pixels of cross-track disagreement from the independent
line fit. Lifted pale stock, an outward ink extension and a neighboring
print claiming the band suppress the hint. JPEG75 recompression of the
blurred target can remove enough transition support to suppress it.
The source check took 24.91 ms warm median and 46.10 ms at p95 in 40
direct one-thread trials. Both modes stay in review. The warp, aspect,
sampling, preview pixels and active print count match the baseline.
The normal upload and review queue expose the same two spans. The
untrimmed preview JPEG is byte-identical. Both named exports and the
legacy trimmed alias return HTTP 404 and premature approval returns
HTTP 409. Mobile disables the completion action on this geometry task
and associates the two-span hint with that action. Desktop keeps the
editor available for a deliberate correction.

The shadow comparison produced exactly one new hint across 97 crops
from the frozen, Golden, Domoritz, NerdScan, contact and ScanCropper
controls. The final synthetic-control replay retains 18/18 detector
matches, zero extras, 6/11 full and 7/12 motif passes. The seeded replay
retains 11/11 and 12/12. Final Golden retains 29/29 matches, and the 12
blurred Domoritz prints retain the exact baseline metrics in both modes.
The draft-only measures and scratch re-render comparison are recorded
with the [near-neighbor edge trials](#compare-paper-edge-previews-beside-a-neighboring-proposal).

### Compare a large mounted print with album-container proposals

The Willett Album development set pins six native album-page JPEGs
from `f02315fbb8ab62f491c1003ce06614daf30cf617`. Its eight approximate
visible rectangles were traced before running the detector. Page 98
is a newspaper control and the drawing on page 109 is excluded from
the photo count. On page 100 three physically separate prints share
the dark album backing. This entire physical album is reserved for
development. The catalogue's educational and research use statement
requires an independent rights check for training, calibration and
product release. The references are source-only drafts for instance
counting.

```bash
git clone --no-checkout https://github.com/historycenter-mld/willett_album.git /tmp/willett-source
mkdir -p runs/crop-quality/willett-mounted-development
for page in 6 10 20 98 100 109; do
  git -C /tmp/willett-source show "f02315fbb8ab62f491c1003ce06614daf30cf617:objects/col003_p$page.jpg" \
    > "runs/crop-quality/willett-mounted-development/col003_p$page.jpg"
done
```

Set `--dataset-manifest data/evaluation/willett-mounted-development.json`
and `--include-classical-fallback` in the registered mixed-legacy replay.
For the 23-megapixel files, running the mounted-page replays sequentially
avoids concurrent high native-memory peaks. The manifest pins both the
page bytes and the label files.

With the pinned mixed-legacy development checkpoint at IoU 0.85,
the hybrid count before the trial yields 7 of 8 matches and 10 extra
active proposals. Page 6 is an actual single mounted print with dark
corner holders. Its complete model mask scores 0.948 IoU against the
coarse reference and takes 40.49% of the page. Cross-source fusion
retains it. The later broad-container rule discards it because several
high-confidence fragments are inside. The hybrid then counts six
false proposals and misses the complete print. Detector-only gives
one correct match and one extra proposal on the same page.

Native line fits have support of 72.3%, 87.5% and 70.5% on three
exterior sides. The fourth stays unsupported, and the mount obscures
the corners. Every nested candidate has at least two inner edges
across which several distinct source scene features continue at
near-native sampling. Physical ownership still needs review. A
nearly registered second print and separate dark-backed sheets with
displaced slanted gaps can preserve the same local signal. A strict
whole-interior step veto blocked 2 of 16 blind native physical split
layouts and one of two continuous native artwork controls.

The initial reassignment trial reduced page 6 from six false proposals
to one correct complete print and one false proposal, with five interior
alternatives in count review. A second exploratory veto tracks dark
backing in any direction across the entire print. It recovered nine of
the fourteen missed native splits and still grouped five split layouts
incorrectly. The 14-pixel uninterrupted tilted control recovered both
natively and after JPEG92. Nearby tilted splits around two to three
pixels remain hard. The working hypothesis also remains possible with
a nearly aligned print of the same scene laid on top.

The implementation keeps the active instance set and offers exactly one
optional complete outline in count review when a strong interior model
container and several distinct fragments meet the source checks. Both
clients show the original visible detector polygon. The desktop editor
loads the four editable corners from the native fits and one provisional
side. Saving a new print and approving its crops still require explicit
actions. Other pending alternatives keep their IDs and stored status.

In the paired replay on page 6, all six active proposals and their crop
metrics and availability are identical for both modes. The complete
print remains an automatic miss, with a new 12-point count-review
outline at 0.948 draft mask IoU and a separate four-corner starting
shape at 0.980 draft IoU. A normal multipart upload reproduced the hint
and the six open geometry tasks. Editing and cancelling the desktop
draft made zero writes. Mobile displayed the same 12 points, rejected
a confirmation without the proposal ID with HTTP 409, and accepted an
explicit count acknowledgement without saving any draft geometry. Both
named exports and the legacy trimmed export returned HTTP 404 for all
six entries. The untrimmed inspection images returned HTTP 200.

The exact implementation still finds all three separate prints on page
100 at IoU 0.85. One nested false proposal remains and no new count hint
appears there. On page 6, the original-pixel check for five fragments
took a warm median of 322 ms over six runs on one OpenCV thread. The
single paired page replays took 11.2 and 11.8 seconds in segmentation
and 44.7 and 50.6 seconds from source to JPEG. Those totals are individual
development runs and include other stages and system load.

The remaining page 100 proposal is a six-vertex model mask inside a
classical whole-print outline. The detector scores the child at 0.949
with boundary evidence of 0.296. Its right side lands about 120 native
pixels out in plain stock beyond the ink. Three remeasured ink sides
have median optical widths of 4.25–4.50 pixels. The outer paper sides
measure 2.75–3.25 pixels in the same scan. These measures recovered an
approximate inner outline in one scratch comparison. A rectification
phase shift of less than half a native pixel moves its projected tilt
from 0.01980 to 0.02027 across the existing 0.020 cutoff. Two separate
near-identical print controls pass the material checks and only fail
that cutoff at 0.02001–0.02003. A looser threshold would admit them.

An independently frozen 160-view set covers 40 generated paper
layouts, with 80 layered views and 80 flat ones. Six pairs have
pixel-identical sources with different paper counts. The existing
four-corner shared-stock route keeps all 160 inner contours as review
suggestions outside the active count. The new remeasurement prototype
abstained on all 160 inputs and also on the native page 100 replay at
each quarter turn. The page 100 count and existing proposals are unchanged.

The frozen hybrid retains 18/18 detection matches and zero extra
proposals, with 6/11 whole-print and 7/12 motif passes. Supplying the
reference contours retains 11/11 and 12/12 respectively. Golden retains
29/29 matched prints, zero extra proposals and its existing crop
results. These are regression checks. Separate production calibration
and verified independent reference groups are still required.
[Willett archive](https://historycenter.mld.org/willett_album/) ·
[Pinned source tree](https://github.com/historycenter-mld/willett_album/tree/f02315fbb8ab62f491c1003ce06614daf30cf617)

#### Diagnose the newspaper and drawing proposals

On page 98, model-only and hybrid inference both retain the same 0.803
confidence proposal. It covers a printed newspaper illustration and its
caption. A native Fourier check finds the 9.05-pixel diagonal print
screen in all 42 sampled regions. A separate OCR diagnostic finds 23
regular lines of body text beside the proposal and five caption lines
inside it. The pipeline applies OCR later and the source has no OCR
sidecar. A separately mounted facsimile of the illustration preserves
the same screen and caption context, so those cues can guide review.

Page 109 retains both true prints and two proposals for the folded
butterfly drawing. Both detectors propose pieces of the drawing before
fusion. Its two fold flanks differ by a median 10.55 Lab units and the
actual cardstock-to-backing edge differs by 47.97. A real photo on the
same page has a 7.28-unit fold. The source cues tested here leave the
ownership and export decisions in place.

### Measure an open edge at an overextended corner

The classical and mixed hybrid proposals on Golden page 31 include an extra
triangle of album paper at the lower-right print. Three corners are within
7 native pixels of the draft outline and one is about 115 pixels away.
The current review can show a line on the measured lower paper edge. A
geometric estimate from three corners locates a search window. Only the
edge supported by original pixels appears in the editor. Both modes stay
in review and retain their current preview pixels and perspective.

The trace appears only on this print across all ten Golden fallback pages.
It persists in all four quarter turns and after a 75% resize with JPEG75.
Blank paper, an unsupported side with a correct seed, and interrupted lower
edges stay without a trace. The positive native check takes about 15–17 ms
with one OpenCV thread. The regression verifies the open line and the unchanged
rectified raster, mode geometry and review statuses. The trace is available
when the face guard also flags possible content cutoff. That warning remains
in effect for both modes.

### Keep an unassigned paper rim in review

The mixed checkpoint on the frozen `touching_same_color` input fitted one
side to an ink seam. Its proposed whole-print crop lost 3.59% of the reference
content and had a maximum corner error of 9.07 native pixels. The linked
motif lost 0.93%. Both previously reported ready geometry despite the
obstructed reference. The mixed checkpoint is still a development artifact.

The new check samples the original outside that seam. It requires pale stock
of the same colour on two other physical sides and an unbroken strip reaching
the other print. A visible gap or a change of paper colour interrupts this
evidence. With uncertain ownership, the review shows both previews with the
same warp and pixels and asks the user to check the shared paper edge.

Each mixed inference mode routes that one additional obstructed instance to
review in both crop modes. The synthetic seeded, classical frozen and
classical Golden measurements remain unchanged. At native resolution the
shared-rim check took 0.53 ms median and 0.66 ms p95 over 100 warm samples
for the positive case. Print-boundary revision 10 records the changed gate.

### Review unsupported fallback outlines before counting a print

On Golden page 04 the mixed development model proposes all three physical
prints. The classical fallback adds an outline from inside the right-hand
print to the frame and a second outline spanning all three prints. Existing
cross-source matching leaves both separate because each extends well beyond
the corresponding model masks.

The precise hybrid path checks the original pixels where a classical-only
outline largely overlaps source-supported model masks. It looks for an
unsupported side where pale material continues across the proposed edge.
A frame-corner proposal needs its sole supported edge to follow the modeled
print's edge. A proposal spanning multiple prints also needs to lack its own
image structure in the exposed area. Competing measured edges or independent
content keep the usual candidate path. Possible extra prints remain dashed
outlines in count review. A reviewer can add an outline with corrected
corners or confirm the current count after checking the outlines.

With the same registered mixed development checkpoint and fixed thresholds,
Golden retains 29/29 matches and moves both false proposals to uncounted
review. All three page 04 detections retain their crop footprints and both
modes remain in review. On the draft full-print outlines, the footprint IoU
ranges from 0.966 to 0.990, content cutoff from 0.51% to 1.22%, and retained
foreign fraction from 0.60% to 2.53%. The native check takes about 0.25 s
for both extra outlines together on the development CPU. The mixed hybrid
retains its existing instance and crop results on frozen synthetic (17/18,
four false proposals), the targeted stock challenge (43/46, 17), Domoritz
(4/4, zero) and the camera contact sheet (1/1, four). The old checkpoint
comparison remains a development result.

Thirty nearby, overlapping, clipped and multipanel controls remained counted
across native, projective JPEG78 and resized JPEG78 views. Six further
controls use the exact false-positive coordinates with actual additional
print boundaries rendered into the source. All six remain counted. The
native page 04 case works at all four quarter turns. The ownership check
abstains at 75% and 50% in the JPEG75 replay with the original proposals
transformed into those views. A separate end-to-end 75% JPEG78 replay finds
the three prints and never invokes the fallback. Both export modes remain
held in the normal upload replay with the current validation configuration.

### Recover historical album scans for intake

The Romanov costume ball souvenir edition provides four full physical leaf
scans from its public Git history. Three pair with later 600 dpi colour
scans with held-out landmark p95 errors of 3.16–3.58 native pixels. The
leaf and scan hashes, accession context and source URLs are recorded in
the [intake receipt](romanov-album-intake.json).

Keep these leaves in one tentative physical group. The reproduction field
may be printed on the page, and the rights for the full native scans and
individual crop boundaries still need confirmation. Intake records stay
outside training, calibration and release evaluation.

### Review partial grids inside a contact print

The native [camera contact sheet](contact-sheet-camera-development.json)
contains one paper print with 35 frames beside a negative-film sleeve. The
registered mixed development model and classical fallback originally counted
one lower section of the print as another print. The enclosing paper contour
had strong boundary evidence, and the inner model outline had weak boundary
evidence along its proposed cut.

The precise path now checks for repeated perforation rows with matching pitch
and spacing across that cut. It samples the original pixels at both paper
margins and looks for a separate rim or cast shadow. With a unique
parent-to-fragment match, the enclosing print keeps its original outline and
the fragment becomes a dashed count-review suggestion. Reviewers can add
the fragment as a separate print if they find a physical edge on the source.
Both crop modes remain in review. Multipanel revision 5 records this change
in the validation and calibration profiles.

With the same mixed checkpoint and the thresholds above, the 4080 × 3072
camera image retains its one match. Counted false proposals go from four to
three, raising precision from 20% to 25% at 100% recall on this image. The
retained print has the same 0.9947 instance mask IoU and the same whole-print
warp. Against the draft paper reference, that warp has footprint IoU 0.9945,
corner distances of 2.0–7.5 native pixels, 0.46% content cutoff and 0.12%
foreign fraction. The 6,491 foreign source samples are unchanged. Verified
aspect, landmarks and independent motif boundaries are still pending for
this source. The normal upload serves the rectified whole print for review.
The motif has no preview and both automatic exports are held. Adding the
dashed outline manually is idempotent.

The source check took about 53 ms warmed on one CPU thread. It reproduced
the review suggestion for the same contours through all four quarter turns
and all five tested scale and JPEG settings. The full detector reproduced
the suggestion on three of the five unrotated inputs. At 75%/JPEG75 the
detector's paper outline lost part of the right side, and at 33%/JPEG55 it
selected a weaker contrast outline. Quarter turns led to different model
proposals. The three remaining native false proposals belong to the sleeve.
Shared artwork, inverted tones, warmth and glare also occur in controls
representing glossy negative prints, so those cues cannot establish the
physical carrier.

Independent controls retained all 48 touching or overlapping comparisons,
six separate rims and 60 clearly raised-layer variants. Generated tests
cover projective JPEG reconstruction, weakly supported inner proposals,
visible rims, directional shadows and interrupted grid spacing. Golden
(29/29 matches, zero false proposals), Domoritz (4/4, zero), the frozen
mixed hybrid (17/18, four) and the generated stock challenge (43/46, 17)
reproduced their previous instance and crop metrics. The frozen seeded
geometry control retains 11/11 whole-print and 12/12 motif passes, with zero
obstructed cases marked ready. No model weights were promoted.

### Identify mounted prints in the Russell railroad album

The Library of Congress catalog cross-references thirteen printed pages in the
bound Russell album with item records and measured print-stock dimensions.
The [intake receipt](loc-russell-album-intake.json) records the whole-volume
accession and publication advisory, the page-to-item mapping, and an item
TIFF. The acquisition vendor's catalogue confirms that the paper prints in
this volume are mounted on heavy stock. The page 3 record points to a public
domain negative that can corroborate interior landmarks after matching.
The managed network returned HTTP 403 for the official full-leaf manifest
and the item download. Original rasters, separate capture and both crop
references remain pending.

### Preserve the paper contour when line fitting follows printed detail

At 75% resolution the contact-sheet Canny contour encloses the complete
paper. Line fitting previously moved its right side 94–114 source pixels
inward onto the printed grid. The prepared edge detector favored the
strong internal contrast even though the exterior of that line was still
the same paper as the outer rim.

The precise classical path now compares a large inward displacement with
the original pixels. It checks the pale stock strip for a second material
transition, verifies that the outer step stops at both corners, and checks
where the adjoining sides meet the outer contour. When those checks pass,
the raw contour supplies the whole-print review warp. The existing paper
edge confirmation holds both crop modes. A retained contour may own an
uncounted raster suggestion, and an already held proposal remains counted
if it is itself the smaller contour. The calibration and validation profiles
record classical contour refinement revision 1.

The development replay decodes the pinned camera original, resizes to
3060 × 2304 with OpenCV `INTER_AREA`, and encodes once at JPEG quality 75
(SHA-256 `ae4a57a70f0ad426bd5c66bcaad9b1df438889bb59c1b222e850acafc2df2862`).
The normalized draft paper reference is the one in the camera manifest.
With the fixed mixed checkpoint, one print remains matched and counted
false proposals drop from four to three. Against the draft paper footprint,
instance mask IoU improves from 0.9160 to 0.9936 and whole-print crop IoU
from 0.9085 to 0.9931. Crop cutoff falls from 9.15% to 0.57%, and the worst
corner distance from 114.8 to 8.6 pixels in the tested source raster.
Foreign fraction rises from about 0.001% to 0.15% (50 to 4,931 foreign source
samples) as the wider warp reaches the draft boundary. The source has slight
edge curvature and still lacks verified aspect and motif references.

The guard took 1.57 ms median and 2.02 ms p95 over 100 warm runs on one
OpenCV thread. Nearby paper, thin separating gaps, a rail extending past
the corners and a white-on-white source stayed outside the retention path.
A broad pale printed rectangle can imitate mounted stock. The confirmation
hold covers that ambiguity. The normal upload serves the wider rectified
whole print for review, exposes the reversible grid suggestion and holds
both exports. The motif still needs a separate correction. The 33% capture
remains a detector problem because it lacks the outer contour before line
fitting.

Golden, Domoritz, the frozen mixed hybrid and the generated stock
challenge reproduced their previous instance and crop metrics. The frozen
synthetic control hybrid retains 18/18 instance matches with no false
proposals. The seeded geometry control retains 11/11 whole-print and
12/12 motif passes, with zero obstructed cases marked ready.

### Suggest an outer outline for the low-resolution camera review

The 33% contact-sheet capture loses the physical paper outline during
initial detection. When both modes already need geometry review and a
classical contrast proposal has an unresolved boundary, a local source
search compares two chroma thresholds around the same pale stock. The
result must enclose the original proposal closely, agree across both
thresholds, and have a visible four-sided perimeter. Native samples near
the ends check narrow gaps that morphological closing can hide. The search
abstains when it would newly surround or cross a competing detection.

The alternate quadrilateral appears dashed in the whole-print editor on
desktop and mobile. It can be loaded into the existing photo's draft.
The original preview pixels, effective warp, mask ownership, and detection
count remain in place until the reviewer saves. Both modes keep their
previous review reasons and exports stay held. An unchanged outer quad
requires explicit whole-print confirmation. A motif-only or rotation-only
edit cannot resolve the pending outline review. A very thin flush join or a
printed shape can mimic continuous stock, so the alternative requires the
reviewer's source comparison.

The camera replay decodes the pinned original, scales both dimensions by
0.33 with OpenCV `INTER_AREA` to 1346 × 1014, and encodes once at JPEG55
(SHA-256 `e31141f16b3c6abd0a3595cce0f9a229752cbd29684af4bc92e693efa1ea3f83`).
Against the draft paper footprint, the existing whole-print preview has
IoU 0.8377 and maximum corner distance 60.4 native pixels. The suggested
outline has IoU 0.9847 and maximum corner distance 4.9 pixels. The right-hand
paper also receives an outline on JPEG75 and the lossless resized input.
On all three variants, the sleeve receives no suggestion and the existing
previews keep their pixels and homographies. At the outline-hint milestone,
JPEG55 had zero matches and five counted proposals at IoU 0.85. The
source still lacks independently verified motif geometry and aspect.

The `print_outline_suggestion` revision is recorded in both configuration
profiles. The source search took 64 ms median and 92 ms p95 over 20 warm
runs with one OpenCV thread. Golden, Domoritz, the stock challenge and
the frozen comparisons reproduced their existing instance and crop metrics.
The frozen synthetic control retains 18/18 instance matches and the seeded
geometry test retains 11/11 whole-print and 12/12 motif passes. This camera
sample remains a development case.

### Corroborate a recovered outer contour with the printed grid

On the same 33%/JPEG55 input, the local stock outline encloses an existing
weak outer contrast proposal and a lower segmentation fragment. The
expanded outline makes both native side margins and repeated perforation
rows measurable across the fragment's proposed cut. The check looks for
paper and cast shadows right up to that cut. One matching pair is enough
to record the wider visible outline for the whole print and move the
fragment to page-level count review. The original proposal still supplies
the crop warp. Both modes explicitly require confirmation of the paper edge.
This behavior is recorded in multipanel revision 6.

At instance IoU 0.85 this camera input improves from zero to one matched
print, from five to three false proposals and from five to four counted
rows. Final mask IoU against the draft reference is 0.9899, up from 0.8392.
The original whole-print preview still has about 16.37% cutoff against the
draft reference. The better outer contour is available as an editable
suggestion. Neither mode can be downloaded automatically. The four
retained preview JPEGs, crop homographies and original warps match the
baseline byte-for-byte or coordinate-for-coordinate. The case has no
independent motif geometry or aspect reference.

The normal upload was replayed on desktop, mobile and through the API.
The editor shows the wider outer contour, and count review shows the
uncounted fragment separately. Saving the outer crop leaves the count
decision available. Adding the fragment gives five rows and repeating
that request returns the same photo. A count confirmation must acknowledge
the deferred fragment. The unchanged native camera source and the JPEG75
and lossless 33% variants reproduced their preceding results. The latter
two still rely on the outer suggestion because the proposed cut lacks the
required perforation spacing across it.

One- to three-pixel separate rims remained counted before and after
JPEG55 recompression. Spanning gaps and stronger shadows on three sides
also prevented grouping. A seam-only dark mark and the weakest raised
layers can still look continuous. On the actual negative sleeve the left
margin continues through the split, while the right margin shows a native
lightness change of about 8.5 Lab units. Accepting one continuous margin
would also accept 16 of 33 separate-layer controls. The three sleeve
proposals continue to require individual review.

Golden, Domoritz, the stock challenge and the prior mixed frozen replay
reproduced their previous instance and crop measurements. The frozen
synthetic control retains 18/18 matches. Seeded geometry retains 11/11
whole-print and 12/12 motif passes. On the positive camera example, the
count-review stage took 100 ms median and 114 ms p95 over 20 warm runs
with one OpenCV thread.

### Straighten the proposed contact print for review

For the annotated 1346 × 1014 JPEG55 input, PhoSca now shows a separate
610 × 782 lossless preview of the corroborated outer contour. Its own
homography projects the four suggested corners onto the frame. Owned
bilinear sampling uses the original page. The desktop and mobile editors
label the preview as a proposal and flag the estimated aspect ratio. The
source preview and the four initial warp transforms match the preceding
revision. Both named exports still require review.

| Whole-print measurement against the development draft | Source crop | Proposed preview |
| --- | ---: | ---: |
| Rendered footprint IoU | 0.8363 | 0.9884 |
| Visible print area cut off | 16.37% | 1.07% |
| Maximum corner error in source pixels | 60.37 | 4.84 |
| Foreign interpolation contribution | 0% | 0.118% |
| Output pixels with foreign taps beyond one source pixel | 0 | 220 |

The preview shows the outer rows and handwritten header. The residual
edge discrepancy includes a trace of wood at the bottom. The case lacks
verified physical aspect, motif landmarks and a separate motif boundary.
It remains development evidence. The JPEG75 and lossless variants still
show the editable outline with their previous source previews.
Searching eight pixels in each direction for a nearby rail pattern in the
rectified analysis recovered a witness on three proposals in those two
variants. The same search accepted 47 of 63 controls with independent
one- to three-pixel paper rims at shifted positions. The fixed separation
check remains in use.

The normal desktop and mobile uploads showed the same PNG. Saving the
edited corners removed the proposal PNG from the current photo and kept
the count decision available. Restoring the deferred fragment invalidated
the overlapping PNG immediately. Reprocessing left the outline hint
available for comparison, with both exports still held. Before that count
change, reprocessing regenerated identical PNG pixels. The PNG is 573,512
bytes. Rendering, encoding and writing took 37.6 ms median and 40.3 ms
p95 over 20 warm runs on one OpenCV thread.

The instance, mode-availability and crop metrics match the previous
replays for Golden, Domoritz, the stock challenge, both frozen detector
comparisons and the seeded crop benchmark. The seeded results remain
11/11 for whole prints and 12/12 for motifs. The configuration profiles
record the preview under `print_outline_suggestion` revision 2.

### Keep distinct prints when a paper rim is offset from the model edge

The fragment contour can fall several pixels inside or outside a physical
paper seam. For recovered grid outlines the separation check now samples
the original page on both sides of the proposed cut, including slightly
slanted lines. It requires similar stock color across all eight sections
of the line. A few bright perforations in printed detail cannot supply
that distributed support. A supported rim retains both count proposals.
The recovered outline remains available for review when the original
ownership checks allow it. The change is recorded in multipanel revision 7.

On crops derived from the 33% JPEG55 development capture, the added check
retained separate proposals in all 40 raster-rim controls that previously
linked, and all 38 that linked after another JPEG55 encode. These were
one- to three-pixel light or muted rims placed at eight offsets. With
subpixel coverage and tilted edges, two-pixel rims went from 60 to zero
linked cases before recompression and from 61 to three after JPEG55.
Thinner rims and weak shadows confined to printed ink remain ambiguous.
The wider search is a veto on grouping. The rail witness and the review
decision for JPEG75 and the lossless variant use the same criteria as before.

The unmodified JPEG55 upload retains four rows and the same source JPEGs,
proposed outline PNG and crop transforms. Both export modes continue to
require review. On an upload with a muted two-pixel rim eight pixels above
the proposed cut, count review keeps six rows instead of five and a
deferred fragment. The existing five rows keep their crop transforms and
preview pixels. Neither named export is available for the held rows.

The added scan took 8.7 ms median and 10.3 ms p95 of CPU
time over 20 warm runs on one OpenCV thread. Both mode metrics and review
counts match the preceding Golden, Domoritz, stock challenge and frozen
benchmark results.

### Mark the disputed divider on the negative sleeve

On the 33% JPEG55 camera view, the left-hand classical contour crosses
the tabletop and has weak native edge support on three sides. Two separate
model proposals cover the upper three and lower four film rows. Their
footprints cover 92.7% of the broad contour. The right-hand paper contact
print and its deferred lower fragment are physically separate.

Multipanel revision 8 checks a weak region that overlaps exactly two
non-overlapping model proposals. It uses their facing edges to rectify the
source for measurement. The pale divider must have a width comparable to
the nearby dividers, and the perforation rows must keep their pitch and
spacing across it. On JPEG55 the resulting source line runs from roughly
`(117, 486)` to `(644, 486)`. On JPEG75 and the lossless 33% copy the same
check also flags the three sleeve proposals and leaves the right-hand
proposals separate. A different print in the lower slot breaks the grid
signal. Flush paper with the same artwork can still produce the signal,
including with narrow tinted rims, so the line supports manual review.
Fixed-proposal rotation checks preserve the source line through all four
quarter turns and at 45°, 225° and 315°.

The desktop and mobile count views show the line on the original. Each
flagged proposal has its own crop task and can be rejected explicitly.
The normal JPEG55 upload starts with four counted proposals. Rejecting
the three sleeve proposals leaves one counted paper print and its deferred
count suggestion. The surviving mask, per-mode crop records and preview
JPEG match their values before the decisions. Both named downloads and
the old trimmed alias stay held.

The review stage took 161 ms median and 190 ms p95 on JPEG55 over 25 warm
single-thread calls with concurrent benchmark load. The frozen, seeded,
Golden, stock, flatbed, clipped-scan and reduced contact-sheet comparisons
reproduced their instance and both crop-mode metrics. For JPEG75 and the
lossless 33% copy, the right-hand contact print still falls below the
instance threshold at 0.8317 and 0.8286 mask IoU, respectively. The shared
grid check reads a different perforation phase at those proposed cuts.
The contact sheet remains a development source.

### Compare the whole contact print with its counted fragments

Multipanel revision 9 keeps the right paper outline, the original crop
and the counted fragments available in the two 33% variants. The measured
paper outlines reach 0.9885 mask IoU on JPEG75 and 0.9898 on the lossless
copy against the draft. The original active masks score 0.8317 and 0.8286
respectively. The wider outline is available for the reviewer to load.

The reviewer sees the marks at the proposed cuts. Source checks require
continuous material beside both ends, the regular grid within eight rows
in the rectified chart and periodic contrast in the nearest native pixel
samples. The cue can also appear over a separate paper piece with a rim
thinner than two native pixels. Both parts remain counted in that case.
An unrelated neighboring outline, a pronounced paper rim or a broken
margin makes the check abstain. The original JPEG55 retains its existing
deferred-fragment flow.

With the normal upload route, Desktop loaded the JPEG75 outline as a draft.
On mobile, rejecting one of the two lossless fragments removed just its own
mark and changed the count from six to five. The other mask, preview JPEGs
and per-mode crop metadata matched their earlier values. Both mode downloads
and the trimmed alias were held before manual geometry review. Manual entry
of test paper dimensions exposed a separate full-print export issue while
a linked row was still counted. The same issue reproduced on the baseline.

The added review stage took 114 ms median for JPEG75 and 182 ms for the
lossless copy in 25 warm runs with one OpenCV thread and concurrent test
load. Both crop-mode metrics reproduced across the regression sets.

### Open the measured contact outline from count review

The JPEG75 parent and its six-vertex fragment pass the initial overlap
checks. At the proposed cut, native row 500 maps to row 502 in the
rectified grid. The stricter deferral check finds no rail witness there.
The review search finds one at row 496 with a pitch of 17 rows. JPEG55
passes the stricter check with the cut mapped to row 489 and a witness
at row 495. The lossless derivative also misses the deferral gate on
initial overlap. It still yields a measured outline and two cut lines
for review. The active evaluation counts remain 0/1 with five extra
proposals on JPEG75 and 0/1 with six on the lossless derivative.

A separate neighboring proposal outside the paper boundary leaves the
JPEG75 review group available. A competing claim inside it or a verified
material rim makes the group check abstain. Nine generated one- to
three-pixel muted rim controls, at offsets up to six native pixels,
retain all five active proposals and produce no shared review group.

**Aussenumriss prüfen** now opens the measured quad for the sole active
whole-print row in a count group. Desktop loads the quad into the existing
inline editor. Mobile requests the geometry task by photo UUID within
the current album and opens it with the draft already applied. The
request honours the same tenant, rejection and completed-review filters
as the ordinary queue. The tested lossless parent was item 33 in the
unfiltered order, outside its default 25-item slice. If the target was
resolved while the count screen was open, Mobile offers a return to the
same count page.

Both clients were tested with both exact derivative uploads. Opening the
draft preserves the active proposal IDs, count revision, masks, both
crop mode records and review JPEG hashes. Both named downloads and the
legacy trimmed alias return 404, and the review JPEG remains available.
Saving the draft with test dimensions of 203 × 264 mm leaves both modes
held while the linked fragment still overlaps. Explicit fragment
rejection and reprocessing make the tested whole-print output available.
The motif continues to show curvature and margin-marking reasons.

| Draft whole-print metric | JPEG75 before | JPEG75 after save | Lossless before | Lossless after save |
| --- | ---: | ---: | ---: | ---: |
| Rendered footprint IoU | 0.8295 | 0.9870 | 0.8246 | 0.9875 |
| Visible area cut off | 16.77% | 1.22% | 17.27% | 1.17% |
| Maximum corner error in source pixels | 68.56 | 5.71 | 69.19 | 5.71 |
| Foreign interpolation contribution | 0.515% | 0.101% | 0.410% | 0.113% |
| Output pixels with foreign taps beyond one source pixel | 1,928 | 158 | 1,489 | 222 |

The derivative manifests share a draft paper annotation. Independent
physical aspect and motif landmarks are unavailable for this measurement.
The measured action is a review improvement. The active detector and
its thresholds remain as documented above. Targeted task lookup averaged
4.41 ms for JPEG75 and 4.91 ms for lossless in eight warm local requests
per source. The saved render metrics are identical before and after
rejecting the linked fragments.

### Keep manual exports held while paper proposals overlap

Saving a measured outer rectangle now checks the other active members of
its paper review against their current source outlines. The comparison
uses their native footprints, including the six-vertex JPEG75 fragment.
The original review records link the proposals across edits. Both crop
modes keep their own geometry reasons and receive the overlap reason
while the source claims intersect. A count confirmation keeps the holds
until the conflicting geometry or active proposals change.

On the normal JPEG75 upload, the measured edit saves the wider rectangle
and leaves the fragment counted. Both downloads and the trimmed alias
return 404. After an explicit fragment rejection and reprocessing the
saved rectangle, the whole print becomes downloadable. The motif keeps
its own boundary review. Restoring the six-vertex fragment without an
explicit warp immediately holds the existing whole-print export and
reopens even an approved geometry task. The synthetic lifecycle test moves
the restored fragment outside the parent's footprint and checks release
after reprocessing. Older manual records with paper-grid evidence need a
current overlap check before their named downloads are available.

### Require an intentional motif outline in review

The right contact print in the 33% JPEG75 upload has a native motif trace
but no proposed four-corner motif. Replaying the old desktop 6% inset on the
measured print retained one of 1,938 sampled header pixels and cut all five
bottom panels. The editor now starts its handles at the paper corners and
keeps **Save** disabled until the reviewer adjusts a corner, loads a measured
suggestion, or explicitly chooses the paper corners for a borderless motif.
The source traces and the mobile review hint identify which content to check.

For the same source with SHA-256
`a3c019249af7ce646813de4e5ec6555699cc1785730b5016994d13206d70988e`,
upload and process the page, load the suggested outer outline, and reject
the remaining counted fragment explicitly. In this development replay,
the test dimensions were 203 × 264 mm. After reprocessing, the whole print
downloads and the motif and trimmed alias return 404. Switch to **Nur
Bildmotiv**, check that a click, a one-screen-pixel nudge and a reverted
drag each leave **Save** disabled, then set the motif corners near
`(34, 7), (571, 7), (571, 789), (34, 789)` in the 609 × 792 test print
plane. The saved 538 × 783 motif retained 1,938/1,938 header pixels and
315/315 anchors across 35 image fields. Source-to-output comparison
confirmed owned interpolation support. The 609 × 792 whole-print download
was byte-identical before and after the motif edit. Both Gallery download
links returned the verified JPEGs. These development measurements use a
manual review and test paper dimensions.

### Keep genuine motif reviews open after legacy geometry approvals

A full-print JPEG can be available while the motif has its own unresolved
geometry. The review queue now uses both modes. A new complete geometry
approval returns 409 while a motif boundary, visibility or protected-content
check is open. At this milestone, Mobile pointed to the desktop motif
editor and left the other review answers and Skip available. A disabled motif classifier or
an older crop without a motif mode can still complete the geometry review.

On the 33% JPEG75 contact print, the prior application had accepted a
mobile geometry approval and hidden the task while the motif and trimmed
alias returned 404. The updated application reopened that task from the
unchanged prior review history. The whole print continued to download.
After the desktop motif edit, both Gallery downloads matched the earlier
verified JPEGs byte for byte. A fresh mobile approval then completed the
task. A focused test also keeps the task open when reprocessing resolves
the motif on the same paper mask without a fresh approval.

### Review and correct the motif on mobile

On both 33% contact-sheet derivatives, the ready whole print has a
contested motif seam. At 203 × 264 mm the fitted right edge bends 2.558
rectified pixels on JPEG75 and 2.543 on the lossless derivative, above
the 2.25-pixel review threshold. Native sampling sees a second parallel
transition about three to five pixels further in. The top edge only
has fragmentary support. A strong mark check detects one region of
the heading from approximately `(829, 161)` to `(996, 178)` in the
original. The mark region covers only a part of the complete header.

A candidate enclosing the high-contrast paper samples retained the
development header points and all 35 panels on both derivatives. It
retained a wide right strip where the existing paper fit also includes
some exterior pixels. The same candidate lost all 135 marked pixels of
a generated faint date near the top, and kept 59 of 135 on a left-side
variant. Some physically separate row controls also satisfied both
film-grid witness checks. These counterexamples leave the current
boundary and release decisions in force. The source overlay draws the
measured sections and prompts the reviewer to check beyond them.

A separate object-type trial compared the nested left candidate with
the adjacent contact print at JPEG55, JPEG75 and lossless PNG. The
35 fields have opposite polarity in every input. Feature matching after
inversion gives 39, 55 and 67 inliers respectively and zero in normal
polarity. The clear lip carries the backing grain, with a correlation
near 0.92, while the light interior measures 0.01–0.02. An opaque
negative print inserted into the same clear protector also passes both
tests, with 202, 206 and 236 inverted inliers. Ownership and material
inside the print remain ambiguous, so the nested candidate stays in
count review.

After the paper outline is saved and the whole print is ready, the
phone editor can switch between the paper and the motif. With a held
motif it starts at the paper corners until the reviewer drags an edge,
chooses to include the paper boundary or loads an existing measured
suggestion. A tap, one-pixel drag, fully reverted drag or exterior quad
cannot save a first motif. The live output aspect uses the source-to-
paper homography. Saving sends the existing outer box, stored paper
vertices, visible polygon, paper measures and the intentional motif
quad to the usual edit route. The named motif URL appears when the
renderer makes that mode ready.

On a normal JPEG75 upload, both corrected variants and the old trimmed
alias were downloaded after the deliberate phone edit. The original
whole-print download was byte-identical. The active four proposals,
their count revision and the stored outer geometry stayed the same.
The 538 × 783 motif retained 1,938/1,938 source heading samples,
251,331/251,331 grid samples and 315/315 anchors across all 35 image
fields. The recorded matrix independently reproduced the export with
owned sampling and JPEG luma mean absolute error of 1.48. The lossless
upload reproduced the same phone workflow, with both overlapping
fragments explicitly rejected. Its 538 × 783 export preserved
1,940/1,940 heading samples, 250,457/250,457 grid samples and the same
315 anchors. Both tests use development paper dimensions and existing
source pixel anchors. They provide manual workflow receipts.

### Review visible markings outside the proposed motif

A straight join between light stock and image pixels can be real even
when dates or captions are printed on the stock. The motif check examines
only the proposed discard strips inside the rectified paper. It confirms
visible contrast using the original source pixels before asking for
review. The same check runs on an otherwise exportable motif under a
photo corner. It keeps the whole-print geometry and render available
according to their own checks, and leaves a deliberate reviewer-supplied
motif outline to the existing manual geometry and sampling checks.

The 33% JPEG55, JPEG75 and lossless contact-sheet derivatives all retain
their curved-edge reason and now mark an actual fragment of the printed
heading. Desktop and mobile show that region on the page. The prompt asks
the reviewer to inspect the entire rim. On the normal JPEG75 upload,
the motif and trimmed downloads returned 404 until the desktop edit.
After the edit, both Gallery downloads succeeded and the motif retained
all 315 anchors across 35 panels and all 1,938 sampled heading pixels.
The isolated added check took 9–11 ms median on the three derivatives
in 45 warm runs per variant with one OpenCV thread under concurrent test load.

The paired 162-case development comparison leaves the whole-print status,
shape and pixels identical. It sends 25 of the 54 previously ready motifs
that cut known markings to review. All 36 previously ready negative
controls and six ready cases with labels inside the motif stay ready.
Controls include rotation, perspective, blank rims, small specks, shadows,
and dates printed on the album surface. Low-contrast pencil, isolated
characters and fully connected handwriting remain open cases. The
validation fingerprint includes the margin check for precise crops with
border analysis enabled.

### Check faint graphite and blue ink against the original paper

When the stronger mark check finds no evidence, the same discarded-paper
strips are checked for a coherent line of faint, shaped strokes. Candidate
strips use luminance or cool ink colour against the local paper. Native
samples confirm a narrow stroke against matching stock on opposing sides.
Both sides must be inside the measured print and outside the motif. A
candidate that depends on a brighter album region or on image pixels
across the seam cannot confirm a faint stroke.

Against the previous mark check, the 162 paired development renders add
three motif reviews with known content loss. All 162 whole-print
statuses, output shapes and pixels reproduce. All 36 previously usable
negative controls and six in-image lettering controls remain usable. The
separate controlled strip set adds 33 detections among 216 faint dates
and adds no review among 660 controls. Twelve of 18 attenuated copies of
the contact-sheet heading also gain a review hint. The three matching
cleared-paper controls abstain. These slices include model-free geometry
fixtures and synthetic derivatives of real handwriting.

The normal upload and export path used a reproducible pale-graphite
fixture. The detector proposed one print and five extra areas on its
synthetic motif texture. Reviewers removed the extra proposals and
confirmed the paper outline. The graphite hint then kept the motif and
trimmed alias at 404 while the whole print remained available. After a
deliberate motif edit, the exported JPEG included all 147 sampled ink
centres and the whole print was byte-identical. The real JPEG75 contact
sheet also reproduced its existing mobile, desktop and Gallery workflow.
The isolated mark check added roughly 1–8 ms median across the tested
small strips with one OpenCV thread. Very faint graphite, isolated
characters and connected handwriting remain open cases. The validation
fingerprint now binds this check.

### Find separate faint markings after a strong heading

Margin review revision 5 keeps the strong-mark result and also checks
eligible pale strips for a separate mark. Each strip contributes at most
one region from the existing faint-stroke detector. A confirmed strong
box is masked with a four-pixel halo in the rectified paper plane. In
this additional pass the remaining strip needs at least 15 pixels with
a Lab luminance residual of nine levels. Every candidate still requires
matching stock on both sides of its original strokes. The decision to
hold the motif was already made by the strong mark, so this pass adds
source hints for the reviewer.

The normal contact-sheet replay maps the original heading region to
approximately `(829, 161)` through `(996, 178)` in both versions. The
new hint points to the handwritten number immediately to its right.
On JPEG75 the second region spans approximately `(1077, 145)` through
`(1170, 161)`. The two regions cover 552 of 1,938 independently sampled
header pixels, up from 344. The lossless variant confirms only the
leading handwritten number. JPEG55 keeps its original heading region.
The whole-print preview remains ready and byte-identical. Both geometry
editors show the two source regions while the motif remains held until
the deliberate edit.

In 120 date transfers onto the JPEG75 and lossless sources, regions
covering more than 45% of the injected ink centres increase from 8 to 22.
Each derivative contributes 11 hits after the change, up from 5 and 3.
The previous hits remain. None of the 408 controls gained an extra
mark. Of those controls, 114 had the strong marker needed to exercise
the amended path. Small 14-level graphite and the transverse writing
along the left-side test cases still lack reliable hints. In the
diagnostic that supplies the correct stroke locations, original-pixel
corroboration passes for 76 of 88 misses by the three-pass prototype.
Those misses motivate checking the candidate contrast and grouping
before opening up the original-pixel validation.

On one OpenCV thread in 45 interleaved warm pairs, the median margin
check went from 4.73 to 24.64 ms on JPEG75 and 4.67 to 26.29 ms on PNG.
The new p95 values were 31.83 and 43.15 ms. Multiple faint passes per
strip recover more injected dates in the comparison but take longer.
The new crop fingerprint records the one-pass revision.

Frozen, seeded, Golden, NerdScan, Domoritz, stock and contact-sheet
replays reproduce detection and both crop-mode aggregates. The frozen
control retains 18/18 detections. The seeded geometry retains 11/11
whole prints and 12/12 motifs. The normal JPEG75 and PNG uploads keep
both motif and legacy trimmed routes held until the intentional edit.
After that edit both exports retain all 35 sampled fields and all of
the reference header points on their respective source.

### Complete a dashed handwritten note using the original pixels

On the lossless contact sheet the ordinary glyph group stops before the
last part of the handwritten number. The last two components have
approximately the same baseline as the prefix. A separate horizontal
dash sits in the gap. The JPEG75 group already spans the entire note.
Increasing the ordinary grouping radius from 1.9 to 2.0 times glyph
height also joins the PNG note in a probe. That adjustment would apply
to every faint row, including an empty gap between neighboring marks.

Margin review revision 6 keeps the ordinary grouping rule. When a
strong mark has already held the motif, a confirmed group of separate
glyphs can check for one nearby dash and a unique pair of similarly
sized, structured components on the same baseline. The dash and each
component must pass original-pixel validation separately, including
matching paper on both sides of their strokes. If a component is absent
from the source, the initial box stays in use. This extends a source
annotation used in review.

The lossless PNG mark extends from about source column 1123 to 1170
and covers all 96 sampled dark centres in the handwritten suffix,
up from zero. JPEG75 retains all 94 corresponding centres with the
same box as before. Removing or turning the dash, widening the gap,
moving the suffix off the baseline or erasing one component in the
original produces no extra coverage. The 408 prior nuisance profiles
and 384 additional neighboring printed-glyph controls keep their
previous regions. The 120 injected-date views keep the same measured
ink coverage, including cases whose genuine original annotation now
covers the suffix.

Desktop and Mobile show both source regions on both normal uploads.
The whole-print render and JPEG match revision 5 byte for byte. The
motif and legacy trimmed routes remain held until the deliberate edit.
After the mobile save, both exports retain all 35 measured image fields
and all 1,938 or 1,940 header samples. In 35 warm paired single-thread
samples against revision 5, the isolated margin check added a median of
0.39 ms on JPEG75 and 1.78 ms on PNG. The PNG p95 went from 27.13 to
29.33 ms. The detector, both crop-mode aggregates and mode availability
match revision 5 on all eleven frozen and development comparisons. The
frozen control retains 18/18 detections and the seeded replay retains
11/11 whole prints and 12/12 motifs.

### Hold motif trims for isolated strong strokes

The new component check handles cases with fewer independent strokes
than the brighter-paper row detector requires. It uses the existing
component labels and local-paper estimate. The strip must
have median Lab lightness of at least 172. Candidate ink is at least 38
lightness levels below the morphological paper estimate. A candidate
component needs a surrounding collar contained in the discarded strip.
Its stroke radii must be consistent. The accepted geometries are
separated enclosed counters along a joined word, vertically arranged
counters with a continuous side stem, or a centred counter with a narrow
apex and two separate legs. The test considers four quarter turns.
The resulting region still needs contrast of at least 32 in the
original image for at least 70% of sampled stroke pixels, with at least
ten valid samples. A confirmed hit uses the existing motif-review reason
and keeps the full-print geometry and pixels. The crop profile records
motif margin revision 3.

The paired strip replay at `df92258` and revision 3 starts from
24 positive patterns and 22 controls. Each is evaluated at three
scales in the native image, JPEG65 and a projective JPEG view. The
patterns include OpenCV lettering and shapes extracted from the real
contact-sheet heading. The positive strip detections rise from 11/216
to 37/216. The same 21/198 control views report a hint in both runs.
The remaining controls include isolated rings, dust, paper fibres,
looped wires, water rings and scratches. A second experimental cue
using differences between neighbouring component shapes raised the
combined result to 50/216 in the same positive views. It remains a
prototype pending broader full-crop and nuisance validation.

In 22 paired whole-image renders, all whole-print JPEGs are byte
identical. Seven additional motif crops go to review. Their previous
crops excluded every sampled centre of the marked strokes. The normal
upload of the synthetic joined-word example produces one proposal on
the inset image (0.6499 draft IoU against paper). After the user
confirms the paper outline, the whole-print download is available.
The motif and trimmed alias return 404, and a general geometry vote
returns 409. After an explicit motif edit, its export contains all
2,055 sampled stroke centres and the whole-print JPEG is byte
identical. The 12 quarter-turn regressions for an isolated letter, a
joined word and paired rings exercise this review marker through
`correct_photo_crop`.

The paired mixed detector runs on Golden, the clipped NerdScan scan,
Domoritz, the JPEG55 contact sheet and the stock challenge reproduce
all previous detection, mask, render and mode availability metrics.
The frozen synthetic detections and the separate 11/11 full-print and
12/12 motif seeded geometry results also reproduce. The very pale date
in the README example was still lost at this revision.

In 36 paired warm samples on one OpenCV thread, the unmarked
`1920 × 1160` fixture took 82.53 ms before and 82.35 ms after at the
median for the complete margin check. At that resolution, the joined
word took 81.02 ms before and 18.24 ms after. A confirmed shape bypasses
the subsequent faint-mark search.

### Audit cross-source fusion and scalloped stock conflicts

The greedy cross-source cluster can transitively relate two proposals
from one family through a proposal from the other. In controlled
two-print overlaps it undercounts 16 of 32 ambiguous trials. A one-to-one
assignment fixes those undercounts and adds one false proposal in all 32
matched single-paper trials. On 16 three-layer trials with a matching
bridge it loses the third print. Two exposed native corners can add
evidence for distinct candidates, but the same internal corners also
appear in a sheet printed with a reproduction of two overlapping
prints. Keep the current assignment for this milestone.

An independent read-only audit of the production cluster members at
`df92258` replayed 122 page/checkpoint pairs. Each of the two registered
512 px development detectors ran on 10 Golden pages, 11 additional
development views (including contact-sheet derivatives), 16 frozen
synthetic pages and 24 stock challenge cases. Of 29 mixed clusters,
two contain two proposals from the same family. Both pairs also pass
the direct same-instance test. The original pixels support one physical
print for each pair. One is the native camera contact sheet. Its
outer paper outline wins with 0.99405 draft IoU. No audited cluster
loses a verified physical print through an incompatible bridge.

The other case is the upper print on the native 4,816 × 6,871
ScanCropper development page with the synthetic-control checkpoint.
The inner classical outline wins fusion (0.85386 draft IoU). A
separate model proposal at confidence 0.460 covers the right half
of that print, native box `[2345, 543, 3550, 2010]` and draft IoU
0.55384. Removing exactly that proposal immediately before the native
scallop check leaves the fusion winner intact. The recovered top mask
changes from 0.85386 to 0.99795. The lower print stays at 0.99770.
Recognition against the two draft prints changes from 2/2 with one
extra to 2/2 with zero extras. The classical-only run reaches the
same measured outlines. Both recovered contours remain in review. The
ablation identifies the ownership conflict and uses knowledge of the
erroneous proposal. Any runtime decision still needs independent
evidence for the paper edges and stacked-print controls.

In a separate NerdScan probe, branch and step contours continue over
the three exposed sides of the contained lower detection. Subpixel
translations and brightness changes of 1–2 levels preserve similar
continuity in several counterfactual controls. That signal alone
cannot establish the count. The existing two disputed proposals and
the count review remain in place.

### Show scalloped paper in ownership review

Revision 2 retains the measured silhouette as review evidence when a
verified outer perimeter has a single, substantially contained partial
claimant. The partial claimant must lack its own verified scalloped
outline. Both rows stay active. They share the source quadrilateral of
the measured perimeter as a same-page claim group. The original
detector masks, symmetric conflict checks and output warps remain in
use. Desktop and mobile read the measured contour via
`suggested_edge_traces` and the editable corner proposal via
`suggested_warp_quad`.

The pinned ScanCropper multipart upload with `c31317f…` still yields
two matched drafts and one disputed proposal. It now exposes a
1,959-vertex contour on the upper print and four editable corners.
The automatic source-corrected JPEG is byte-identical. The outline
gives the reviewer the measured paper evidence without replacing
the active detector footprint. The independently run controls
include a genuine second photo and nearly registered layers of the
same art. Very thin layers preserve crossing texture and can be
visually indistinguishable from an internal image contour.

In the previous API replay, manually saving the four corners made
`full_print` return 200 while the overlapping proposal was active.
Both modes and the legacy trimmed alias now return 404 across the save
and reprocess. The same-page claim survives manual edits via the
original prediction snapshot. Restoring a rejected, overlapping row
reopens the sibling's crop review. The regression covers the existing
paper-grid flow and the new claim type, including stale read metadata.

Once the competing row is explicitly rejected, the physical paper
boundary still matters. Rendering the four-corner JPEG as a named
whole-print crop would cut some scallops and sample backing in other
places. The reprocessed review returns an RGBA preview of the
measured polygon. Changing 1,497,334 outside-paper pixels in a
controlled source replay left the entire preview byte-identical.
A deliberately selected motif is eligible for manual export only if
it passes the normal crop checks and clears the paper and neighbours
by at least two native pixels at the sampled side stations. The
test includes a 0.5 px² overlap at a motif corner and a quad which
reaches the paper cuts. Neither qualifies. Stale motif files require
the current sampling receipt. The scalloped whole-print JPEG remains
in review.

Paired runs reproduce the previous detection and crop summaries for
the frozen detection and seeded geometry benchmarks, Golden,
NerdScan, Domoritz, the JPEG55 contact sheet and the stock challenge.
The native ScanCropper views are development evidence.

### Measure seed displacement and competing print edges

The frozen synthetic-control detector `c31317f…` with fallback at
`110415d` matches all 18 instances at IoU 0.85. Seven full-print modes
and six motif modes are deliberately unrecoverable. The other 11 and
12 modes respectively are the denominators for the crop test. The
negative page expects zero proposals and yields zero. The first
detector pass succeeds for 6/11 full prints and 7/12 motifs. The separate
reference-jitter control succeeds for 11/11 and 12/12.

| Frozen instance | Expected full / motif | Synthetic-control full / motif |
| --- | --- | --- |
| `borderless`, `portrait`, `white_on_white`, `white_motif`, `cast_shadow`, `glare` | Pass / pass for each | Pass / pass for each |
| `white_margin`, `camera_tilt`, `touching #0`, `touching #1`, `overlap #1` | Pass / pass for each | Review / review for each |
| `touching_same_color #0`, `touching_same_color #1`, `overlap #0`, `photo_corners`, `clipped`, `curl` | Review / review for each | Review / review for each |
| `paper_corner_mount` | Review / pass | Review / pass |

For the five recoverable failures, the decoded model, fused proposal
and renderer input have identical corners. The visible source masks
improve from roughly 0.96–0.97 to 0.98–0.99 IoU while the selected warp
seeds remain unchanged. Seven native sides have competing fitted lines.
The existing search already finds the physical edge on all seven,
0.035–0.453 px from the reference on average by side. Each set of
alternatives includes an ink edge roughly 8.84–12.16 px inside the paper.
The general evidence score favours the ink edge on two sides. Holding
orientation and varying the corridor alone exposes further alternatives.

In a separate geometric control, each recoverable print's reference
edges move in parallel by the specified fraction of its shortest
native side. Keep the fixture's visible polygon, neighbours, protection
and renderer settings. Use the existing jitter seeds for the known
obstructed prints. The metric is the ordinary synthetic mode pass,
including correct review of obstructed cases as a separate check.

| Native side offset | Full print passes | Motif passes |
| ---: | ---: | ---: |
| −4.00% | 5/11 | 6/12 |
| −2.50% | 2/11 | 3/12 |
| −1.50% | 2/11 | 3/12 |
| −0.75% | 7/11 | 8/12 |
| 0.00% | 10/11 | 11/12 |
| +0.75% | 11/11 | 12/12 |
| +1.50% | 11/11 | 12/12 |
| +2.50% | 10/11 | 11/12 |
| +4.00% | 10/11 | 11/12 |

The exact reference seed can still be attracted to the inner line on
the front `overlap` print. The material on both sides of the
competing lines distinguishes the edge of the paper strip.

Print-boundary revision 11 keeps the three-pixel stock retry. If it
finds no line, the retry can reuse that side's original native search
radius. Exactly one side must be ambiguous. The three other sides must
corroborate pale stock against backing. The new line must still satisfy
support along the side and on the four adjacent corner arms, and the
result may not intersect a neighbouring print. The geometry remains
in review. The regression covers a closer and a farther initial
proposal plus a corner obstruction and a neighbouring print.

| Improved synthetic-control review preview | Max corner error | Cutoff beyond 1 px | Output pixels with foreign contributions | Landmark p95 |
| --- | ---: | ---: | ---: | ---: |
| `white_margin` | 5.27 → 0.49 px | 1,320 → 0 | 263 → 0 | 4.318 → 0.363 px |
| `camera_tilt` | 5.65 → 0.34 px | 706 → 0 | 466 → 0 | 3.565 → 0.192 px |

The two previews gain paper in the source footprint. Their aspect
errors change from 2.25% to 0% and 1.43% to 0.62%, respectively.
The synthetic-control detector count, the motif results and all
18/18 seeded instances with 23 recoverable mode passes reproduce.
The next unresolved geometry cases have multiple competing sides,
different backing colours across the gutters or a trusted inner edge.
Paired replays on Golden, both NerdScan runs, Domoritz, both
contact-sheet JPEG variants and the stock challenge reproduce previous
instance and crop summaries. The stock retry sends no additional
usable mode to review in these comparisons.

The isolated native stock retry took 0.69 to 3.48 ms median and
0.90 to 3.89 ms at p95 for `white_margin`. The `camera_tilt` pair took
0.74 to 3.31 ms median and 0.91 to 3.64 ms at p95. These values use 40
warm paired samples on one OpenCV thread with the original evidence
and fitted proposals already available.

### Compare local paper centering and noise scaling for faint dates

The very pale README date separates into three aligned components at
multiple rectified contrast levels. Two are wide, joined strokes. The
existing component grouping yields no proposal, so the native flank
check never gets to evaluate the date. A separate scratch grouping
looks for the aligned run and variation across both directions. It
passes the mark to the existing native matched-stock check without
asking for closed counters. The exact missing date goes to review in
that prototype. The hint covers 683 of 727 labelled stroke centres.

Across 432 synthetic date views, the existing implementation marks
131. The new grouping marks 164 with an absolute contrast floor, 153
after subtracting the local median and 139 with median and MAD
normalization. On 660 controls, the absolute version adds one review
on pulp fibres. The locally centred and MAD versions add none. Both
also add zero among 360 extra white-stock controls. The 162-image
paired MAD trial adds four reviews for known full lettering loss and
preserves the whole-print bytes. The paired locally centred variant
also adds four reviews for known full lettering loss and adds a
location hint to two already-held variants. All 162 whole-print
outputs remain byte-identical. The existing controls and real
development replays keep their previous review availability.

Revision 4 uses the locally centred variant after the existing
neutral and blue-ink passes. A mark must exceed the local paper
residual median by at least 14.2 Lab lightness levels. The run
needs three similar-height components with a shared baseline and
two wider components with structure in both directions. A final
native matched-paper flank check uses the original contrast.
The generated README example now routes the very pale date to
review. The prior crop retained 0 of 727 sampled stroke centres.
In the normal upload flow the example produces one proposal.
After confirming the whole-print corners, the whole print returns
200 while the motif and the legacy trimmed alias return 404.
A generic geometry approval returns 409. An explicit motif edit
then keeps all 727 sampled stroke centres in the export, with
the same whole-print JPEG bytes before and after the motif edit.

Paired final runs reproduce both modes' summaries for the frozen
benchmarks, Golden, NerdScan, Domoritz, both contact-sheet JPEGs
and the stock challenge. The exact strip matrix gains one additional
faint-date detection (52/216 to 53/216) and adds no hold among
660 controls. The complete isolated margin check for the README
date takes 6.05 to 9.08 ms median and 7.33 to 11.82 ms at p95 over
40 warm paired one-thread samples. With the same paper cleared, the
median changes from 5.70 to 5.80 ms.

### Verify that catalogued pairs represent independent acquisitions

A leave-one-out affine fit of the 15 saved American Colony draft landmarks
gives 0.409 native pixels at p95. Axis-aligned crop and resize gives 1.267
pixels. A transformed derivative fits the available coordinate evidence.
An independent exposure still needs confirmation using the native raster
noise and pixel grids. The saved transfer inventories do not contain the
two exact Colony file hashes. Earlier near-duplicate comparisons did not
cover this candidate or the Russell album.

The Getty Bayard album was disbound in the 1980s. The published page
viewer reconstructs the volume digitally. Catalogued dimensions and
individual print records are available, but the viewer does not establish
an independent capture of the mounted leaves. [Getty catalogue preview](https://api.pageplace.de/preview/DT0400.9781606068946_A49387925/preview-9781606068946_A49387925.pdf) ·
[Reconstruction description](https://www.museaward.com/winner-info.php?id=231877)

The conservation report for the two Ioannis Lampakis albums of the 1896
Olympics documents scans of mounted prints and the removal and preservation
of images from one of the albums. The family archive restricts reuse of
its website material. Downloadable paired masters and permission for this
evaluation remain unconfirmed. [Conservation report](https://resources.culturalheritage.org/pmgtopics/2013-volume-fifteen/46-T15_Adamopoulou.pdf) ·
[Family archive terms](https://www.aol.org.gr/ioannis/)

The Denver Camp Nizhoni album in collection ARL39 cross-references mounted
pages and individual scans. Page ARL-168 lists prints ARL-164 through 167,
and the print records reciprocate. The catalogue marks copyright as
unevaluated. The documented reproduction route requires licensing and the
native pixels have not been checked for acquisition independence.
[Mounted page](https://digital.denverlibrary.org/nodes/view/1105398) ·
[Individual print](https://digital.denverlibrary.org/nodes/view/1105300) ·
[Reproduction terms](https://history.denverlibrary.org/about/conditions-and-image-permissions)

The Missouri State Archives finding aid for `MS506` identifies the
Phelps album and 68 physical photographic cards. It records mounted
page scans before removal and separate scans of both sides of the
original cards after removal. Box 278 folders 68, 73 and 74 map to
album pages 1, 5B and 5C. The finding aid lists no access or publication
restrictions. The scans are held in the vault and on a reference DVD.
The native pixels remain unavailable through the public catalogue.
[Missouri finding aid](https://www.sos.mo.gov/CMSImages/Archives/manuscripts/RG998_MS506.pdf)

Malta's National Archives documents pre-removal overhead captures for
the Rodgers album `PDE_0075` and 600 dpi TIFFs of removed items. Item
`PDE_0075-059_R` has a public record. Its public image request returned
HTTP 403. The exact mounted leaf and reuse rights remain unverified.
[Malta archive report](https://embed-rech-01.dialog.cm/nationalarchivesmalta/docs/nam_official_annual_report_2024)

The Leiden Gandhãra Album 2 conservation report documents photography
of the mounted leaves followed by separate digitization of the removed
albumen prints. The public derivative for page 45 cross-references
original print `item:51876` and carries CC BY 4.0. The paired
pre-removal page master and its rights remain unavailable. Requests
to the published image hosts returned HTTP 403 from this workspace.
[Conservation report](https://resources.culturalheritage.org/pmgtopics/2005-volume-eleven/11_10_Maes.html) ·
[Print derivative](https://commons.wikimedia.org/wiki/File:Album_2_-_part_48.jpg)

Museums Victoria catalogues two McKenna albums as `ST 40219` and
`ST 40220` and marks their catalogue media as public domain. The sample
from `ST 40219` is a hinged print measuring 108 × 84 mm on a 174 × 141 mm
page, with media IDs 510624 (recto), 1944157 (mounted view) and 1944161
(verso). The workspace received HTTP 403 on the archive API. The source
pixels and separate acquisition of the recto remain unverified. Both
accessions stay in intake.
[ST 40219](https://collections.museumsvictoria.com.au/items/384572) ·
[ST 40220](https://collections.museumsvictoria.com.au/items/407538) ·
[Hinged print](https://collections.museumsvictoria.com.au/items/1561811)

The Bancroft Red Album study documents a mounted, measured print on
page 8 and a 1999 State Library of Queensland copy negative showing the
same paper edges. The public article contains a page figure. A native
full-page file and reuse terms for the pair remain unconfirmed.
[Acquisition study](https://doi.org/10.1080/23257962.2023.2190085)

The Karl Kleeman collection at the Tennessee State Library and Archives
documents whole-page album scans in Box 1, Folder 1, accession 2006-059.
Individual digitized prints are available and carry the archive's
No Copyright – United States designation. The mounted page rasters
could not be located publicly, so no pair enters the final split.
[Finding aid](https://sos-tn-gov-files.s3.amazonaws.com/forms/KLEEMAN_KARL_M_WORLD_WAR_I_PHOTOGRAPH_COLLECTION_1914-1918.pdf) ·
[Individual print](https://teva.contentdm.oclc.org/digital/collection/WWI/id/83/)

The Library of Congress record for `LOT 15564` documents 22 physical
cartes de visite and tintypes dismounted for digitization. The record
lists no known publication restrictions. The June 2022 auction trail
has yet to link an original mounted page raster to a specific item from
the lot. The catalogue API returned HTTP 403 from this workspace.
[Library of Congress lot record](https://www.loc.gov/item/2023630850/)

The Getty finding aid for the Pascal Sébah album `2010.R.16` documents
117 prints dismounted from their pages. The manifest request returned
HTTP 403. An independent capture before dismounting and reuse terms
for the paired images remain unverified.
[Getty finding aid](https://www.getty.edu/research/collections/static/pdf/2010.R.16.pdf)

The Mississippi Department of Archives and History documents individual
600 ppi scans of 1,073 original prints from two Painter albums
(`PI/1988.0006`). The capture inventory preserves their print order and
identifies the two physical album objects. The catalogue reports public
domain status. The inventory has no corresponding mounted-page masters.
[Painter collection](https://da.mdah.ms.gov/series/painter) ·
[Capture inventory](https://da.mdah.ms.gov/series/painter/colldesc) ·
[Rights and technical information](https://da.mdah.ms.gov/series/painter/gentech)

The Lochinvar album (`PI/HH/1984.0001`) from the same archive has
public-domain scans of 29 whole pages and individually indexed
prints. The capture inventory lists 31 files for pages and covers.
The dedicated records for prints 5 and 6 use the same leaf file
`100345-01-photo.tif`. Prints 7 and 8 have the same relationship to
`100347-01-photo.tif`. The print indexes therefore provide page
locations. A separately acquired print raster was unavailable, and
the workspace request for the native imagery returned HTTP 403.
[Lochinvar collection](https://da.mdah.ms.gov/series/lochinvar) ·
[Shared source record](https://da.mdah.ms.gov/series/lochinvar/detail/22535) ·
[Rights and capture inventory](https://da.mdah.ms.gov/series/lochinvar/gentech)

Cincinnati's *John Robinson’s Ten Big Shows* album was digitized both
before and after conservation. The conservators lifted affected prints,
lined them, and returned them to the album. The old Virtual Library
record `1924` returned HTTP 502. The current CONTENTdm item is
`p16998coll15/151909`, whose metadata endpoint returned HTTP 403 from
the workspace. The reports establish a separate capture of the treated
album. A capture of an original print during the unmounted interval
and rights for a native pair remain unverified. Repaired visible image
areas also differ between the two stages of conservation.
[Pre-treatment report](https://libapps.libraries.uc.edu/liblog/2015/04/preservation-exhibit/) ·
[Post-treatment report](https://libapps.libraries.uc.edu/thepreservationlab/2022/10/vestiges-of-color-in-an-album-of-john-robinsons-ten-big-shows) ·
[Current album record](https://cdm16998.contentdm.oclc.org/digital/collection/p16998coll15/id/151909/)

The Royal Carriages Museum conservation study records scanning original
prints in the album with 85 pages. The public paper includes examples
of the physical album and scanner captures. Downloadable native pairs,
independent capture of unmounted prints and rights for the museum
rasters remain unverified. The article itself carries CC BY 4.0.
[Conservation study](https://isprs-archives.copernicus.org/articles/XLVIII-M-2-2023/1449/2023/)
