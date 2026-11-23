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

The report assigns detections to visible instances once and scores the final
source-to-output transform of each crop. It reports the sampled footprint,
missing content, foreign samples, samples outside the capture, and corner
error in source pixels when annotated corners exist. Aspect and landmark
errors require paired reference geometry. For banded cutoff and contamination,
one source pixel around the annotation boundary accounts for annotation and
raster uncertainty. The evaluator replays the sampler recorded for each mode
against the source ownership mask. It counts every output sample with a foreign
contribution and requires zero such samples on the synthetic references.
The footprint uses the inverse-mapped output pixel centers, which avoids an
extra half-output-pixel margin on downsized crops. The gate also checks output
sampling along both axes and the correspondence between the stored quad and
the actual homography. Real release requires at most two source pixels of
error at every reference corner.
The `runtime.image_to_jpeg` duration includes decode, preparation, detection,
protected regions, crop rendering and JPEG writes. It excludes metric
computation, database access, OCR and later enhancement.

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
recording `split_policy.album_independent: true`. The release check requires
at least two distinct physical album groups, unique pinned image hashes and
verified annotations for every page.

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
automatically eligible results per mode among recoverable prints. Each mode
needs recoverable references and passing coverage in at least two album
groups, with the 90% threshold applied to each group. Report schema 17 binds
per-mode eligibility to native instance-mask scoring, the source-stock
occlusion check, album-window detection, dense-page tiled and deskewed inference,
the classical line-pair fallback, the configured visible mask refiner and
detector-bound confidence calibration. A verified motif can
qualify separately when its print corners require review.
Detector calibration and the other release gates still apply.

Validate the real reference files before running the detector on an independent
split. `--check-references` checks hashes, geometry and review coverage without
inference. With `--require-final-quality` it returns a nonzero exit code while
the reference split is still incomplete. The report lists dataset, page and
mode issues, including missing measurements and unverified annotations.
Instance indices start at zero and follow the page's YOLO labels. The report
also names modes without recoverable verified examples in two source groups.

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
before exporting scores. Each eligible prediction gets its raw score and
a correctness label from one-to-one matching at IoU 0.85 with the reviewed
instance boundaries. Each fitted detector source needs at least 20 predictions
with both correct and incorrect examples. With calibration required, sources
without enough examples stay in review.

```powershell
python scripts/evaluate_onnx_segmentation.py --dataset-manifest path/to/calibration-albums.json --include-classical-fallback --calibration-output runs/crop-quality/calibration-scores.json --output runs/crop-quality/calibration-diagnostics.json
python scripts/fit_confidence_calibration.py runs/crop-quality/calibration-scores.json --holdout-id calibration-albums-v1 --output models/segmentation-calibration.json
```

Use the deployed detector, registry, optional refiner, OpenCV version and
detector settings for the fit and the release report. Calibration schema 2
binds the fit to that configuration. The release evaluator checks both album
identifiers and page hashes for overlap with the calibration sources.
If orientation is enabled, the configured orientation classifier must provide
a decision for an automatically eligible crop. Include
`--include-classical-fallback` for the default production configuration.

The local intake under `runs/crop-quality/calibration-intake-sources/`
contains eight views with 66 draft print outlines from four source groups,
including the [Allston album](https://nmaahc.si.edu/object/nmaahc_2009.31)
and the [Baxter leaf](https://nmaahc.si.edu/object/nmaahc_2021.13.13a-f).
The duplicate screen compares those views with the available training,
Golden and reserved pages. Both detector profiles produced 66 YOLO and five
classical edge-contour predictions. A single crowded page contributes 49 of
the 71 predictions. The draft remains pending independent review before fit.

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
the source album pages in the Prokudin-Gorskii collection. Three print scans
are staged in `runs/crop-quality/independent-intake-prokudin/intake.json`.
The corresponding page scans could not be retrieved during intake. The
catalog records nominal dimensions of 8 × 8 cm and the scan rasters include
material beyond the image. Both reference boundaries still need verification.
[Library of Congress digitization notes](https://www.loc.gov/pictures/collection/prok/digitizing.html)

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
uncertain aspect. The workflow keeps the rectified full preview and the
applicable geometry of both modes in review until the target is measured.

The independent test in `runs/crop-quality/plane-reference-agent/` covers
2200 px and 4400 px views with one-axis tilt. A 0.45 px perturbation of one
reference corner changes the estimated target aspect by 0.044%. A hinged
plane, small reference and contradictory measured dimensions all leave the
aspect unresolved. A wrongly measured aligned reference can still produce
a plausible rectangle, so the correction records which measurement was used
and keeps both downloads in review.

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
0.66 native album pixels across both crop modes. Aspect ratios agree within
0.10%. [Library of Congress catalog](https://www.loc.gov/pictures/item/2007675295/)

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

A development experiment zoomed into the bounded gap on the
[sparsely mounted Met page](https://www.metmuseum.org/art/collection/search/289066).
It found the central fifth print in the original scan at 0.904 mask IoU.
Its uncertain top edge left 9.6% of the draft footprint outside the proposal.
The new branch added no detections in 14 modified negative cases, but missed
all 13 resized, recompressed or geometrically transformed positives in the
robustness check. The diagnostic reports are in
`runs/crop-quality/dense-tiles-probe/gap-negatives-v2/`. The runtime keeps the
validated detection path while this case remains unresolved.
