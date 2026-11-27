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

The Doro Levi album conservation record identifies inventory 20 in
the historical photograph collection of the Florence Archaeological
Museum. Figure 3b identifies leaf 22 verso before treatment. The photo
archive records separate captures before and after lifting prints. The
online catalogue returned HTTP 403. Native files and a verified link
between that leaf and an individual paired capture remain outstanding.
The project agreement restricts reuse of the copies. This album remains
a source lead.
[Conservation study](https://faic.wpenginepowered.com/pmg-topics/wp-content/uploads/sites/9/2022/11/T18-142-150-Cattaneo-Formatted.pdf#page=4)
and [project agreement](https://bncf.cultura.gov.it/wp-content/uploads/2019/11/566.pdf#page=3).

JMU Libraries documents a second acquisition method. Its magnetic
album `VCCA004_001_009` was scanned with the mounted photographs,
then the separate set `VCCA004_001_009a` was scanned after removal.
Both catalog entries carry a Public Domain Mark. The file requests
returned HTTP 403 from the workspace proxy in the intake run.
Manawatū Heritage publishes CC BY 4.0 links to original TIFFs for
two physical Monrad albums, `A25/258 S5V2` and `A25/258 S5V5`.
Three tested download paths returned the same access error.
These remain calibration leads with the original bytes and
individual references pending.
[Mounted JMU album](https://omeka.lib.jmu.edu/erp/items/show/26520) ·
[Separate scans](https://omeka.lib.jmu.edu/erp/items/show/26521) ·
[Monrad S5V2](https://manawatuheritage.pncc.govt.nz/item/8b0880c5-897b-4b7d-b613-1afd9e9b082c) ·
[Monrad S5V5](https://manawatuheritage.pncc.govt.nz/item/4be99901-52e0-4dd7-a03c-44567b1f43a4).

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

The local user intake under `runs/crop-quality/local-calibration-source-draft/`
adds a source-review draft for calibration. It pins six original WebP files
from the supplied archives with 26 draft paper instances. The provisional
groups correspond to a white ring binder and a cream magnetic album. All
six image and label files pass the hash preflight. The byte and decoded
raster screen found no exact duplicates among 421 documented hashes.
The source-only review sheet awaits independent count and outline review.
Canonical album identities, usage rights and separation from the model's
training sources remain pending. Keep the packet in the calibration lane
and reserve other physical groups for final release evaluation.

```powershell
python scripts/evaluate_onnx_segmentation.py --dataset-manifest path/to/locked-album-manifest.json --include-classical-fallback --require-final-quality --visuals --output runs/crop-quality/real-release.json
```

Configure `CROP_VALIDATION_REPORT_PATH` with the passing report. A report for
a different checkpoint, settings or synthetic split cannot enable automatic
downloads. Manual confirmation can make a reviewed crop available. Removing a
report also revokes access to the automatic downloads that used it.

Generated reports and visual comparisons belong under `runs/`, which is
ignored because reports can contain local paths and private derivatives.

Use `--score-review-previews` to measure opaque quadrilateral review crops
with the same source-space ownership and interpolation metrics as ordinary
crops. Each fixture lists them under `review_previews`, linked to the matched
reference mode. The report records unmatched previews and transparent
outlines as unscored. Preview measurements are diagnostic. Release counts
continue to use the delivered `crop_modes` and their eligibility.

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
The public batch documentation maps item and resource JSON to declared
scan files. The exact pair IDs above can be matched through the MARC
mounted-page links before obtaining and hashing the intact TIFFs.
The `proka.0005a.08` record also links at least two other paper prints.
The individual images associated with `2018679192` and `2018679195`
were made from the glass negative. References to the physical prints
on that page are needed for complete release annotation.
The complete 2,610-row Free to Use bulk manifest omits both print
LCCNs and both corresponding volume LCCNs.
[Digitization notes](https://www.loc.gov/pictures/collection/prok/digitizing.html) ·
[LOT 10333 no. 128](https://www.loc.gov/pictures/item/2018679139/) ·
[LOT 10334 no. 54](https://www.loc.gov/pictures/collection/prok/item/2018679194/) ·
[MARC records for additional prints](https://www.loc.gov/pictures/item/2018679192/marc/) ·
[Second additional print](https://www.loc.gov/pictures/item/2018679195/marc/) ·
[Public image batch workflow](https://libraryofcongress.github.io/data-exploration/loc.gov%20JSON%20API/Accessing%20images%20for%20analysis.html) ·
[Free to Use bulk manifest](https://data.labs.loc.gov/free-to-use/manifest.html)

Princeton album `2016-546` has two public-domain catalogued leaf
rasters for accession `2016-546.3 a-b`. Its catalogue distinguishes
leaf, paper and image dimensions. The per-print measurements are
approximate. Native download requests returned 403 in the workspace.
The separate Smithsonian album `2020.10.3` is also catalogued with
public-domain media. A separately acquired raster of the same paper
print and a native page file remain to be matched. These entries
remain source leads beside the paired-geometry preflight.
[Princeton record](https://artmuseum.princeton.edu/art/collections/objects/129854) ·
[Smithsonian record](https://collections.si.edu/search/detail/edanmdm:nmaahc_2020.10.3?print=yes)

Third-party catalogues identify the exact Suna reproduction `02032` and
the Venice control print `02023`. The Suna catalogue links a JPEG
republication and attributes the TIFF master to the Library of Congress.
The workspace requests for that JPEG and the Venice archive index
returned HTTP 403. No native leaf and print files could be hashed there.
Figueres lists five physical Rubaudonadeu volumes and offers a Public
Domain Mark TIFF for volume I print `0000299293` / `J.M.C. 1167`, an
18 × 13 cm albumen print. Its TIFF request returned HTTP 403. A separate
native whole-leaf acquisition for that print is still unverified.
[Suna index](https://ps-spb2008.narod.ru/murm_suna1.htm) ·
[Venice catalogue](https://prokudin-gorsky.org/gallery/photo/1176) ·
[Figueres album catalogue](https://www.bibgirona.cat/regira/municipis/figueres/colleccions/album-rubaudonadeu)

A byte-level inventory against 228 pinned images checks 1,682 Git blobs,
10,747 workspace images, and 22 image members in 11 local archives. The
only matching Git assets are 17 known Golden or legacy JPEGs. The Git
LFS inventory is empty. Four Romanov whole leaves and five edition
rasters can be tied to their published Git blobs and local SHA-256
values. Their physical boundaries, print process and rights still
need verification. Seven published whole-leaf JPEG hashes match the
Kreis albums `SGV_10A_00031` and `SGV_10A_00050`. These published
derivatives use CC BY-NC 4.0. The separate print acquisitions and rights
for the required evaluation remain open. These checks added zero shared
verified release groups. Receipts are under
`runs/crop-quality/displaced-print-edges/reference-intake/`.

The Romanov edition records later colour scans of loose reproduction
leaves followed by web crops. The available files cannot verify a
separately captured, measurable photographic paper perimeter. The
Kreis collection publishes JPEG tiles under CC BY-NC 4.0 and offers
separate permissions and reproduction requests. Seventeen exact TIFF
names for `SGV_10A_00031` and `SGV_10A_00050` are cross-checked against
the archival indexes. The original rasters, the placement of each
separately captured print on its leaf and applicable rights remain
pending. This intake adds no verified release group. [Romanov edition](https://selinacochran.github.io/romanovball/#about-this-edition) ·
[Kreis archive usage and orders](https://www.ekws.ch/de/archiv/nutzung)

Columbia catalogs the Barbara Curtis Adachi collection's page scans
before album dismantling and preserves the unmounted regular prints
as a separate series. Album-page records `01-12` and `46-38` identify
different physical albums. The archive documents 400 ppi, 48-bit
page TIFFs captured before dismounting. The full public catalog at
`70c830d` contains 13,476 slide images, 1,750 album pages, 54 other
objects and zero individual paper-print captures. Its `images.json`
SHA-256 is `24506dc33a19361c7fd9aedf0ec8fcb2197f701ddeb94fe2360f57513fe8c461`.
The published JPEGs carry a nonprofit educational-use license.
The archives hold dismounted prints numbered on their backs. A
request would need the capture and placement of the same numbered
piece of paper, a use authorization, and a check excluding the
273 later impressions from negatives. The Ermakov archive
matches sales-album images and individual stock prints by negative
number. Those matches can be different impressions from the same
negative and cannot establish physical paper identity. [Columbia finding aid](https://www.columbia.edu/cu/lweb/eresources/archives/eastasian/adachi/ldpd.6226404.001b.html) ·
[Adachi page `01-12`](https://bunraku.library.columbia.edu/images/44986/) ·
[Adachi page `46-38`](https://bunraku.library.columbia.edu/images/46513/) ·
[Columbia image catalog](https://github.com/cul/ldpd-bunraku/blob/70c830d07d93cd22ed2010473a4635138025e456/_data/images.json) ·
[Columbia license](https://bunraku.library.columbia.edu/rights/) ·
[Ermakov conservation report](https://resources.culturalheritage.org/pmgtopics/2011-volume-fourteen/14-17_Maes.html)

Mountaineers accession `MTR.2018.3` describes scans of Edward K.
Triol's 1911 album pages and 34 high-resolution individual images.
Two individual motifs are published with attribution. The native
files, positions on the page, proof of separate capture of the same
paper, and photograph usage rights remain open. The HistoryLink
site license excludes photographs. The Smithsonian Arpee catalogue
identifies `FSA.A2011.03 A.24a` and `A.58b` as the same motif on
two separate mounted physical prints. The locally available Rudge
Red Album image `3REDA.JPG` and scan `REDA2.jpg` register with 224
inliers and held-out p95 1.32 px. The album window obstructs the
paper boundary and the rights remain unverified. This intake adds
zero verified shared release groups. [Triol collection](https://mountaineers.atlassian.net/wiki/spaces/ARCH/pages/11567109/Digital%2BPhotographs) ·
[Published Triol motifs](https://www.historylink.org/File/23121) ·
[HistoryLink terms](https://www.historylink.org/File/20001) ·
[Arpee finding aid](https://sirismm.si.edu/EADpdfs/FSA.A2011.03.pdf) ·
[Rudge individual scans](https://rudged.github.io/home/redalbumis.html)

The 2,048 × 1,536 Rudge spread `12REDA.JPG` has two separately
published matches, `REDA16.jpg` and `REDA17.jpg`. The bytes were
matched to archive revision `9422452`. The right individual scan
exposes motif and mount stock omitted by the album's horizontal
window strip. The 1,003 feature inliers have held-out p95 error
of 2.20 native pixels. A 455 × 50 native-pixel region in the strip
maps entirely onto the individual motif and covers about 4.56% of
the separate scan field. A second visible part of that same physical
print lies below the strip.

With the registered v37 hybrid development settings at proposal
threshold 0.25 and acceptance threshold 0.70, the spread yields
three proposals for two prints. The left and right matches have
mask IoU 0.810 and 0.685, below the 0.85 threshold. One extra
proposal covers the left album page and the right proposal misses
the lower visible part. Both right-hand crop modes stay in review
with boundary ambiguity, possible content cutoff and uncertain
aspect. The classical-only run proposes the left page and misses
the right print. These views support a development control for
occluded content and instance ownership. The Red Album has now
been inspected in development and remains outside calibration and
release. The source and replay receipts stay local under
`runs/crop-quality/rudge-red12-occlusion/`.
[Mounted and individual files](https://github.com/rudged/home/tree/9422452820380ca089db3e0baf0cfaf3dc6c7cdc)

The Roy Burn collection documents 43 matte 150 × 100 mm prints
on five magnetic album leaves. The digitization report records
whole-leaf scans before removal and separate TIFF exposures of the
prints in physical order. The public PDF paths returned 403 during
intake. Two individually identified records list CC BY-NC-ND 4.0.
The native scan hashes, verified motif boundaries and permission for
commercial reuse remain open. [Digitization report](https://www.aveburyhouse.co.nz/uploads/4/7/2/0/47203855/richmond_village-roy_burn_photos-metadata.pdf) ·
[Catalogue record](https://www.canterburystories.nz/collections/community/richmond-history-group/roy-burn-collection/ccl-cs-84777)

ÉBAF documents digitization of 302 paper prints from the Notre-Dame-
de-France albums in the Assumptionist archive. Project records
describe scans after physical removal and subsequent thematic
cataloguing. The public finding aid leaves the original print-to-
leaf crosswalk and separate native leaf exposures unverified.
Access and reproduction require authorization from the community.
No usable native pair with a proven split assignment is available
from this intake. [ÉBAF collection history](https://www.ebaf.edu/the-photographic-collection/history-of-the-funds/) ·
[Assumptionist finding aid](https://archives.openjerusalem.org/index.php/archives-of-the-community-of-the-augustinian-fathers-of-the-assumption-in-jerusalem)

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

The local preflight at `5d5d255` checked 99 submitted image-file and
decoded-raster hashes against 672 registered digests with no exact hits.
The registered historical model still lacks reviewed training provenance.
U005 and U042 have pinned source rasters and labels, with 12 unverified
mode references between them. Both U042 motifs lack an independent aspect
reference. Seven pinned archive rasters were unavailable across the
Talbot, Gardner, Quine and Whiting drafts. This preflight adds zero
verified physical album groups. Its source-only receipt is under
`runs/crop-quality/recentered-rotation-review/reference-preflight.json`.

The Priscilla Smith Album 8 documentation identifies the same five
physical cards before and after removal from the mounted leaf. The
separate front and back views and the named leaf positions provide a
potential link for motif landmarks. Album windows cover parts of the
paper boundaries. Volume I describes the leaf and cards on printed
pages 103–104. The repository index locates the actual figures 3.25–3.27
on printed pages 57–59 of Volume II. The CORE copy links to the text
volume. The image volume and its reproduction terms remain unverified.
The workspace received HTTP 403 when resolving the repository handle.
This source has no assigned dataset role.
[Volume I documentation](https://fileserver-az.core.ac.uk/download/360554527.pdf) ·
[University repository record](https://hdl.handle.net/10523/7504)

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

The conservators confirm separate front and back digitization of
each of the 44 cartes de visite after removal, and digitization of
the album as a physical object after treatment. The Smithsonian
records the Tubman image as 94 × 57 mm and the paper mount as
100 × 62 mm and publishes the stand-alone asset under CC0. The
checked native Smithsonian requests returned HTTP 403 in this
workspace. The earlier Tubman pair remains a development example
for unrecoverable content in an album window.
[Conservation report, p. 83](https://faic.wpenginepowered.com/pmg-topics/wp-content/uploads/sites/9/2024/08/T19-72-85-Formatted-Evers.pdf#page=13) ·
[Smithsonian asset record](https://avpreservation.si.edu/object/nmaahc_2017.30.47)

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

The latest local cache and archive audit recovered zero of seven pinned
Whiting TIFFs. For the `15831`/`15830` pair the existing registration
is compatible with a resampled extraction. The `13708`/`13709` catalog
and stored timestamps support separate acquisition. The additional LoC
record links back to the same asset series, and Commons has a JPEG
derivative of print `13709-00055`. A native-grid comparison remains
necessary to verify either pair.
[LoC cross-reference](https://www.loc.gov/item/mamcol000128/) ·
[Individual print derivative](https://commons.wikimedia.org/wiki/File:GermanStnAbouAugeileh00055v.jpg)

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

`getty-talbot-measured-review-draft.json` pins two native Getty images. The
catalogue's Image dimensions are 181 × 170 mm and 203 × 166 mm. The
provisional source fits differ from those aspect ratios by 0.26% and 0.06%.
The dimensions are recorded only for the motif drafts. A verified paper
measurement or separate reference scan is still needed for both whole
prints. The earlier provisional coincidence of both boundaries remains a
visual hypothesis. The source preflight needs both pinned `image_url`
assets at the paths in the manifest.
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

Visual Ancestor catalogues separate physical albums `10162` and
`10410`. The records place print `10162-127` on leaf 116 and print
`10410-103` on leaf 104. The archive describes scanning pages and
unmounted prints separately and marks the linked print records as
public domain. The native leaf request returned HTTP 403. Separate
acquisition for these exact files, the complete visible paper
edges and the specific page raster rights still need verification
before either album can enter the reference split.
[Print 10162-127](https://www.visualancestor.com/image/?group=10162&image=10162-127) ·
[Print 10410-103](https://www.visualancestor.com/image/?group=10410&image=10410-103) ·
[Archive method](https://www.visualancestor.com/about/)

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

#### Compare the visible capture corner in the original

The count review on desktop and mobile now offers a bounded view of the
original source around an ambiguous capture corner. Two observed traces
must share an interior endpoint and each run to a different frame axis.
The UI maps the normalised coordinates back to native pixel centres and
adds a margin of 8% of the shorter span, with a minimum of eight native
pixels. It clips this rectangle to the source image. The two measured
guides stay open at the frame. An optional layer shows the current
linked proposal contours in violet and other active contours in orange.
The underlying image is the same authenticated original used by the
main count editor.

On the 180° NerdScan scan (`d01616fb…`, 2179 × 3000) the view starts at
`(0, 2035)` and spans 1222 × 965 native pixels. This includes the visible
fitted footprint and the adjacent context. An unrelated active contour
intersects 15.65% of that footprint. The comparison therefore uses the
original as captured, including the disputed pixels. It leaves the
individual proposals, count and stored crop geometry as they are. It
creates no rectified variant. Both crop modes and the legacy trimmed
download remain held for the linked proposals.

The native-window regression covers reversed trace direction, the
opposite capture corner and invalid or disconnected traces. The separate
native replay also checks all four rotations and active-set changes.
After the reviewer explicitly excludes one linked proposal, the view
can still show the remaining two. With only one active member the shared
comparison closes. This is a review aid for an unresolved detection. The
180° development replay remains at two matched prints out of three
with four extra proposals.

The exact multipart upload was replayed through desktop and mobile
Chromium. Each view loaded the same pinned original and traced two open
sides, with three violet and three orange proposal outlines available.
Opening the view and toggling the layer made no write requests and
preserved all six masks, both modes' geometry and protection records and
the review JPEG digests. Both named downloads and the legacy trimmed
alias returned 404. A separate explicit rejection reduced the count from
six to five. The five surviving rows kept their existing render records.
The local browser and API receipts and screenshots are under
`runs/crop-quality/capture-corner-original-review/`.

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
the existing acceptance criteria. The mixed NerdScan replay at that
revision cuts off 35.3% of the draft footprint at one clipped print.
Both modes remain in review.

On the same seven source and label hashes, a matched code replay
compared `edf1287` with `cb290c7` at native mask IoU 0.85.
Both mixed arms used `1e95a1a9…`, 512 × 512 inference, proposal
threshold 0.25 and acceptance threshold 0.70. The original
nine-view cohort still lacks its source and label files and
registered checkpoint. Its reported 22 of 28 matches and 17
extra proposals stay separate from this comparison.

| Seven development views | Matches | False proposals | Misses |
| --- | ---: | ---: | ---: |
| Mixed at `edf1287` | 23/24 | 3 | 1 |
| Mixed at `cb290c7` | 24/24 | 2 | 0 |
| Classical at `edf1287` | 18/24 | 16 | 6 |
| Classical at `cb290c7` | 22/24 | 8 | 2 |

All eight initial mixed model records and first fused selections
matched, including a separate contact-sheet stress case. In
NerdScan the upper print starts at 0.92093 model mask IoU. Fusion
selects a 0.64749 contour on both revisions. The current
capture-edge recovery brings it back at 0.99900. Two ambiguous
proposals remain active in review. On the JPEG75 contact-sheet
control both revisions still show zero matched prints and five
active nonmatches. Its best active outline scores 0.83174, while
the observed paper envelope offered in review scores 0.98862.
The scan does not establish which competing parts share one paper.
[Per-source hashes and counts](development-detector-replay.json)

The mixed replay at `4bbf7d5` used the same native sources, label hashes,
model and thresholds. The seven views again match 24 of 24 prints at
instance IoU 0.85. Both extra proposals are on the clipped NerdScan scan
and require review. The other six views have no extra proposals. Precision
on the seven views is 92.3% and recall is 100%.

The separate JPEG75 contact-sheet case again misses the one annotated print
and leaves five nonmatches in review. The best counted mask has IoU 0.83174.
Against the draft full-print footprint, its regular warp omits 16.77% of
the annotated content. The alternate paper outline in review has a crop
footprint IoU of 0.98702, omits 1.22% and retains 0.10% foreign area. The
alternate outline does not resolve physical paper ownership.

The evaluator ran both crop branches on all eight inputs with
`--score-review-previews`. The development annotations supply draft paper
footprints. They contain no verified crop geometry for either mode, so
they cannot measure production eligibility, motif content loss or native
landmark and aspect accuracy. Decode through both crop branches and JPEG
writes took a median of 4.59 s and p95 of 15.34 s per input on the single
CPU-thread development run. Local reports are in
`runs/crop-quality/real-development-refresh/`.

The paired `8241e4c` rerun reproduces all eight inputs' ordered final
mask geometry, instance measurements, both modes' measurements and
separately scored review previews from `4bbf7d5`. The seven-view result
is still 24/24 with two extra proposals. The separate JPEG75 case still
has no match and five active nonmatches. Across the eight inputs, 30
of 31 whole-print rows and all 31 motif rows are held. One whole print
is locally ready. None has a verified automatic release. The eight
decode-to-JPEG samples take 9.98 s median and 30.82 s p95 under concurrent
host load, including possible model initialisation on the first view.

On JPEG75, the raw whole-page model mask reaches 0.8224 IoU against
the drafted paper footprint. Three quarter turns of the exact decoded
source reach 0.7900–0.7906 after remapping. Twelve source-window probes
use margins around the existing measured paper perimeter or around its
perspective-normalised view. The best recovers 0.4688 IoU against that
measured perimeter. The JPEG55 and lossless copies show the same local
mask failure. The model often selects a strip of image panels. The
existing source comparison is still available and includes the
handwritten heading. These probes do not support enabling a rotated or
closer inference retry. Receipts are under
`runs/crop-quality/real-development-refresh/8241e4c/`.

A separate contrast probe on the same reduced contact captures varies
only the source pixels outside the measured paper outline. In the
strongest JPEG75 trial, the taper brightens up to 56 code values across
a 120-native-pixel outside band. The raw mask crosses the draft IoU
threshold, from 0.82243 to 0.85273. Of 2,162 sampled dark heading centres,
it still retains only 1,589, up from 558. The sample uses gray values up
to 150 inside the same original-source window `(760, 135)–(1208, 177)`,
above the first image rail. This is a controlled input alteration. The
ordinary detector and crop geometry keep their original inputs. The
probe ran 31 settings per encoding on JPEG55, JPEG75 and lossless PNG.
Reports and scripts are in `contrast-probe/` under
`runs/crop-quality/contact-paper-edit-first/`.

An independent authored proof set varies heading contrast and upper paper
margin while keeping the underlying grid, camera and sleeve fixed. It
contains four independent scene roots, 24 count-scored cases and four
optically identical flush aliases held outside count scoring. From
4080 × 3072 it makes a 33% reduction and records exact owner IDs and
separate heading, rim, motif and sleeve masks. All 420 artifact hashes
and 1,484,629 sampled inverse-projected owner centres were checked. The
pinned mixed raw detector at 0.25 matches 4/28 prints in JPEG75. None of
the eight truly separate prints or four wider-rim sheets match. One
joined sheet passes at IoU 0.872 and retains 6.06% of its strong heading.
That case demonstrates the need to score heading retention separately
from instance IoU. This authored cohort is reserved for development
comparison, outside all training and release references. Its manifests,
generator, verification and result are under `header-proof/` in the
same local report directory.

The previously tested sparse and dense arms both find all 20 joined
JPEG75 sheets even at confidence 0.70. Each finds two of eight separate
prints and has 15 extras. At 0.25 the sparse arm finds 22/28 with 22
extras. The dense arm finds 24/28 with 37 extras. The existing
architecture-only training run produces no proposals above 0.25 on the
cohort. These are raw masks. The earlier normal-path and two-mode
benchmarks still rule out both old transfer arms. The new test isolates
heading and rim retention and keeps the exposure of adjacent prints as
a separate measurement.

#### First development replay of a newly supplied spiral album

The album group contains five views, including the illustrated front cover.
Reserve the entire group for development. Use the manually drawn, source-only
outlines on one open spread with eleven separate prints. The front cover has
zero mounted prints. The draft outlines pin the original rasters, distinguish
the paper from the foil background and carry no verified dimensions.

| Detector at native mask IoU 0.85 | Spread matches | Spread extras | Cover extras |
| --- | ---: | ---: | ---: |
| Mixed development checkpoint and classical evidence | 1/11 | 2 | 4 |
| Same checkpoint without classical evidence | 1/11 | 2 | 4 |
| Classical only | 0/11 | 0 | 1 |
| Forced retry with the existing four 70% tiles, detector-stage experiment | 6/11 | 1 | 4 |

The two unmatched mixed proposals each combine a pair of distinct prints.
The classical channel provides no proposal on the spread. The full-frame
pass returns three masks, which falls below the existing dense-tile trigger
of sixteen. Forcing this retry supplies ten tile proposals and takes
0.54 s for the tiled detector stage on the development CPU. The cover
produces no new tile proposals and keeps its four false proposals. Both
images retain all seven original mixed proposals in review across the
upload and reprocessing checks. Untrimmed previews are available and both
crop exports are held. The upload check used lossless PNG copies whose
working pixels were identical to the WebP sources.

Keep the tile experiment isolated until tests distinguish adjacent paper
from a common print, including multipanel prints and decorative covers.
The provisional album group, original images, annotations and per-image
reports are stored locally. No private source material is checked in.

Two additional spreads from the same album were annotated before running
the detector on them. Across the three spreads the full-frame detector
matches 6 of 35 prints with three nonmatches. The forced four-tile probe
matches 20 of 35 with one nonmatch. The development evaluator with both
crop branches also matches 6 of 35 and emits five nonmatches on the three
spreads. The tile probe scores detector output before the crop stages.

One tile can cover about half of a merged proposal and still suppress the
whole proposal in dense-view selection. On the first spread this loses
the unresolved partner on the right. A left-hand pair covers 92.49% of
its parent with two disjoint candidates. Keeping the right parent and the
unrelated full-frame mask yields three of eleven matches in this detector
counterfactual. Source comparisons on two spreads follow a continuous
ridge in the exposed album material between paired prints. Matching colour
alone also accepts a synthetic same-colour seam. The brightness pattern
across the ridge distinguishes those originals from the flat and
phase-shifted controls at the sampled positions. These results were
obtained on one physical album with selected measurement bands.

A source-derived reproduction puts both motifs and the original album
texture on one new physical sheet. It preserves the complete sampled
ridge and passes the same local check. A separate experimental rim check
recovers the quad for the light-paper reproduction. It fails to recover
the dark-paper rim when it must find the outline itself. A local separator
or failure to find a rim therefore does not establish distinct paper.
The forced tile probe preserves all 17 full-frame matches on the locked
synthetic detector subset and removes one extra proposal in the overlap
case. The original miss on that overlap remains. The synthetic count
check does not cover the ambiguity exposed by the reproduction.

#### Missing prints in count review

A risky full-page pass with three to eight masks can run the existing
four overlapping inference tiles for additional suggestions.
The detector keeps the full-page pass and its normal fusion. Tile candidates
need support on all four sides in the original pixels, including three
strong unambiguous sides. Any intersection with another paper claim above
one native square pixel rules out the suggestion. The check includes the
initial masks, final detections and open count-review evidence.

On the first spiral spread the extra pass took 0.54 s and found one
previously missing print with 0.98181 IoU against the draft outline.
The other two spreads and cover added no count suggestions. Existing
active detections and crop results remained unchanged. A reviewer can
confirm the new proposal as a separate paper print. The server checks
the current count revision, the offered outline, the edited corners and
the current full-print claims before accepting it. The real upload replay
retained three active detections until confirmation. After confirmation and
photo reprocessing, the untrimmed preview was available and both crop
exports remained in review.

The eight frozen and development comparisons preserve the prior detection
and crop metrics in both modes. The frozen detector matches 18/18 prints.
The seeded geometry set passes 11/11 whole prints and 12/12 motifs.
The detector profile binds the new tile review to revision 3. The spiral
results use the development checkpoint and draft geometry. Local receipts
are under `runs/crop-quality/tiled-print-review/`.

On a later spiral page, the full-frame model and classical stages miss
both rounded prints beneath a mounted souvenir card. Three model masks
claim parts of the fabric at the capture boundaries and fail the existing
aspect-ratio check. The card's upper photo and printed caption require
separate ownership review.

The tile review in revision 4 requires at least two geometrically valid
full-frame masks covering 2% to 20% of the capture each. A larger card
proposal and invalid edge proposals can coexist with these masks. A
curvature finding can support an additional count suggestion when all
four sides have unambiguous native support of at least 0.90. The existing
review route still requires confirmation and holds the proposed crops.

On the souvenir card page the retry takes about 0.40 s and finds the
previously missed lower-left print at 0.98318 IoU against the draft. The
full-size JPEG75 and 75%-size JPEG75 derivatives retain the same offer at
0.98302 and 0.98264 IoU. A quarter turn instead recovers both lower prints
as active detections at 0.95658 and 0.91895 IoU on that derivative. The
offered print and the automatically detected instances are scored
separately. The upload replay confirms that the offer leaves all five
original proposals in place. After confirmation
and photo reprocessing, the preview is available and both export modes
remain held. The lower-right tile candidate still fails native edge
verification. The original cloth and card proposals require review.
Local receipts are under `runs/crop-quality/spiral-card-recovery/`.

#### Confirm a missing print across opposing quarter turns

Revision 5 runs the 90° and 270° full-page views after the tile pass
finds an additional outline. The masks map back to native pixel centres.
Each candidate needs a unique partner from the other view at mask IoU
0.80 or higher. Both masks need a model score of at least 0.55, and one
must reach the configured acceptance threshold. The original raster must
support four unambiguous sides at 0.90 or higher,
and the fitted outline must reach 0.84 IoU against each view. If both
native fits succeed, their corners must agree within four source pixels.
Current and pending paper claims still veto overlapping suggestions.

On the original souvenir card capture, both lower prints now appear
in count review at 0.98318 and 0.98407 IoU against the draft masks.
The possible caption card and fabric proposals still require review.
The earlier spiral spread keeps its single offer. The cover adds none.
Both JPEG75 derivatives retain both offers. The right print reaches
0.98407 IoU at full size and 0.98334 at 75% scale. In the upload replay
each extra print needed its own confirmation, and reprocessing the
added right print kept both crop exports held.
With three warm single-thread CPU samples per arm, median time in the
tile review stage on this capture rose from 0.81 s to 1.19 s.
The eight comparison runs retain the previous detection, crop and
count-review results for their existing inputs.
Local receipts are under `runs/crop-quality/opposing-print-views/`.

#### Confirm a weak tile mask from a centered view

Dense tile review revision 6 can retry one otherwise unclaimed tile
winner per eligible page. The first mask must have confidence in
`[0.65, 0.80)`, boundary score at least 0.80 and a supported native
paper outline. The second view uses 56% of the page width and 60%
of its height. It is centered on the mask bounds and clamped to the
page. The same detector must return one matching mask at confidence
and boundary score at least 0.80. A second mask with IoU of at least
0.65 to the first proposal makes the match ambiguous. The unique
match needs at least 0.90 mask IoU and four unambiguous source sides
at support 0.90 or higher. Its native outline must reach 0.85 IoU
against both masks and agree with the first native fit within four
source pixels at each corner.

Development spread U039 has SHA-256
`d82b6a755c4dfa72a98a3a68251458e6b658b7db57140b40ee00e55adbfedebd`.
The registered development checkpoint has SHA-256
`1e95a1a90ff897df7099ff80675186ec1d8145f7bb07559fe918b3dd3a6db748`.
The tile score rises from 0.692 to 0.912.
The masks reach 0.968 IoU and the fitted corners differ by at most
3.44 native pixels. The bottom print next to the binding on the left
leaf appears in count review at 0.97435 IoU against the draft.
Existing active detections and earlier count offers keep priority.
Post-fusion validation still vetoes any overlap with current or
pending paper claims. A separate missed print on U039 remains held
back by a crossing classical contour.

The same new offer appears at full size after JPEG75 compression.
The JPEG55 derivative has only two eligible full-page model
proposals, so it does not start the existing tile review. The
75%-size JPEG75 derivative also adds no centered offer. All five
original spiral captures keep the same active detections. The other
four keep their previous count offers. A separate single-thread CPU
replay measured the new view and verification at 178 ms on the
original and 257 ms on JPEG75.

The upload and reanalysis replay reproduces the offer. Adding the
print requires current count revision and explicit confirmation.
Reprocessing provides an untrimmed review image. Whole-print and
motif downloads both remain in geometry review. All eight frozen and
development comparisons keep their detection and crop metrics in
both modes. These measurements use the registered development
checkpoint and draft annotations.
Local receipts are under `runs/crop-quality/centered-tile-review/`.

#### Refit an unmeasured tile from two closer views

Dense tile revision 9 reserves the first successful ordinary review
path. If those paths yield no offer, it can retry one weak tile winner
whose original source fit failed. The raw seed must be unclaimed, have
model confidence in `[0.65, 0.80)` and a boundary score of at least 0.80.
The retry uses centered windows of 56% × 60% and 48% × 53% of the page.
Both follow the normal margin check at an artificial tile boundary.
Each must return exactly one plausible match to the seed at mask IoU
0.65 or greater. Acceptance requires confidence and boundary score at
least 0.80, 0.90 IoU to the seed and 0.90 IoU between the new masks.

Both views must yield original-pixel fits with at least 0.90 support
on every side and at least three unambiguous sides. The fitted corners
must agree within four native pixels. Each fit must overlap all three
masks at 0.85 IoU or higher. The final fusion checks current and pending
paper claims again. The result uses the existing standalone print
confirmation and remains a count suggestion until that confirmation.

On U040 the registered development model first returns a 0.779 seed
on the lower inner print of the right leaf, which fails the ordinary
source fit. The two centered views raise the model scores to 0.868
and 0.914. Their fitted corners differ by at most 3.11 native pixels.
The lower rail remains ambiguous. The new suggestion reaches 0.96410
mask IoU against the development draft on a 1024-cell comparison
raster. All five existing spiral captures keep their active detections
and previous count suggestions. U040 adds this one suggestion.
At native dimensions JPEG quality 90 also produces the suggestion
with 0.96585 draft IoU. JPEG qualities 55 and 75, JPEG 75 at 75% scale
and the two quarter-turn derivatives produce no new suggestion. The
recorded active masks match the baseline for all seven U040 inputs.
In five warm native runs on a shared test host, median time in the
tile stage was 0.74 s at `619ad65` and 1.09 s with the retry.

The exact decoded U040 pixels were uploaded through the ordinary
PNG upload path. Repeat analysis retained four active rows and one
pending offer. The current count revision and explicit confirmation
were needed to save it as the fifth row. The baseline review image
was available. The full-print and motif downloads and cropped alias
remained held after save and reprocessing. Both creating a manual crop
without linking the offer and moving an active row to its contour now
return HTTP 409. The check uses the same 55% proposal coverage and 75%
submitted outline coverage as explicit acceptance. Both attempts on a
three-native-pixel inset left the active rows and count revision intact.
The independent native probe also finds a plausible reproduction on one
paper carrier. The paper count remains a reviewer decision. The draft
supplies a visible paper outline. Independent aspect and motif references
are still outstanding for this print.

The eight frozen and development comparisons preserve detection and
both crop metrics, including 18/18 locked synthetic detections and
11/11 whole-print plus 12/12 image-only recoverable synthetic crops.
The local receipts are under
`runs/crop-quality/recovered-centered-tile-review/`.

#### Check a failed tile fit against ribbed backing

Dense tile revision 10 checks one additional candidate after the
ordinary tile review paths. It takes the highest-confidence unclaimed
tile seed with confidence and prepared boundary score both at least
0.80 whose ordinary source fit failed. The shortest initial side must
measure at least 180 native pixels. The method uses the original
decoded raster and adds no model inference.

The source check looks for a 5-pixel-wide strip outside the seed.
The strip must have a coherent warm Lab palette for at least 82% of
its sampled stations and 68% of each eighth. Its high-pass luminance
must repeat at a lag of 4–12 native pixels with at least 0.52
correlation and have matching phase across the strip. Existing paper
claims exclude reference strips. Admissible palettes more than five
Lab AB units apart make the evidence inconclusive.

The reference palette determines the outside of the four edge fits.
Each fit searches at most 16% of the shortest seed side, capped at 44
pixels, with a signed colour or lightness change. Every edge needs
90% support overall and 80% in each eighth. A separate rib must pass
along one of the two sides adjacent to the reference side. The rib
must also be measurable beyond both ends and match the reference
colour. The fitted outline must overlap the original mask at 0.85 IoU
and stay inside the capture. The raw and fitted outline, reference strip
and a five-pixel corridor on each side of the rib trace must clear the
final active and pending print claims and existing alternative outlines.
The ordinary suggestions are decided first. The ribbed backing check
leaves those suggestions intact.

On U039, the registered development checkpoint yields an unclaimed
seed 0.855 with a prepared boundary score of 0.8233 for the car
photograph at the outer right of the middle row. The source choice
uses the strip to its left at Lab AB [129, 144] and the continuous
rib below it. The recovered outline has 0.96592 IoU to the development
draft. The adjacent car photograph remains separate. Two plausible
lower transitions in the independent source probe differ by over four
native pixels at one end. Shadow, flat printed gold, and continuous
car imagery replacing the gap fail the source checks. A physical
sheet reproducing those same source pixels passes them. The new
outline is a count suggestion and requires explicit confirmation.

Normal upload of the exact decoded raster and repeat analysis preserve
the four previously active rows and the two pending suggestions. Adding
the new suggestion requires the current count revision and standalone
confirmation. The previous pending suggestion remains available. The
untrimmed review image renders successfully. Both named downloads and
the legacy trimmed alias remain held after reprocessing. No new offer
for this print appears in the full-pipeline JPEG 55, 75 and 90 runs,
the JPEG 75 downscaled run, or the two lossless quarter-turn runs.
Source-only fixed-mask replays and end-to-end detection are tracked
separately. A separate tile inference probe still sees the target on
the three full-resolution JPEG derivatives at confidence 0.836–0.842
and mask IoU 0.856–0.865. This bounds the remaining end-to-end failures
to the tile selection or subsequent source and ownership checks on
those derivatives. On JPEG 75 and JPEG 90 a competing full-frame mask
includes more than 96% of the target draft and more than 99% of the
similar car print above it. On JPEG 55 a classical contrast proposal
includes more than 96% of the target and crosses three other drafts.
The same probe sees an approximately 0.93 IoU
tile mask at confidence 0.749 after the 90-degree turn. The 270-degree
turn yields only a partial full-frame proposal. In nine warm local
measurements on a shared host, median time for the added native source
check was 195 ms. In the five original spiral captures, U039 adds one
count suggestion. All recorded active polygons and earlier pending
suggestions stay unchanged. The eight frozen and development reports
preserve the detection and both crop-mode metrics, including 18/18
locked synthetic detections and 11/11 whole-print plus 12/12 image-only
recoverable synthetic crops. The receipts are under
`runs/crop-quality/ribbed-backing-tile-review/`.

#### Review a tile outline blocked by a capture frame contour

Crossing outline revision 4 compares a previously verified tile
with a single active classical contrast claim after fusion. The
classical outline must come from one minimum-area rectangle
proposal and reach two axes of the capture frame with two
vertices. An overlapping pending claim, multiple active owners
or multiple eligible alternatives to the same claim leave the
comparison closed.

The four classical sides must have native support at most 0.40
without a competing local edge. The tile is checked within four
native pixels of its measured outline. Each side must have support
at least 0.90 and the corner positions must stay within four
pixels. One old side must run through the tile at least 13 pixels
inside it over at least 96 pixels and 45% of the tile's shorter
side. The old side must still lack a supported paper transition
in that span.

On the original U039 capture the classical overlap is about
7,622 native square pixels. The old four side supports range from
0.228 to 0.361. Its diagonal runs through the measured print for
about 149 native pixels at the required interior distance. The
alternative reaches 0.97875 IoU against the draft paper outline
on the 1024-cell comparison raster.
It appears on the existing reviewed row while the separate offer
below it is still available in count review. The U040 overlay
comparison has three supported classical sides and does not enter
this review path. A near-flush overlay can still require a
separate source interpretation. The native check took 57 ms
median and 66 ms at p95 over 12 warm one-thread samples on U039.

The upload replay replaces the U039 outline on the same ID. A
generic approval and an unchanged geometry save both return
409. Current neighboring and pending contours veto overlapping
corrections. The add route rejects a duplicate of the proposed
or corrected contour at IoU 0.90 or higher. A repeat page analysis
keeps the corrected ID and the same active count. Photo reprocessing
keeps the whole-print and motif downloads in review. The frozen
and development comparisons keep their previous crop and detection
metrics in both modes. The annotated source and local receipts are
under `runs/crop-quality/contested-tile-review/`.

#### Show open extent guides on a partial print

Dense tile revision 7 can retain one secondary tile nomination for
extent review. A strong tile candidate whose raw mask overlaps one
earlier model claim by at most 1.2% enters this path only if its
native fit is clear of the earlier claims and agrees with its mask
at IoU 0.85 or higher. The nomination never becomes a count offer.
After fusion it must contain at least 95% of exactly one
minimum-area classical edge contour, which must account for
35% to 70% of the fitted tile. Other current and pending claims
veto the comparison.

Three classical sides must align within four native pixels of the
corresponding tile sides. The fourth must pass through continuous
image structure without a supported transition along the proposed
cut. Two original-pixel fits must extend the lateral sides and agree
with the inner rail projections within two pixels. The terminal search
must yield two distributed transitions 2.4 to 8 native pixels apart.
The sample just inside only one transition must be darker and cooler
than its exterior. An aligned overlay can reproduce image continuity
and a printed band can reproduce this color change, so the four open
guides leave ownership and the exact crop to review.

On development spread U040, SHA-256
`17898ad4cd68da2ff3440c117790351cb0bb51a3612da4e1a7c93aa7a400e706`,
the delivered guides span 105 and 101 native pixels on the sides
and about 263 pixels along each terminal transition. The terminal
paths differ by about five pixels. The existing upper transitions,
stored warp and four active proposals are preserved. The optical
check retains all four lines across twelve fixed-geometry quarter-turn
JPEG controls at qualities 55, 75 and 95. An inserted album gap, a
different flush scene and two false rib-shadow controls fail that
check. Aligned same-art overlay controls can still produce guides.
The standalone native check took 44 ms median and 79 ms at p95 over
12 warm single-thread samples during concurrent tests. In the
end-to-end replays, Q55 and Q75 re-encodes and the 75%-size Q75
derivative supply no new extent guide through the detector.

The normal upload and reprocessing replay verifies that both exports
and both rendered comparisons remain held. The untrimmed review image
is available, and a generic geometry approval returns 409. A later
overlapping paper claim hides the four guides. Reprocessing with
the precise crop setting disabled still retains the stored hold.
Measurements and the annotated example are under
`runs/crop-quality/partial-print-extent-review/`.

#### Compare two possible prints within one count proposal

Dense tile revision 8 examines one unique whole-page proposal with
model confidence in `[0.40, 0.65)` and exactly two contained
tile winners. Both tile scores must be at least 0.65 and both edge
scores at least 0.80. At least one tile score must reach 0.80.
At least 94% of each raw mask must lie within the parent. The two
masks must have IoU at most 0.01 to each other and together cover
80% to 97% of the existing proposal at nomination.

The original raster must support both perimeters. The side search
uses exposed backing color from the ends of the gap and checks
signed color differences, local edge strength and distribution
along each side. The two fitted outlines must remain disjoint.
At least 96% of each fitted outline must lie inside the original
claim and their combined coverage must be 78% to 95% of it.
PhoSca then follows a measured rib through the gap at the phase
observed independently at its ends. Overlapping active or pending
paper claims veto the outlines and the separator.

The left proposal on U008, SHA-256
`e0f64096b018b54fcd3827d2a13c9eeed40b02f499bf890f924da2bd34ce7eb6`,
keeps one active row. Its upper and
lower guides reach 0.94313 and 0.97679 IoU against the development
drafts on a 1024-cell comparison raster. Eight fixed-geometry
quarter-turn JPEG controls retain the source cue. A copied scene,
a flat separator in the same color and a shifted rib pattern
fail the source check. Reproductions on white and dark single
paper stock can preserve both perimeter and rib evidence. The
source cue therefore offers a visual comparison while the physical
count remains a reviewer decision.
The source check took 176 ms median and 220 ms at p95 over
12 warm single-thread samples during concurrent tests.

The normal upload and reprocessing replay shows the pair alongside
the original parent outline, preserves three active records and
keeps the separate pending offer. Both crop downloads stay held.
The API rejects adding either shown outline as a standalone print
while the shared proposal is open. A new conflicting paper claim
hides the comparison and leaves the hold in place. Reprocessing
with precise crops disabled retains that hold. The end-to-end
JPEG55 and JPEG75 re-encodes of U008 did not show the comparison.

An explicit manual save of the upper outline succeeded. When
adding the lower outline as a new print, the API returned
`protected_region_cut`. A face detector candidate covers foliage
on the next print and starts at native y=706. Its 19.8-pixel
protection margin reaches into the proposed print. The development
draft for the same print triggers the same guard. That correction
sequence was blocked at this revision. The follow-up adds an explicit
[margin review](#review-a-protection-margin-outside-a-manually-added-print).

The annotated source and receipts are under
`runs/crop-quality/merged-print-pair-review/`.

#### Measure a possible pair with one separate tile mask

Dense tile revision 11 can also compare two possible prints when only
one reliable individual mask survives ordinary tile inference. A
whole-page mask must have confidence 0.25–0.38, valid geometry and
cover 4.5–12% of the page. The one tile mask must score at least 0.80
for both confidence and prepared boundary support. At least 90% of
the tile mask must lie inside the parent and it must cover 35–62% of
that parent. An ordinary pair with two masks keeps priority. Multiple
eligible parents or tile masks give no new comparison.

The original decoded raster is checked without another inference pass.
The parent must have a short-to-long side ratio of 0.42–0.70 and a
short side of at least 200 native pixels. The single tile occupies
one end of the parent. The method searches across the middle 43–57%
of the long axis and trims each end of the search chord by 3.5%, 7%
or 10.5%. This lets the original rib matcher sample the exposed album
backing when one side of the parent is close to fabric. At least two
measurements must agree on the same rib to within 1.75 pixels when
projected to common endpoints. Conflicting palettes or rib positions
leave the comparison unresolved.

The other seed uses the parent corners at its far end and the rib at
its near end. The original pixels must support all four signed backing
transitions for both seeds. The fits must be disjoint, with at least
2.5 pixels between each facing line and the crest and a minimum
combined clearance of 7.5 pixels. The crest must continue beyond both
children and fall off on both shoulders inside the gap. A common stock
outline, a conflicting print claim or less than 0.84 fitted-to-raw IoU
for the tile vetoes the comparison. The existing parent containment
and combined coverage checks apply.

On U039 both full-resolution JPEG75 and JPEG90 show two candidate
outlines on the existing car-photo row. Their IoU to the development
drafts is 0.9500 and 0.9646 for JPEG75, and 0.9450 and 0.9651 for
JPEG90. The source check took 467 ms and 481 ms median over six warm
runs per encoding on the shared test host. Flat printed colour, a
phase-shifted rib and a foreground print covering part of the gap
fail the source check. Reproductions on a white or dark paper carrier
that preserve the entire local scene still pass it. The original five spiral
captures and the prior U039 native suggestion retain their recorded
active and pending polygons. JPEG55, the downscaled JPEG75 variant
and the end-to-end quarter-turn derivatives yield no new pair. The
eight frozen and development comparisons preserve detection and
both crop-mode metrics.

In real multipart JPEG tests the comparison preserves four active
rows and two independent pending offers. Repeat analysis and crop
reprocessing preserve the pair. Both crop modes and the cropped
alias stay held. Direct geometry approval and duplicate additions
are refused. To resolve the pair as separate prints, save the current
row around one complete measured outline and outside the other. To
confirm one shared sheet, explicitly select that option and submit a
quad enclosing every measured corner from both outlines. The API
checks this in native coordinates with a 0.05-pixel tolerance, logs
the selected paper ownership, and updates the count revision. Both
interfaces can load either individual outline in the crop editor.
After a one-child edit, adding the remaining print changes the count
from four to five. The other two pending offers are preserved and the
two edited crops without measured geometry still require review.

The replays, source controls and annotated example are under
`runs/crop-quality/u039-narrow-divider/`.

#### Check a risky page with two model proposals

Dense tile revision 12 admits a risky whole-page result with two to
eight model proposals. As before, at least two geometry-valid proposals
must each cover 2–20% of the page and at least two predictions must fall
below the normal detector confidence threshold. The source-boundary
checks, overlap vetoes and count-review requirement still apply. The
detector weights and automatic release gate are unchanged.

For the 1600 × 1110 JPEG55 derivative of U039, the whole-page detector
returned two usable masks at 0.504 and 0.312 confidence. Four overlapping
tiles yielded one additional count-review proposal for the middle outer
print on the left page. Its IoU to the development outline was 0.9784.
All three existing active polygons, including the classical proposal,
were identical in the before and after replays. Across five warm runs,
the median complete analysis was 2.52 s and the tile stage took 0.99 s
on the shared test host.

The multipart upload replay started with three active rows and one
pending count proposal. Explicit confirmation produced a fourth active
row with an untrimmed perspective preview. Unconfirmed and stale
requests were refused. Repeat analysis and crop reprocessing preserved
the review state. Both crop exports and the cropped alias stayed held.
The large diagonal classical proposal on the right still needs review.
An independent inspection of the decoded pixels found weak boundary
support on all four sides and lines crossing visible photo content.
The verified development sample therefore still has unresolved count
and geometry cases.

The other six U039 encoding and orientation variants and eight earlier
real replays retained their active and pending outlines. All eight frozen
and development comparisons preserved their detection and crop metrics.
The locked synthetic checks retained 18/18 detected prints and 23/23
recoverable crops. The U039 replays and the rendered review preview are
under `runs/crop-quality/u039-jpeg55/`. The outline is a development
reference and does not count towards independent release acceptance.

#### Confirm an unclaimed print in two local views

Dense tile revision 13 can look beside an existing near-square model
proposal after the normal tile path supplies an extra count offer.
The layout requires one compact proposal covering 3.5–9% of the page
with a native short-to-long side ratio of at least 0.80, and two or
more elongated proposals covering 5.5–16% each with ratios in 0.42–0.70.
Each needs valid geometry and a prepared boundary score of at least
0.76. The elongated axes agree at absolute cosine 0.97 or higher,
and the closest compact axis agrees at 0.96 or higher. PhoSca projects
the compact proposal in both directions by 1.16 times its mean side
length along that axis. It measures only projections at least four
native pixels inside the capture and clear of earlier paper claims.

The two local windows cover 56% × 60% and 48% × 53% of the page.
Their axes transpose when the projected direction is horizontal.
Each needs a unique mask with at least 0.55 IoU against the projected
search area and prepared boundary score at least 0.80. The normal
artificial tile margin applies. The mask scores must reach at least
0.60 in both views, with one reaching 0.80 and the configured
acceptance threshold. Their native-coordinate masks must agree at
IoU 0.90 or higher.

The original pixels must support three matching, unambiguous rails
within four pixels across both fits. A fourth rail from the higher
confidence view must be ambiguous and its rival must lie 5–20 native
pixels outward at both ends. The inner rail requires signed
backing-to-scene support at 80% of sampled stations. The outer rail
must have less than 30%. Outside two transverse sides the backing
has a matching periodic signal that breaks inside the print. Both
views must then independently support a revised fourth rail within
four pixels of each other. A measured enclosing four-side perimeter
or an overlap with a current outline, a pending suggestion or an
adjacent witness strip blocks the new offer. At most two search
positions with two model views each are evaluated per page.

On the original U008 development capture, the full page and the four
ordinary tiles miss the complete swing photograph. The two new masks
score 0.872 and 0.619 and agree at 0.940 IoU. The verified outline
reaches 0.97586 IoU and covers 99.53% of the draft polygon on a
1024-cell scoring raster. The upper projection remains inconclusive.
U008 keeps all three active rows and its earlier pending print.
The additional outline is a count suggestion. Its corner accuracy,
physical aspect and motif footprint still need independently measured
references.

All 14 independent native replays preserve their active outlines.
Only the original adds a pending outline. A separate replay tests both
search directions on 56 encodings and source controls.
An exact-window reproduction on one wide paper carrier can pass the
local check when no claim is supplied, because both unprepared search
windows contain the original pixels. Its overlapping active claim
blocks the offer in the complete pipeline. A smaller enclosed dark
carrier also fails the perimeter check. Pale and capture-clipped
carriers can lack measurable closure. Physical count still requires
explicit review.

Nine JPEG, downscaling and quarter-turn derivatives of U008 produce
no additional offer. The other four original spiral pages keep their
active and pending outlines. The active detector matches 6 of 38 draft
prints across this five-page development set. The new find is a pending
count suggestion. In nine alternating warm single-thread runs, the
median segmentation time on U008 was 2.82 s before the change and
3.12 s after it. The tile review stage went from 0.87 s to 1.45 s.

In normal multipart upload of a lossless PNG containing U008's exact
decoded pixels, repeat analysis produces the same four corners and
count revision. A linked, current and
explicit paper confirmation adds one row. Exact request replay returns
the saved row and preserves the other pending offer. Dismissing the
offers leaves the count at three. After reprocessing and count
confirmation, the 330 × 317 untrimmed preview is available. Both
named exports and the legacy trimmed alias remain held with
`PROJECTIVE_PRINT_OUTLINE_REVIEW`.
The held preview's rendered footprint has 0.97882 IoU against the
development paper drawing. The draft flags 45 omitted source centres
and 1,835 rendered samples with some foreign contribution beyond its
one-source-pixel tolerance band. The rounded corners still show some
backing. The motif has no independent transform in this held state.
The eight frozen and development reports retain their detection,
mask, crop and review results. Their configuration fingerprints now
record dense tile revision 13.
The paired measurements, annotated original and API receipts are
under `runs/crop-quality/regional-print-review/`.

#### Recheck an unclaimed tile in smaller views

Dense tile revision 14 reuses masks from the ordinary 70% tile run
after that run has produced a count offer. A scout must cover
2.5–12% of the page, with model confidence at least 0.60 and a
prepared boundary score of at least 0.80. The three highest-confidence
unclaimed scouts are eligible. Only 52% grid windows that contain
the complete scout and satisfy the artificial-edge margin are run.
A cached window supplies all scouts that can use it. One supported
window can be paired with a centred 52% × 50% view.

Each confirming window must contain exactly one match at scout
mask IoU at least 0.65. Each admitted mask needs confidence and
prepared boundary score at least 0.80, and one needs the configured
acceptance score. The retained masks from different views must all
agree at IoU 0.90 or higher. If more than two masks agree, the two
highest-confidence ones reach the source verifier.

The unenhanced image must support all four fitted sides at 0.85 or
higher. One fit must have no ambiguous side and the other exactly one.
The three other sides agree to within four native pixels measured
perpendicular to each rail. The disputed side can differ by up to
15 pixels. A separate transverse strip establishes a repeating
backing palette. Of 160 stations on the middle 84% of a disputed
line, at least 90% in all and 60% in each eighth must show that
palette outside and a decrease of at least 12 OpenCV Lab L units
over two native pixels into the image. The median contrast across
four pixels must reach 24 units. Multiple supported offsets spanning
more than three pixels, or a competing independently supported rail
more than 2.5 pixels away, prevent an offer. The chosen rail can move
the unambiguous fit by up to two pixels. A neighboring claim crossing
the witness strip, raw masks or fitted outline vetoes the offer. A
measurable larger enclosure does as well.

Two previous omissions on the original spiral album now appear
in their count review. U008's polar bear draft reaches 0.97067 IoU.
Its ordinary and smaller model views can agree while both follow an
outside backing rib. The centred context supplies the correct polarity
at the top. U040's slide photo reaches 0.96685 IoU. The paired lower
rails differ by roughly four pixels. The signed original step selects
the outer visible edge before the narrow gap to the next photo.
U008 retains three active rows and two earlier pending offers. U040
retains four active rows and its one earlier pending offer.

The independent material controls include folds, alternative rails,
gaps, adjacent papers and wider carriers that copy the same image
and backing. All 17 altered normal-pipeline controls preserve their
previous offers and active polygons. Identical local image evidence
can still support an inner contour on a single carrier when its
claim is absent. These offers require explicit confirmation of a
separate paper print, with both geometries in review.
None of the 18 tested JPEG, downscaling and rotation derivatives
produces a new offer. Three other native pages from the same album
preserve all active and pending polygons.
Across seven alternating warm single-thread runs per arm, U008's
median segmentation rose from 2.95 s to 3.39 s and U040's from
2.13 s to 2.34 s. The tile stages rose from 1.33 s to 1.81 s
and from 0.78 s to 1.07 s, respectively.
The raw 3 × 3 grid at 52% found agreeing masks for 28 of 35 draft
prints across three open development spreads. The four 70% tiles
alone reach 19. Those are proposal measurements before the native
perimeter, conflict, count and crop checks.

Multipart upload of the original decoded rasters as PNG and repeat
analysis preserve all earlier photos, warps, available previews and
pending polygons. A linked explicit confirmation adds one photo on
each page. Reprocessing and count approval preserve the untrimmed
303 × 299 and 264 × 277 previews. Both named downloads and the legacy
trimmed alias remain held. In a bilinear projection of the development
drawings, 2,141 U008 output positions and 679 U040 positions receive
some contribution from outside the draft masks. The drawings leave
physical aspects and exact corner ownership unresolved. Receipts,
source controls and rendered examples are under
`runs/crop-quality/multiscale-review/`.
All eight paired benchmark reports preserve detection, masks,
crop results and review metadata for the previous inputs. Their
configuration profile records dense tile revision 14.

#### Compare whole-page detector resolution on dense spreads

At `697069a`, two entries in the local development registry
register the same mixed ONNX bytes, SHA-256
`1e95a1a90ff897df7099ff80675186ec1d8145f7bb07559fe918b3dd3a6db748`,
with NCHW 512 × 512 and 768 × 768 input respectively. The normal
hybrid evaluator runs both manifests at proposal threshold 0.25,
acceptance threshold 0.70 and scoring IoU 0.85. Both runs apply the
same classical, native-source and crop checks. On the five Spiral
development views, all crop variants remain in review.

| Same detector and inputs | 512 × 512 | 768 × 768 |
| --- | ---: | ---: |
| Spiral matches at IoU 0.85 | 6/38 | 26/38 |
| Spiral extra proposals | 11 | 15 |
| Frozen synthetic matches | 17/18 | 18/18 |
| Frozen synthetic extra proposals | 4 | 9 |
| Frozen synthetic passing recoverable whole prints | 9/11 | 7/11 |
| Frozen synthetic passing recoverable motifs | 10/12 | 8/12 |

The synthetic rows here compare the same mixed checkpoint at both
resolutions. The separately frozen synthetic-control checkpoint
and seeded geometry regression still supply the 18/18 and 23/23
regression bars. The larger whole-page configuration loses four
passing recoverable crops and creates five additional frozen extras
relative to its 512 baseline. It stays a development comparison.
At the raw detector stage, the same weights and scene pixels rise
from 7 to 26 correct masks at 768 while extras rise from 11 to 18.
The dense pages therefore expose a sampling-scale problem before
the native contour checks. The recorded training and validation
cohorts have at most four prints per frame, compared with eleven
or twelve on the open Spiral spreads.

The four complete normal-pipeline reports, local manifests and
`resolution_pilot.py` are under
`runs/crop-quality/detector-resolution-pilot/`. The raw resolution
audit and training source counts are in its `independent/` folder.
The experiment uses no reserved calibration or release references.

#### Check high-resolution nominations against existing physical claims

The 768 input has more small masks, but some describe a print already
represented by a pending contour or a count correction. The native replay
compares them with 512 active, pending, crossing and merged-outline claims.
On U008, the polar bear fit reaches 0.9723 IoU against the development
outline and is already offered by revision 14. The glare-heavy upper
candidate has 0.834 detector confidence, but its four native boundaries
cannot be resolved. The elongated right proposal also yields two child
masks. The lower child reaches 0.9753 draft IoU after an ordinary fit.
The upper child does not have a verifiable perimeter. Sixteen contextual
layouts per child, with duplicate windows removed, produce 9 upper and
15 lower observations. The existing stock and separation checks do not
confirm any of the 25 tested pairs, including the two original seeds.

On U039, a 768 nomination leads to two independent 512 views centred on
the measured backing. Their raw masks agree at 0.9611 IoU and their fitted
corners differ by at most 1.997 native pixels. The two fits score 0.9816 and
0.9853 against the development outline. The unchanged 512 pipeline already
claims the target on two rotated inputs. JPEG quality 55 and 75 do not
reproduce the native nomination and fit. An optically similar wider
single-sheet control also passes the local rails when its outer boundary
falls beyond the search window. The outline stays a development finding.
The original-pixel checks, outline and seeded preview are under
`runs/crop-quality/resolution-review/`.

In the frozen comparison, all nine final active 768 extras touch an
existing 512 claim. With one real peer deliberately withheld in a
same-stock stress test, the unclaimed 768 confidence gate admits 8 real
individual-print outlines and 24 false inner outlines from single carriers.
Adding the existing four-side native fit and enclosure checks leaves 4
real and 10 false outlines. An offline enclosing-perimeter check over
the union with the known peer rejects 6 of those 10 false candidates
and preserves all 4 real ones. The four remaining false candidates have
open ends. A separate check selects a unique disjoint 512 peer without
draft polygons, measures three native rails with the same exterior and
looks for both long rails to reach the same capture edge. It witnesses
the carrier in three of those four stress cases and none of the four
genuine cases. One quality-75 JPEG still has insufficient rail evidence.
These diagnostic checks are outside the app.

#### Isolate print density in a matched fine-tuning trial

At `7ee38fc`, two 512-input candidates start from mixed checkpoint
`8fa869a3fbafbac6217b8d99ed6e39f8a0c78fdd87d4128716d09851b80a0d86`.
Each arm sees 96 training views. The 40 pinned legacy derivatives,
24 inherited procedural views and 8 stock and cover controls are common.
Only 24 paired training views differ. The sparse views contain two to
four focal prints. Dense views keep the same focal artwork and exact
owner pixels and place other prints around them. The median generated
training footprint is 4.197% of the frame in the sparse arm and 4.265%
in the dense arm. Both train for three epochs with seed 0 and geometry
and compositing augmentation disabled. Selection uses 20 shared
views. The 32 shared challenge views use separate roots and a different
procedural rendering variant. Reconstruction verifies all recorded
image, owner-mask and label hashes and no cross-split reuse of root
groups, image hashes or procedural artwork IDs. The real Spiral
development views are evaluation only. Reserved calibration and release
material are unused.

| Measure | Mixed parent | Sparse arm | Dense arm |
| --- | ---: | ---: | ---: |
| Raw dense challenge matches / extras at 0.25 | 51/139 + 10 | 99/139 + 12 | 139/139 + 4 |
| Raw dense challenge matches / extras at 0.70 | 4/139 + 1 | 41/139 + 2 | 137/139 + 0 |
| Raw non-photo cover extras at 0.25 | 24 | 2 | 25 |
| Raw enclosing-sheet matches / internal fragments at 0.25 | 0/4 + 23 | 0/4 + 0 | 0/4 + 22 |
| Frozen hybrid matches / extras | 17/18 + 4 | 18/18 + 3 | 18/18 + 4 |
| Frozen hybrid passing whole prints | 9/11 | 6/11 | 8/11 |
| Frozen hybrid passing motifs | 10/12 | 7/12 | 9/12 |
| Spiral hybrid matches / extras | 6/38 + 11 | 13/38 + 18 | 30/38 + 18 |

All hybrid rows use the normal classical and source checks, proposal
threshold 0.25, acceptance threshold 0.70 and scoring IoU 0.85. Both
fine-tuned weights lose the camera-tilt example in both crop modes.
The dense arm swaps which of the two touching prints passes. The sparse
arm also loses the glare case in both modes. On Spiral the dense arm
reduces mean whole-print footprint cutoff against the development outlines
from 0.1581 to 0.0677. Three locally ready whole-print candidates and one
locally ready motif lack verified crop references. Local readiness provides
no release evidence. Neither checkpoint is adopted.

On a separate set of eight available development views, mixed and dense
both match 24 of 25 draft paper outlines. The number of extras grows
from 7 to 12. Four of the added proposals are on the clipped NerdScan
source. The quality-75 reproduction of the 35-frame contact sheet stays
unmatched at IoU 0.85, with five extras for mixed and six for dense.
Both retain the draft paper preview at footprint IoU 0.8295 with 16.77%
cutoff. These source-limited drafts cannot establish clean motif extent.
The complete reports and matched comparison are in
`runs/crop-quality/crowded-print-training/known-real/`.

The dense model's U008 merge propositions also include two separate
print masks. In a source replay using masks alone to nominate a parent
and children, the existing merged-pair measurement confirms one
two-paper ownership conflict, on the left pair already covered by the
app's review. It does not justify removal of other proposals.

The sparse ONNX SHA-256 is
`690162aa452ddf77ab0ad2348fdc1a09757ae56ab639f60910c699d9c15b3495`.
The dense ONNX SHA-256 is
`32cfb0e3cb24b042e63a014f6437e844a844ab372dc161a983ca3272d73278ff`.
The corresponding training receipts, source inventories with roots,
splits and hashes, raw challenge results and hybrid reports are under
`runs/crop-quality/crowded-print-training/`. To reproduce the comparison
from two matching evaluator reports, run:

```bash
python scripts/compare_crop_evaluations.py \
  runs/crop-quality/detector-resolution-pilot/locked-512.json \
  runs/crop-quality/crowded-print-training/dense-locked-hybrid.json \
  --output /tmp/phosca-dense-locked-comparison.json
```

The comparison records per-reference changes in crop state, reasons and
automatic eligibility. Geometry summaries use the same reference rows
with measurements from both runs. It also includes native corner and
landmark error, aspect, cutoff, foreign interpolation counts and timing.

#### Replace eight controls with visibly distinct paper-owned cases

A third arm keeps 88 of the 96 dense training views, the mixed starting
checkpoint and the same training settings. It replaces just the eight
common controls with three matched scene pairs and two non-photo album
covers. Each pair uses the same art on a single enclosing carrier with
a visible perimeter in one view and on two separate papers with a
physical gap in the other. The new train roots start at 6624700. The
eight prospective material views use separate roots starting at 6625700.
The 20 selection and 32 original challenge views remain byte-identical.
All three splits and the new material stress set are outside one another
by source group, source-image hash and procedural art donor.

On the material stress set, the new arm preserves 8 of 9 matches at
confidence 0.70 and reduces extras from 7 to 3 relative to the dense
arm. All six separately mounted prints still match. At confidence 0.25,
it keeps 9 of 9 and adds one extra, from 12 to 13. The original challenge
contains optically near-identical cover and carrier renderings. At 0.70
it has 169 of 176 matches as before, but extras rise from 7 to 49 there.
At 0.25, the dense subset changes from 139/139 + 4 to 138/139 + 6.
The frozen normal path has 18/18 matches and one extra. Crop passes fall
to 7/11 and 8/12 from the dense arm's 8/11 and 9/12.

On Spiral, this arm reaches 33 of 38 matches and 13 extras. All three
additional matches over the dense arm are on U039. Mean whole-print
draft footprint IoU rises from 0.8542 to 0.9154, and the mean omitted
draft fraction falls from 0.06768 to 0.04074. Two whole-print rows are
locally ready, with no verified reference or automatic release.

The trial ONNX is
`3f8fb0b98def1bda9f3325144f1d6c86dd61084fb275da45f5cd4f54da32170b`.
Its source inventory, raw controls, training receipt and normal reports
are in `runs/crop-quality/crowded-print-training/stock-controls/`.
The arm stays a development experiment.

#### Corroborate one rotated mask in a local view

The normal 512 path already compares two whole-page quarter turns after
an initial tile finds a missing print. On native U039 the 270° mask for
the lower peacock print has model confidence 0.715 and prepared boundary
score 0.8088. Its 90° counterpart is too weak for the direct comparison.
The new path considers exactly one unclaimed rotated mask that meets the
existing grid scout floors. It measures the four original boundaries
against the surrounding ribbed backing. One additional 60% × 60% view
is centred on the measured outline and shifted towards the page centre.
On U039 this is the native window `(576, 444)–(1536, 1110)`. The local
mask has confidence 0.806 and prepared boundary score 0.8228.

The two masks must agree with IoU at least 0.90 and their independently
measured corners within four native pixels. The original pixels must
show the signed album-to-print transition on every side. The backing
phase must end at the photo on two opposing sides and their witness
bands must be unclaimed. A larger enclosing perimeter, a raw or fitted
collision with a current claim, or a conflicting open review vetoes the
offer. This path reuses the existing detector and adds at most one local
inference call, with no additional whole-page pass.

The normal native U039 upload retains four active proposals and both
previously pending outlines. It adds a third pending outline at
0.9828 IoU against the development draft on a 2048-pixel scoring raster,
with a maximum corner discrepancy of 3.70 native pixels. It covers
98.59% of that draft. Both measured views agree to 1.31 native pixels.
The upload, reanalysis, explicit standalone-print confirmation, count
revision and reprocessing tests preserve the four original photo
records and preview JPEGs. Confirmation raises the active count to five.
It serves the original-pixel rectified preview and keeps the two named
export variants held. Reprocessing produces identical preview bytes.

The three other native open pages and the front cover do not add an
offer. Rotated and recompressed U039 variants and single-carrier
lookalikes also abstain. On JPEG75 the candidate in the additional
local view hits a known neighbour. The source claim veto prevents the
app from offering it. The eight frozen and development regression
reports reproduce their prior detection and per-mode measurements.
Both configuration profiles record dense tile revision 15.
Five warm single-thread development measurements on U039 put median
segmentation time at 3.61 s versus 2.76 s. The tile and review stage
accounts for 1.93 s versus 1.21 s. This is the qualifying page, including
the additional source measurements.
Receipts are under `runs/crop-quality/recentered-rotation-review/`.

#### Check a vacant position beside measured prints

On native U040, two held outlines in the same column are already
measured. One came from the existing smaller-grid check. The new
path compares their centres and matching side directions. The
axis cosine must reach 0.96, the two side lengths must agree within
1.25×, and the pitch must be 0.93–1.35× the mean matching side length.
Extrapolate in both directions and proceed only when exactly one
projected four-corner position meets the frame and claim checks.
No annotation determines the position. On this page the one vacant
centre is near `(1012.2, 282.4)`.

Two fresh local 512 views cover 36% × 42% and 44% × 49% of the page.
Each must supply a unique matching mask, with model confidence and
prepared boundary score at least 0.80. The two raw masks agree at
0.9573 IoU. The original-pixel verification requires signed intensity
and material transitions on all sides. It fits the two transverse
rails independently and rejects a competing line on the inner edge.
On at least one lateral flank, the periodic signal must repeat in
three independent exterior strips and cease inside the outline. On
the selected U040 right flank the far-to-near backing correlation is
0.936, while far-to-inside correlation is −0.042. Native witness
bands and the fitted footprint must be clear of active, pending and
earlier raw claims. The fitted corners differ by at most 1.5 native
pixels. A larger closed source perimeter vetoes the offer.

The two native fits differ by 0.656 pixels at the farthest corner and
reach 0.9593 and 0.9612 IoU against the proposed U040 development
outline. Against the draft paper footprint, the held rendered comparison
omits 4.12%. The largest native corner discrepancy is 7.24 pixels. Most
omitted draft pixels near the paper edge resemble neighboring album
stock. This source has no independently confirmed mask, aspect
or landmark references for either mode.

The normal multipart upload keeps four active detections, their
stored outlines and preview JPEGs, and both prior pending items
identical. It adds one uncounted print suggestion. Reanalysis
reproduces it. A stale revision or absent standalone-print
confirmation returns 409. Explicit confirmation raises the count
to five. Reprocessing preserves the 266 × 256 source-corrected
review JPEG. Both named variants and the legacy trimmed alias
return 404.

The unchanged detector finds this print at 90°. The 180° and 270°
uploads contain incomplete or competing proposals and no suitable
pair of measured neighbours. None of the three gains a duplicate
offer. JPEG55, JPEG75 and JPEG90 also fail the ordinary page
prerequisites, even though the JPEG90 source comparison can fit the
paper given two matching masks. A diagnostic that also considered the
active candidates as anchors found no additional candidate with four
supported original sides in JPEG90, 180° or 270°. Wide same-art
carriers and real-paper overlap controls exercise the claim, source
and enclosing vetoes.
A carrier with visible edges outside the local detector windows
requires the wider 1.65× search. With an optically identical open
carrier the visible scene can still pass a fixed-mask comparison.
The suggestion therefore requires explicit standalone-print review.
All eight regression suites preserve the prior detection and crop
results. On the qualifying U040 page, five warm single-thread
measurements under concurrent development load give a median
segmentation time of 4.26 seconds versus 3.54 with this check disabled.
Both profiles record dense tile revision 17. Reproduction,
projection and upload receipts are under
`runs/crop-quality/neighboring-print-review/`.

#### Show open print edges near the capture boundary

On native U008 the mixed detector's standard 70% tile pins an edge
of the giraffe print to the capture boundary. Its remaining edges
do not establish a closed outline. Two model views at 56% × 60%
and 55% × 50% of the source agree at 0.9404 IoU, at confidence
0.871 and 0.906. Normalized masks nominate the same source region
without a new whole-page inference call.

At the original scale the material beside the frame shares the palette
and repeated lightness of independent ribbed backing. The source check
fits local rails and retains only intervals with distributed signed
transitions. A nearby active or pending outline vetoes a conflicting
perimeter, and evidence stations within nine native pixels of a claim
are excluded. On U008 this leaves eight open intervals on three sides
and withholds the edge by the upper neighbour. The closest displayed
endpoint is 11.85 native pixels from the capture frame. The original
does not establish the exact extent of the corner there.

The new evidence is a page-level overlay in the desktop and mobile
count review. It supplies no crop quadrilateral or inherited paper
geometry. Opening it sends no write request. Both surfaces draw
the same eight source intervals, with three active and three
previously deferred detections. The ordinary upload and reanalysis
retain the earlier contours and crop preview bytes. A manually drawn
draft on the same page intersects a protected region and returns
`protected_region_cut`. Confirming a different deferred photo leaves
the eight intervals present and the two named exports held. A separate
test covers retiring the overlay after an accepted manual outline
occupies its source witness band, including a partial claim and a
polygon-only correction. This retirement carries no geometry into the
saved photo.

With the two observed model masks held fixed, eleven of sixteen
rotation and JPEG cases regenerate the hint. All ten copies with at
least three columns removed abstain, as do two physical overlays
placed across the capture or bottom evidence and a claimed common
carrier. An optically identical unknown carrier remains ambiguous.
On the qualifying source, the two bounded inference calls and native
measurement take a median 0.396 seconds over three warm samples.
All eight comparison runs preserve the earlier detection and both
crop-mode measurements.

Both configuration profiles record dense tile revision 16.
Development receipts are under
`runs/crop-quality/near-capture-print-review/`.

#### Separate native crop ambiguity from missed detection

The crowd-trained camera-tilt failure keeps the same detected print.
Its top mask moves enough that the search can include the top ink
line, about ten native pixels inside the true paper edge. Both lines
then have full distributed support. The existing paper witness fits
the exterior accurately and keeps the count and crop hold. The held
motif preview measures 0.0835 native pixels of corner error, 0.0659
pixels at p95 across landmarks, and zero cutoff or foreign samples
beyond one source pixel. An apparently identical flush inset and a
two-level shadow on a separate inset both pass a stricter stock test.
The accurate preview does not resolve which material is paper.

On the touching-print controls, an inward retry can select a printed
border and appear unambiguous while omitting over 2,100 owned pixel
centres. A source-assisted estimate using a stable neighbour restores
966 omitted centres on one held cut without adding foreign centres.
A different estimate overlaps the neighbour's raw claim and is
blocked. The preview experiment keeps the existing hold and remains
outside the product.

For the JPEG75 contact print, one of 17 tested detector contexts
finds a fragment mask that passes the original-pixel grid check.
Applying that measured mask to thin separate-paper stress controls
still accepts all 32 one-pixel rim variants and 8 of 32 two-pixel rim
variants. This is a detector-phase effect and insufficient evidence
to change the count. The existing explicit paper and motif review
workflow remains in use for that print.

An additional independent screen uses 30 real development views and
eight source families. The ready-neighbour paper-stock experiment
reaches no matching actual renderer case in 97 active contours. Five
comparable original crops also have aspect uncertainty. In the closest
examples a blank rim is absent or the peer or corner witness is too
weak. The synthetic recovery remains a development control.

With the original, sparse, dense and stock models, the native photographed
contact sheet has one true match from the same classical perimeter.
They have respectively 3, 4, 5 and 3 extra proposals. At 33% resolution
and JPEG75 all four miss the paper at scored IoU 0.85, despite a
separate held editable outline at 0.9885 IoU to the draft. The three
trained models offer no repeatable ownership advantage on this source.
The existing sleeve and paper review remains necessary. The tested
seam expansions also admit pixels from separate papers with narrow
backing gaps and stay outside the app.

#### Replace old contact layouts with paired physical ownership examples

The fourth development arm keeps the mixed starting checkpoint and
the stock arm's settings. It replaces 16 inherited procedural train
views with eight fresh scene pairs. Each pair shares artwork, projection,
blur, orientation and JPEG settings. One view has a single repeated
contact print. The other has two papers with an authored gap of one,
two, three or six pixels before projection. The 80 other training
rasters, the 20 selection views, the 32 crowd challenge views and
the eight prospective stock views are identical. Reconstruction verifies
the new raster, instance, and owner-label hashes. The replaced views
have 35 instance labels while the new views have 24, so this comparison
does not isolate the physical ownership cue from the training-instance
distribution. No new real source enters training or selection.

| Development comparison | Stock | Paired contact |
| --- | ---: | ---: |
| Raw fresh contact matches / extras | 7/36 + 88 | 16/36 + 71 |
| Normal fresh contact matches / extras | 7/36 + 101 | 11/36 + 85 |
| Normal fresh contacts on an enclosing sheet | 3/12 | 3/12 |
| Normal fresh contacts on separate papers | 4/24 | 8/24 |
| Normal photographed contact JPEG75 | 0/1 + 5 | 0/1 + 5 |
| Frozen normal matches / extras | 18/18 + 1 | 18/18 + 3 |
| Frozen passing whole prints | 7/11 | 5/11 |
| Frozen passing motifs | 8/12 | 6/12 |
| Spiral normal matches / extras | 33/38 + 13 | 23/38 + 15 |

Five of the paired arm's eight raw enclosing-sheet matches fall below
the threshold during fusion. A direct stage replay finds the model
mask at 0.867–0.914 IoU and the selected classical edge contour at
0.784–0.828 on these five sources. The subsequent perimeter stages
preserve that change. On the real JPEG75 print, the draft whole
footprint IoU rises from 0.8295 to 0.8496 while the rendered preview
loses the entire handwritten heading above the first row. Both modes
remain held. The frozen losses include new cast-shadow and glare crop
errors. The product model and release policy stay in force.

The ONNX SHA-256 is
`236796403f8c6f858a906554c4d7529ee7bcc8fa4af0509c73196df350449b05`.
The training inventory, owner-mask manifests and matched reports are
under `runs/crop-quality/crowded-print-training/paired-contact-controls/`.

An independent source-only replay retraces the five enclosing-paper
fusion losses. On the authored sheet `1449607-joined`, the losing
classical seed allows a four-sided native paper measurement. It
improves the held full-print comparison from 0.8038 to 0.9939
footprint IoU and reduces the strict omission from 23,664 to 62
source centres, with zero foreign interpolation samples beyond one
pixel in either preview. Two other measured alternatives intersect
still-active print claims, and one case has an unsupported fourth
rail. With the current mixed checkpoint, the same wider outline
for `1449607-joined` would enclose a separately active model
proposal. The overlap veto holds. Some near-identical separate
stocks can reproduce the optical signal. The comparison stays
a held development prototype. Its
source and rendered tests are in the `carrier-proof/` subfolder.

#### Render the measured paper comparison in classical fallback

An independent replay with no usable ONNX model finds two active rows
on each of the JPEG55, JPEG75 and lossless contact-sheet derivatives.
The print is on the right. The other row follows the negative sleeve.
The stock measurement already suggested an editable outer quad for the
print, while the renderer returned only the tight primary crop. The
new review asset independently rectifies the measured quad at the
original source resolution and uses the owned bilinear sampler. The
branch requires both modes to be in review, checks previously included
protected content and vetoes a competing current or raw claim above
one square source pixel. It leaves the earlier comparison for an
explicit paper-grid dispute in place, including the projected cuts.

| Exact source with classical fallback | Primary to comparison footprint IoU | Draft foreign interpolation locations beyond 1 source pixel | Other active claim inside the comparison |
| --- | ---: | ---: | ---: |
| JPEG75 | 0.8295 → 0.9870 | 1,928 → 156 | 0 px² |
| JPEG55 | 0.8363 → 0.9884 | 0 → 220 | 0 px² |
| Lossless PNG | comparison withheld | comparison withheld | 125.74 px² |

On JPEG75 the comparison preserves 1,938 of 1,938 sampled heading
pixels and 315 of 315 panel anchors, up from 1,429 and 312. The
largest corner discrepancy against the draft decreases from 68.56
to 5.71 native pixels. The 609 × 781 rectification takes about 8.4 ms
warm CPU time on one thread. The paper and motif references and the
physical aspect ratio still lack independent release verification.

Paired multipart uploads against `d926d72` leave the two fallback
counts, source masks, saved warp, motif metadata and primary JPEG
unchanged. The Q75 model-assisted case retains its existing five rows
and byte-identical comparison PNG. The three export URLs stay held
for the contact print. Reprocess reproduces the new JPEG75 comparison
PNG. Adding a claimed overlap invalidates its URL immediately and a
later reprocess keeps it unavailable. Eight regression comparisons
preserve both modes and detection results. The outline-suggestion
configuration revision is 11.

A separate wider search is still an experiment. Across 16 resizes,
encodings and rotations of the same real source, nine wider outlines
pass its material checks, at 0.984–0.991 draft IoU. Eight of these
overlap an existing proposal. This search leaves the upright JPEG90
and the 90° and 270° JPEG75 variants unresolved. A follow-up fits each
side from the signed change along the locally measured Lab chroma
direction. It recovers an outline at roughly 0.9914 draft IoU on all
three. The JPEG90 and 90° JPEG75 fits enclose another active proposal.
At 270° the contact fit is clear of the three other active masks. The
same optical refit also measures an enclosing rectangle around the
negative sleeve in that view, so it cannot establish paper ownership.
Among the 31 real development views and 24 authored joined/separate
controls, the stricter search only yields complete outlines on
two encodings of this same contact-sheet source. Both prototypes stay
outside the product. Evidence and reproducible scripts are under
`runs/crop-quality/standalone-paper-comparison/`.

#### Locate paper edges around a held contact grid

The 270° JPEG75 contact-sheet view has four active proposals. The
right-hand paper in the original orientation is at the bottom of the
rotated view. Its rendered whole-print mode is held for ambiguous
boundaries, uncertain corners and unknown aspect. Two preceding hints
mark competing lines along part of the current crop. In the new route a
held classical contrast proposal with boundary score at least 0.78 can
add four open source guides alongside those lines. The route requires
both modes to be in review and gives precedence to an existing outline
or rendered comparison. Explicit geometry and unresolved stock groups
stay on their respective review paths.

In a patch no larger than 1.2 million source pixels, three Lab chroma
thresholds must find enclosing contours with corresponding corners no
more than 5 native pixels apart. On each side, 360 source stations
estimate the local chroma change between two plateaus. The fitted edges
use the half-height crossing where the two-pixel bands supply signed
contrast. The independent lines must support a convex in-frame scope.
Its measured area must contain at least 98.5% of the primary footprint.
Dense interior samples check the pale rim and reject a sustained break.
Five or more period-matched pairs of rails must support a regular printed
grid. The median body-minus-rail lightness must be at least 18 OpenCV
Lab L levels. At least 80% of pairs must reach +8, and no pair may fall
below −10. The rails must also retain the measured pitch in direct
nearest source samples.

In the 1,014 × 1,346 JPEG75 view all 14 source rails pass the native
period test. Body-minus-rail medians range from +8.8 to +86.5 on the
paper and −30.5 to −22.8 on the sleeve. The measured paper scope
improves draft overlap from 0.8057 to 0.9915. It covers all 3,255 sampled
heading centres below Lab L192, versus 1,819 within the detector mask.
Both shapes cover all 315 interior landmarks. These measurements compare
source scopes. The ordinary crop still needs a checked paper outline and
an independent physical aspect measurement.
Six alternating warm executions on one OpenCV thread report 337 ms
median crop CPU time, versus 218 ms with the extra source check bypassed.

The UI labels the four teal spans as measured transitions. The endpoints
stop short of the corners. A pending or active claim intersecting the
full source scope by more than a native square pixel vetoes the added
guides. An enclosing scope is used for this check even if none of the
drawn segments touches the claim. A protected region can also veto the
guides. The upload and reprocess keep the existing primary warp and
preview byte-identical. An interior manual claim removed the new spans
without touching either side of the earlier hint. The two named and the
legacy trimmed downloads returned HTTP 404 throughout.

The archived normal API accepted an unchanged tight warp when the user
only added diagnostic physical dimensions. Its whole-print download
then returned HTTP 200 even though the unchanged footprint covered
80.7% of the draft. An observed contact edge discrepancy now activates
the existing explicit contrast-outline confirmation and native boundary
check. The marker survives reprocessing of the same automatic row after
a peer invalidates the lines. Saving without the confirmation returns
HTTP 409. A confirmed retry with the same tight corners stays held after
the native check. Even the diagnostic wider edit is still held for
boundary ambiguity in this case.

Across 16 rotations, resizes and encodings of this source, the normal
two-mode crop replay adds one guide. Eight frozen, seeded and real
development comparisons preserve the detection and both crop results.
The outline-suggestion configuration revision is 12.

Near-identical separators, overlapping plates and positive artwork
behind simulated clear film can reproduce the optical proof. In an
independent 76-raster control study the measurement alone passed all 16
faint same-chroma separations, 12 overlaps and 12 positive-sleeve
treatments. The fixed four guides indicate a possible boundary for
review. The contour is used internally to test conflicts. Exports and
instance ownership still follow the ordinary review decisions.
Receipts are under `runs/crop-quality/printed-contact-edge-review/`.

#### Require separate confirmation for an unsupported contrast outline

The normal JPEG55 upload of U039 produced a held `contrast_region`
proposal whose perimeter cuts across visible photo content. A diagnostic
size-only edit previously made both named crops downloadable. The saved
quadrilateral was identical to the original preview. The issue was in the
manual save route that converted an automatic proposal into manually
supplied geometry, which uses a different boundary check.

The route now requires an explicit paper outline and an explicit
confirmation for an automatic contrast proposal whose whole-print
mode remains held with `PRINT_BOUNDARY_UNSUPPORTED`. Proposals at the
capture edge already have a separate hold that survives manual edits.
A motif edit, implicitly generated corners or an ordinary size or
orientation edit returns `409` and preserves the saved state. The
desktop and mobile editors bind confirmation to the four displayed corners. Confirmed
edits continue through the existing geometry and crop checks and record
the confirmation in the correction revision. «Falsches Objekt» removes
the proposal from the active count and retains its review history.

On the same real upload, the dimension-only and orientation attempts
now preserve both export holds, including after crop reprocessing.
Rejecting the false proposal reduces the active count from three to two
and leaves the separate pending print offer intact. A source experiment
measured spans of 210 and 243 native pixels where the diagonal proposal
cuts through visible scenes. A faint-shadow test produced both spans
too. They remain diagnostic evidence while a reliable contour
correction is investigated. Development receipts and browser captures
are under `runs/crop-quality/u039-jpeg55/outline-confirmation/`.

A separate manual edit tested a four-corner source fit of the lower car
print, with IoU 0.9611 against the development outline. After explicit
confirmation both named exports returned the rectified single print.
This verifies the manual path on the real source. The crop is a
development example, with rounded corners still visible in the render.

#### Show source detail across an unsupported contrast outline

The source advisory runs when the existing tile review examines a
risky page and fusion retains a single eligible classical contrast
rectangle among two to eight detections. The rectangle must be fully
inside the capture, with at least 12 native pixels around its corners
and a shortest side of at least 320 pixels. All four sides must lack
independent boundary support above 0.42 and have no competing edge or
positive paper rim. A veto stops before regional model inference.

PhoSca takes up to two views centred near each side. Each witness uses
a regional segmentation mask with model confidence of at least 0.76
and prepared boundary evidence of at least 0.68. Two opposite sides
of that mask must have source-measured rails that bracket the suspect
line. The method keeps the interval shared by plausible fits, away
from the observed scene boundaries. The original source must contain
at least five connected structures crossing a native 21-pixel band.
The connections must cover at least six of eight equal sections and
span at least 60% of the length. Exactly one pair on different axes
and with distinct regional masks must agree. The original proposal
must survive refinement with the same geometry.

On U039 JPEG55, the review shows 210 and 243 native pixels on the
upper-car and flower scenes. The normal upload and reprocessing
retain three active proposals and one pending offer. Both named
exports and the cropped alias remain held. Desktop and mobile show
the same endpoints and clear the guides when the editing corners
move. The orange marks refer to local image detail. Geometry, crop
status and the active count depend on the existing checks and review.

An independent replay withheld the advisory on 24 real control
contours, including a physical contact sheet. Six of ten targeted
recompositions with narrow seams, faint rims or weak shadows still
produce both marks. The trace pair does not resolve stock ownership.
In three real pipeline comparisons, both mode records and rendered
pixel arrays match the run with the advisory disabled. U039 took
about 1.15 s for the advisory in one single-thread CPU replay.
The development source reports and browser captures are under
`runs/crop-quality/u039-jpeg55/crossing-traces/`.

#### Recheck a confirmed contrast outline in native pixels

An explicit correction of a held automatic contrast outline replaces
the automatic paper mask with the entered outline. The server's recorded
confirmation enables a native paper-boundary check. The check measures
the sides of the entered outline on the original raster. It never
substitutes its fitted quadrilateral for the user's
saved corners. Unresolved measurements keep both modes in review.
Measured competing lines can be shown on the source. The untrimmed
review view keeps sampling within the submitted quadrilateral.

A clear fit more than 0.75 native pixels from any entered corner also
requires review. A suggestion is offered only when every side has at
least 0.90 support and 0.70 corner support and the overlap and content
checks accept it. This is a positional disagreement threshold. Pixel
ownership at subpixel distances still depends on the source evidence.
Two supported rounded-corner arcs can trigger the existing transparent
whole-print preview. A lone arc carries no new transparency decision.
Existing saved crops with a confirmation but no native-check record
stay out of the named downloads and return to the geometry queue
until reevaluated. Geometry approval and the cropped alias use the
same guard.

In U039 JPEG55 the manually entered lower-car quadrilateral keeps its
original coordinates. The two source lines at its lower edge now
appear in review. Both named exports and the cropped alias remain
held after reprocessing. A second entered quad on the same paper
triggers the supported upper-corner comparison. The lower corners
remain unresolved. An authored full-bleed control passes with its
exact corners. A one-pixel inset loses 1,316 known source centers
from that control and now holds both modes with a separate measured
outline suggestion.

The U039 untrimmed raster was pixel-identical in the before-and-after
renderer runs. Across nine warm single-thread runs, the median was
126 ms before and 38 ms after. The ambiguous boundary holds the case
before the renderer starts motif inference.

The source report, normal upload receipts and review renders are under
`runs/crop-quality/u039-jpeg55/manual-boundary/`.

#### Compare a measured outline on ribbed backing

After the native recheck of a confirmed contrast outline, a comparison
can be measured while both crop modes are held for an ambiguous
paper boundary. It requires a passed protected-content check, no other
outline preview or loadable outline, and no other reason apart from
possible corner or aspect uncertainty. Each saved side must be at
least 180 native pixels long. Two or more fits per side at search
radii of 3, 5 and 7 pixels must agree within 0.85 pixels of their median
endpoints. Each fit requires at least 0.73 edge support and 0.52
near its corners.

The fit uses the midpoint of the local Lab transition between paper
and the warm backing. It samples at quarter-pixel offsets through
the transition. The middle 82% of the edge supplies the measurements.
At least 88% of the edge stations and 58% of each of eight sections
must support the transition. The robust line's absolute residual is
at most 1.2 pixels at p90. On two opposite sides across the ribs, the
near exterior pattern must repeat and match the more distant backing
in phase and amplitude. The measured outline must move at least one
corner by more than 0.75 pixels and all corners by at most 6 pixels.
Claimed neighbouring paper, removed protected content, incomplete
source bounds and evidence of a larger closed perimeter veto the
comparison. The closing check is bounded to a 64-pixel band around
the measured bounding box and a 1.2-megapixel region of interest.

The desktop and mobile editors draw the measured outline in blue on
the original and put an opaque rectification alongside the existing
untrimmed view. The saved geometry and any measured competing lines
stay available. Moving a corner hides the comparison until the next
save. A subsequent manual claim intersecting the recorded convex
measurement scope retires the preview and its image route. The
comparison supplies no edit action, rounded corner mask or export.

The U039 JPEG55 lower-car example used the exact saved P1.7 geometry.
With straight-edge agreement across encodings of the same capture,
excluding a half-pixel band and the corner neighbourhoods, the
comparison reduced omitted clearly inward source centres from 423 to
0 and retained clearly exterior centres from 8 to 0. Output pixels
receiving a clearly exterior interpolation contribution fell from
28 to 0. Output pixels sampling the uncertain band rose from 936 to
1,104. The complete physical corner shape and aspect remain
unverified. The normal upload, edit and reprocess path produced the
same 275 × 275 preview on reprocess and kept the original untrimmed
JPEG byte-identical. Both export modes and the cropped alias stayed
held, as did the geometry approval. The three active detections and
one deferred choice stayed the same. Across 25 warm single-thread
measurements on this source, the outline measurement took 49 ms
at the median and 74 ms at p95.

The development controls included 68 source variants with lighting,
compression and nearby prints. Known claims blocked all 30 close
neighbour placements. All seven visibly larger reprints blocked the
comparison, as did both protected-text losses of 12 and 14 source
centres. A larger sheet with no visible perimeter and no claim can
have the same local pixels. This evidence cannot establish its
physical identity. The eight benchmark runs matched the preceding
milestone on detections, metrics and mode availability. The frozen
synthetic counts stayed at 18/18 detections and 23/23 recoverable
crops. These examples are development evidence. The receipts,
source comparison and renders are under
`runs/crop-quality/u039-jpeg55/ribbed-comparison/`.

#### Recover a measured strip along a bowed image edge

An automatic ONNX crop can produce a transparent comparison when
both modes already need review for an unsupported paper edge.
The search requires one unsupported side and at least 0.90
support on the other three. No side may report competing line
fits. The shortest side must be at least 420 native pixels. An
existing geometry or preview suggestion takes precedence.

The search samples the original in half-pixel steps along the
normal. A measured station needs a nearby pale exterior and a
lightness drop towards the image. A connected path penalizes
across-edge jumps so that a strong album fold cannot win solely
by contrast. Verification needs 95% station coverage and 80% in
each of eight sections. At least 72% of stations must also have a
darker band farther out.
The longest supported path must cover at least 87% of the stations,
with measurements near both ends. Its cubic consistency check
allows 1.65 pixels of residual at p90 and at most 4 pixels.
The added area uses the observed stations. The cubic is
only a consistency check.

The comparison adds the measured image strip where the old edge
falls inside the image. It keeps a 1.35-pixel allowance inside
the observed transition and leaves the first and last 12 native
pixels near the corners as they were. It uses the existing
homography with an integer output canvas translation and
premultiplied-alpha sampling. A competing nearby paper claim or
protected-content violation vetoes the comparison. The API
retires the PNG and its trace when a later claim intersects its
measurement scope. Both named outputs stay in review.

On the original 2,560 × 1,920 Golden 12 right-hand print, the
adjacent pale band is 5–12 pixels wide. Near the bottom, a
stronger line follows the album fold. An independent optical
check uses opposed signed bands. The outer line continues above
the top of the photograph, which leaves physical paper ownership
unconfirmed.

| Source or sampling measurement | Previous view | Comparison |
| --- | ---: | ---: |
| Known scene centres omitted in the measured band | 1,604 | 75 |
| Additional source centres over the whole outline | 0 | 1,574 |
| Previously included source centres removed | 0 | 0 |
| Added centres in the independently classified exterior bands | 0 | 0 |
| Output sample positions reaching past the entire pale band | 4,770 | 4,828 |

The last row includes sampling at the existing lower overrun
and a small coverage difference from the transparent output
canvas. The comparison retains that overrun. It establishes
neither a physical aspect ratio nor a complete paper or motif
boundary. Across the pinned original, quarter turns, three JPEG
re-encodes and three blur strengths with fixed geometry, it adds
1,570–1,673 source centres and removes none. The measurement
and render take 79 ms median and 86 ms p95 in 21 warm one-thread
runs on the original.

The normal upload and reprocess path reproduced the original
JPEG and held geometry. It served a 703 × 994 comparison PNG,
which both the desktop and mobile clients displayed with the
measured source line. The test fixed orientation and disabled
OCR to isolate crop geometry. The existing download and review
approval gates held. A later manual claim through the measurement
scope retired the PNG, including its old asset route. The source
comparisons, browser examples and receipts are under
`runs/crop-quality/golden12-bowed-scene/`.
The eight benchmark pairs retain their detections, crop metrics
and mode availability. The only output difference is the intended
new preview metadata on this Golden 12 print. [Original source](../golden_fixtures/page_12.jpg)

#### Compare a measured exterior band beyond a pale rim

The bowed-edge comparison can also make a measured exterior
strip transparent. It searches 4–15 native pixels outward
from the observed image transition for a second, signed step
from the pale rim into a darker field. The field must agree
with a separate outward sample and sustain its color across
the intervening collar. The first and last 12 pixels near the
corners stay outside this operation. The supported run has to
be at least 100 native pixels long, and the comparison keeps a
2.4-pixel band outside the second transition as well as the
entire pale strip.

The final veto checks every native pixel centre in the proposed
removal against the local reference. For possible lettering or
ink, it samples opposing flanks along the edge. Four connected
centres darker than both flanks or with a coherent blue shift
suppress the removal. An intersecting protected region or a
larger closed perimeter measured on four native sides also vetoes
it. When the removal fails, the original add-only comparison
can still show the observed image extension. A neighbouring
paper claim in the recorded measurement scope retires the
comparison asset and both guides.

On the original Golden 12 source, the accepted run covers
native rows 1314–1437. The comparison gained zero new source
centres over the preceding add-only preview and excluded 557.
All 557 are beyond the paired pale-rim measurement. The
measured image and pale-rim centres remain included, as do
the previous 1,574 added centres. Independent sampling of the
actual alpha output finds 4,799 to 4,205 sampled output
positions with contributions beyond the whole pale rim.
Source centers outside the supported run, including the lower
fold, stay available. Both compared PNGs have the same
703 × 994 canvas, homography and fully opaque colors where
both include a pixel. At the cut's antialias, one nearly opaque
pixel shifts in color by one channel unit.

A visibly larger reproduction, a second paper edge, adjacent
photo texture and known protected content preserve the add-only
view. Nine of twelve faint-ink controls also preserve it.
The three weakest graphite or blue controls can still match
the nearby field closely enough to be hidden in the comparison.
JPEG compression and blur can also weaken marks enough to
defeat the veto.
Physically different surfaces with equivalent visible evidence
also remain possible. This is a provisional review asset and
both named exports and geometry approval remain held.

The original multipart upload and reprocess path reproduce the
primary JPEG and per-mode geometry. Desktop and mobile display
the new comparison and two guides. A subsequent claim through
the measurement scope retires the PNG. The source-pixel metrics,
controlled replays and screenshots are under
`runs/crop-quality/golden12-outside-band/`.

Eight paired benchmark runs preserve detection, crop metrics
and mode availability. Only the Golden 12 print's full-print
preview outline, second trace and exclusion flag change.
On that native crop with one OpenCV thread, 24 alternating
warm runs put the median preview measurement and render at
64 ms with the band check disabled and 114 ms with it enabled.
The median paired increment was 48 ms.

#### Preserve local marks in the exterior comparison

Before making the prospective exterior band transparent, PhoSca
also compares its source centres to a 7 × 7 spatial median in
OpenCV Lab. It keeps the original pixels if two connected centres
have a full Lab residual of at least eight units, or three
connected centres have a chroma residual of at least three units.
Connectivity includes diagonals. The checks complement the
earlier along-edge flanks for dark or blue marks. An isolated
residual can occur in paper grain where the median window crosses
the pale rim. When either check triggers, the comparison can
still add the measured image strip and keep all the original
edge pixels.

An independent counterfactual suite evaluated 357 fixed-proposal
captures. Under the previous check, 123 variants lost authored
mark centres still differing from the decoded background by at
least three units in one BGR channel. The new check retains
the marks in 13 of those variants and preserves 2,089 of those
centres. All 28 eligible shadow, grain and clean controls in the
suite keep their comparison, including the exact Golden 12
original. A separate sweep of tapered chromatic fibres vetoes
20 of 93 previously eligible exclusions, all on uncompressed
inputs. These constructed cases expose one ambiguity. A lower,
brightness-only threshold protected further marks while
suppressing the optional comparison on seven of the
28 eligible controls. It was left out. Larger, weak passages
and codec-attenuated marks can still go undetected.

In 101 alternating warm single-thread measurements, median
time inside the band consistency check rose from 2.30 to
2.39 ms. The full crop and export use their existing geometry
and review decisions.

Multipart upload of the exact original preserves the comparison
and the previous 557-centre exclusion. A lossless upload of the
authored blue-mark control preserves 322 previously hidden
marked source centres.
Reprocessing reproduces each preview and the primary JPEG.
The full-print and motif exports remain in review on both inputs.
All eight paired benchmark reports match the preceding results
for detection, crop metrics and review previews.

The lower fold on Golden 12 remains visible. In the dark area
past the thin pale line, the bottom image transition continues
within two native pixels of the projected line on 12 of 13
tested columns. A cut just outside the pale line there would
remove 1,633 currently visible source centres, including 863 in
the intervening dark area. The broad outer album field supplies
six positive-alpha output positions and zero with alpha at
least 250. A further cut through the intervening area has
insufficient evidence of physical ownership. The original,
rendered comparisons, and ledgers are under
`runs/crop-quality/golden12-fold-next/`.

#### Review a protection margin outside a manually added print

Off-print advisory revision 2 adds a path for a manually added print
with one uniquely matched, unverified page-level face or head detection.
The raw detector rectangle must be completely outside one side of the
quad, by at least 6 native pixels and 10% of the detector rectangle's
longer dimension. At most 6% of the expanded rectangle may intersect
the quad. A second crossing region prevents the advisory. The existing
content-removal check also applies when editing a previously saved crop.

The source check requires at least 90% line support across the
affected edge section and at each end, an opposing signed color
transition and a continuous rib in the intervening backing. A current
paper claim crossing the measured strip prevents the offer. At most
5.5 native pixels of disagreement may be shown for comparison. A
reviewer can confirm the mark after the measured section and the
submitted quad agree within 2 pixels. The saved quad stays as drawn
throughout. The check runs again on confirmation and reprocessing.
Only the extra margin of that particular detector hit is suspended for
that exact quad after manual confirmation. The source protection list
and raw detected rectangle stay in force. The decision is bound to
the source hash, protection digest, count revision and reviewed quad.
The original padded footprint stays attached to an unresolved crop.
If the protection entry disappears, reprocessing and changes to
unrelated corners keep any crop intersecting that footprint held.

On U008, the raw rectangle covers foliage in the next print and does
not overlap the proposed print. The first guide differs from the
locally measured edge by approximately 4.4–4.9 native pixels. A plain
manual add through the normal upload path now saves that lower guide
with both crop modes held. The active count changes from 3 to 4 after
the reviewer narrows the upper print and separately marks the lower
print. The existing pending offer remains. Confirmation of the first
guide returns 422. After separately aligning the edge and confirming
the mark, both modes still report `PRINT_ASPECT_UNCERTAIN`. Their
downloads and the cropped legacy alias return 404. The untrimmed
source returns 200 with the same results after reprocessing. The
measured edge is local evidence. A precise physical paper aspect
ratio remains unverified for this print. The projected preview is
294 × 298 pixels. The source-only pinhole check abstains. The page-plane
fit also abstains using all eleven development outlines, of which
one meets the existing source edge and corner requirements.

A flat separator, a contested gap, a core crossing and manual or
pinned protection remain blocked. A same-stock reproduction can copy
the local signals. The original and six of eight JPEG controls at
fixed geometry produced the advisory. The two white and dark
exact-pixel reproductions had the same result on those encodings.
The reviewer still checks physical count. The native optical check
took 10.2 ms median and 14.8 ms p95 over 20 warm samples.
Eight benchmark comparisons retained the preceding detection,
geometry and crop availability results.
The API receipts and comparison are under
`runs/crop-quality/manual-outside-margin-review/`.

#### Compare supported corner shapes on a manually added print

Manual corner review revision 1 is evaluated on manually added prints
with a short side of 150 to 650 native pixels. It requires the active
protected-region guard and no unresolved protection or paper boundary
conflict. For each corner, two exposed side bands must agree on the
warm backing chroma. At least 81% of each band must be within eight
CIELAB chroma units of the shared backing estimate. The source must
support a continuous transition on at least seven consecutive rays
out of twelve. The trace uses the measured positions and allows
departures from a fixed circle.

The comparison removes a smaller wedge behind each accepted trace,
with a 2.05-pixel clearance in the near-native local coordinate
system. The proposed wedges are sampled separately at half-pixel
spacing. At least 99.5% of those samples must match the backing within
eight CIELAB chroma units. At least 94% must match the extrapolated
rib phase within eleven lightness units. The phase comes from three
exposed columns outside the crop. The comparison needs at least two
supported corners. Other paper claims and protected regions veto
intersecting corner evidence. The separate RGBA preview uses the
existing premultiplied ownership sampler. The paper warp and original
JPEG are preserved. The comparison covers the recorded rectangle.
Unmeasured backing and visible paper beyond it require a separate
check.

The active record retains the corner hold across edits, source proof
loss and reprocessing. The named crop downloads and legacy cropped
alias remain unavailable, even when a paper size is supplied or all
four corners are moved inwards. An overlapping claim invalidates the
affected corner comparison, while a stale protection confirmation
also prevents access to that preview. The untrimmed preview is
available for review. The normal crop acceptance gates stay separate.

On the native U008 source the comparison supports the upper left,
upper right and lower left corners. A conservative optical agreement
mask from three separate chroma-threshold fits finds 114
backing pixels within the original marked rectangle and 70 still
present in the comparison. The number of output pixels sampling those
confirmed backing pixels falls from 166 to 115. The comparison loses
no source pixel within the rectangle that the mask identifies as
visible paper. The preliminary chroma mask flags 265 possible paper
pixels beyond the marked rectangle. Their ownership remains uncertain
under the [source-relative side check](#inspect-a-local-edge-section-outside-a-manual-crop).
The lower right corner, the rest of the outer contour and the paper
aspect remain unresolved. These numbers are optical development
measurements. Physical verification of the reference outline remains
pending.

The native helper agrees on the original and two of eight JPEG
controls at fixed geometry. A flat rounded design printed on a square
sheet and a square image reaching the paper edges trigger no offer.
A paper mount suppresses the affected corner. A facsimile with the
same local pixels can produce the same review. A low-quality JPEG
control of a bevel with perturbed corners shows one lost foreground
center in its comparison. Both export modes stay held.
The helper took 3.6 ms median and 6.6 ms p95 over 20 warm trials.
The eight benchmark comparisons match the preceding detection, crop
metrics and availability outcomes. Native upload, revision, continued
review and download status are recorded under
`runs/crop-quality/rounded-corner-review/`.

#### Inspect a local edge section outside a manual crop

Manual paper extent review revision 1 runs with the supported,
held comparison for a manually added print. For each side it
samples 144 stations over 13–87% of the side, along seven native
pixels either way in 0.25-pixel steps. It fits directed transitions
in the 8-bit CIELAB b-minus-a channel at levels 4, 6 and 8. Each
level needs a unique transition at 75% or more of the stations and
a 90th-percentile fit residual of at most one native pixel.

The second check compares the exterior chroma at offsets −7 to
−4.5 with the interior chroma at +3.5 to +6.5. It projects the
sample at −0.5 onto the local direction between those colours.
Each selected eighth of the sampled side needs 85% agreement on
the directed transitions and 65% agreement after the relative
chroma check. The view offers one contiguous interval on one side
with clear source claim and protection scopes. The line marks the
sample positions at −0.5. It represents an interval to inspect.
The recorded quadrilateral, baseline JPEG, rounded-corner PNG and
both crop-mode decisions are preserved.

The native U008 replay shows a 24.8-pixel interval near the upper
left, from approximately (95.11, 397.56) to (119.90, 396.46).
The independent sampling check found no sample with a relative
photo projection of at least 0.75 one native pixel beyond the
top, right or left side. The bottom has a localized mixed signal.
The backing rib remains visible there and a separate unsigned
line fit agrees with the current side within 0.20 pixels. These
measurements leave the ownership of the previously flagged 265
centres and the complete paper contour unresolved.

An exact synthetic boundary triggers no trace. Separate one- and
three-pixel insets on each of the four sides trigger a trace over
the same material. A claim intersecting
the measurement band vetoes it even when the two trace endpoints
are outside the claim. A borrowed rib, three shadow strengths and
the separate contact-sheet JPEG55 development image all abstain.
The original U008 fixture offers the interval and its eight
fixed-geometry JPEG re-encodes abstain. A facsimile containing the
same local source pixels can preserve the signal. The helper took
14.5 ms median and 34.1 ms p95 over 20 warm trials.

The native API replay saves the same manual crop and serves the
same baseline JPEG. A late edit to an existing neighbor invalidates
the previously confirmed protection corridor with the same count
revision. The current correction loses its validation receipt.
The corner PNG and edge trace then disappear from the detail,
variant and geometry review responses. The direct PNG, both named
downloads and the cropped alias return 404. The baseline review
JPEG remains available. In a separate native replay an upper claim
intersects the sampled band outside the marked interval. That edit
retires the interval and preserves the independent corner preview and
protection confirmation. Reprocessing reproduces that result. Eight
benchmark comparisons keep the preceding detection, crop scores and
availability outcomes. The receipts and reports are under
`runs/crop-quality/manual-paper-edge-review/`.

#### Review fabric along the capture frame

Fusion ownership revision 7 measures otherwise unresolved, aspect-invalid
model masks at the capture frame. It requires most of the mask to lie
outside a warm, connected album surface. A short side of the proposal
must show comparable chroma and high-frequency texture across its
boundary in the source raster. A coherent colour change or narrow quiet
paper rim vetoes the offer. The trace shows where the materials were
compared. A stock edge hidden by texture or compression can remain
ambiguous.

On the souvenir card page, three frame masks move from the active count
to reversible review, leaving the card and upper print among the
two original active proposals. Both lower-print offers remain. The
upload replay confirms the lower prints independently, dismisses the
three fabric proposals through count review and holds both crop
downloads after the right print is reprocessed. A separate restore
test keeps the second overlapping alternative open and rejects
restoring it over an active paper claim.

The full-size and 75%-size JPEG75 replays still defer all three fabric
proposals. The quarter-turn replay defers one and retains both matched
lower prints in the active set.

The production material check preserves all 108 controlled thin-rim
variants derived from this source, plus five existing NerdScan proposals.
A separate high-noise stress image can still lose the appearance of a
narrow rim after compression. Those decisions stay reversible. Local
receipts are under `runs/crop-quality/capture-fabric-review/`. The warm
median for the three-proposal material check was 0.21 s on one development
CPU thread during concurrent test runs. All eight comparison runs keep
their previous detection, crop and count-review results.

#### Show three open motif guides on the souvenir card

The uploaded development raster is 1200 × 1600 and has SHA-256
`08544646ef8aa6b383e708ca0115a6fa6dbbc8abc48dab7992b7632d6a853cdf`.
Glare and the construction of the souvenir card leave two plausible
upper paper claims. The outer paper boundary and metric aspect of
the right-hand proposal still require review.

In this combination of holds, PhoSca can show three inner transitions.
It searches up to 17% and 21% of each side's inward depth in the
provisional rectification. The two searches must select the same
three pale paper edges with a matching colour and fits at most 1.5
rectified pixels apart. The original raster must corroborate each
line continuously under the native seam check and without a competing
step. A neighboring paper claim that intersects the two-pixel buffer
around a guide vetoes the set. Adding or editing a neighboring claim
retires the stored guides immediately if it enters that buffer. On
the left, the provisional paper
warp starts at the observed picture transition, so the rectification
cannot establish a separate pale rim there. The editor receives only
open polylines. Its fourth side has no trace, and the held motif has no
proposed quad or rendered preview.

On this source the upper, right and lower guides each contain 12
measured knots. Median distances to a separate source-pixel line fit
are 0.07, 0.20 and 0.16 pixels. The largest local difference is 2.66
pixels on the right. The provisional warp has unresolved corners,
so these are guide agreement measurements. A JPEG75 replay and a
0.6-pixel Gaussian blur kept the three guides with the same provisional
paper transform. JPEG55 and the 0.9- and 1.3-pixel blur replays left the
guides empty. The guide check took a 56 ms warm single-thread CPU
median on the original, measured after rectification.

A caption, a ruled outline and a frame returning to blank paper
produce no guides. A decorative card with a printed scene and a
caption outside a measured edge can still produce three guides.
This optical evidence cannot assign either the caption or its paper to
a physical print. That control keeps both modes in review and
preserves the untrimmed image.

The upload replay exposes the three guides for the right-hand
proposal through geometry review, and reprocessing preserves them.
The caption proposal has no motif guides. Both crop downloads and the
motif preview remain unavailable, while the untrimmed review image is
available. The server rejects ordinary geometry approval without the
required correction. All eight frozen and development comparisons
keep their existing detection, crop and review metrics, with no newly
held usable crop. The recognition model and automatic eligibility
rules are unchanged. Local measurements and the annotated comparison
are under `runs/crop-quality/partial-motif-seams/`.

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

#### Trace the visible edges of a missing neighbour

On the 2560 × 1920 Golden 11 original, the mixed path finds both prints.
The classical fallback keeps one line-pair proposal at 0.9114 mask IoU
and misses the right print. The fallback's observed outline sits roughly
42–54 native pixels from the strong shared transition. JPEG recompression
can shorten the detected lines without removing the material edges from
the source. A separate transition from the album sheet to the floor
runs near the right print. The distance between those two transitions
is about 12.5 native pixels at the median and drops to roughly 2.5 pixels
at the lower end. An independent optical fit measures up to 7.23 pixels
of bow on the right paper edge.

Crossing-print revision 2 looks for a staggered neighbour when the
fallback holds a single line-pair detection and finds no adjacent
outline. A shorter survey line from the reduced image can seed a
longer native search. Four consecutive source fits on the bottom edge
and four along the far edge must have high support and meet within
two pixels at their joins. The top and shared scene transition also
need distributed native support. Three open, signed traces then
mark the visible transitions towards the album. The internal scene
transition supplies no display trace. The stage persists no estimated
quad, and leaves active count, automatic masks and crop rendering
unchanged.

On the original, both count views expose the same three traces with
65, 72 and 76 measured vertices. The reviewer can use them while
marking a separate print in the desktop editor. Saving an explicitly
marked second print consumes the pending trace and preserves the
curved right edge as review evidence. Entering diagnostic paper
dimensions of 70 × 100 mm and reprocessing still holds the whole-print,
motif and legacy trimmed downloads. The untrimmed inspection JPEG
remains available.

Holding the original left seed fixed, JPEG95, JPEG75 and JPEG55
all produce three observed sides. The source stage also finds the
three sides after each quarter turn, with some sides shown as
separate measured segments. Erasing the right print while retaining
the album edge produces no traces in the four quarter-turn checks.
The complete classical detector
on those unrotated recompressions can lose the left seed in its
own line-pair search, before the new stage runs. The original
reference is a development draft, and the physical ownership of
any potential overlap at the shared border stays in review.

The native trace search took 336 ms median and 467 ms p95 of CPU
time on the original over ten warm calls at one OpenCV thread.
The browser replay uses the actual upload, detector, geometry
and variant routes with OCR and automatic orientation disabled.
The paired controls reproduce 18/18 frozen detections, 11/11
whole-print and 12/12 motif passes with supplied seeds, and
29/29 Golden mixed matches. The full model-free Golden replay
retains 14/29 matches with 15 extra proposals. Detection and
crop-mode summaries match the disabled-trace arm.

#### Review source edges after an empty line-pair search

At the original 2560 × 1920 raster size, recompressing Golden 11 at
JPEG95, JPEG75 or JPEG55 can remove the left line-pair detection. The
detected lower line then starts at approximately x895. Its native
paper intersection is near (819, 1750) across the original and all
three recompressions. The x782 estimate was an average of the slanted
left rail. The source at that position is pale album stock. A narrow
shadow lowers the local exterior brightness without removing the
paper-to-album transition.

The additional search runs in the precise fallback when the normal
detectors returned no candidate and recorded a rejected narrow or
frame-touching contour. Two rails and two crossbars supply a
perspective-aware search hypothesis. The source needs a material step
on every side, pale exterior evidence on three, and four supported
sides. At least two neighbouring corners need signed transitions on
both arms that terminate at the corner. A single draft and measured
source traces on a potential neighbour are required to offer it in
count review. The projective hypothesis itself creates no photo row.

In the three unrotated recompressions the review keeps the automatic
count at zero, offers one violet left outline and shows the observed
right edges as blue traces. The user can add the left draft and
mark the right print in the desktop editor. Saving the left draft
preserves the right trace until its print is separately added. The
first save is a count decision. A separate geometry save must
confirm the paper corners and pass native edge and corner support
before the left crop can export. Reopening or reprocessing a held
row preserves that check. The right curvature uses the existing
source-trace guard.

The left draft reaches 0.9937, 0.9930 and 0.9925 native mask IoU
against the development annotation in JPEG95, JPEG75 and JPEG55.
The draft's maximum corner errors against that annotation are 3.08,
3.41 and 3.51 px. Each of these runs starts with zero active rows.
In a separate three-quarter-turn JPEG75 control, the 90° and 270°
runs still offer no source draft. The 180° run has one classical
proposal and insufficient source evidence for the neighbour.

Through the real JPEG75 upload and desktop editor, the active list
goes from zero to one to two after the two manual additions. The
temporary paper dimensions are 70 × 100 mm, and OCR and automatic
orientation are disabled. The original bytes and the saved right
curve are retained. Both reviewed rows give HTTP 404 for whole print,
motif and the legacy trimmed alias, and HTTP 200 for the untrimmed
review JPEG after reprocessing. The existing left draft remains held
when its original corners are resubmitted as a geometry confirmation.

A source inspection found up to 4.9 native pixels of omitted paper
at the estimated lower-left corner in the recompressions. This
excludes immediate rendering approval from that draft. On a
controlled near-registered overlay, a one- or two-pixel step still
triggered the projective outline in all eight source and JPEG
variants. A four- or seven-pixel offset and two visibly different
overlaid stocks abstained in all 16. A pixel-identical reproduction
on one sheet still produces the evidence for manual review.

The measured search for the draft and neighbour takes about
0.49–0.55 s in individual one-thread measurements on the three
recompressions. The full classical runs take 1.69–2.22 s of CPU time
in the local comparison. The locked model and model-free replays,
seeded geometry and Golden comparisons reproduce their
instance and both mode metrics. No extra active proposal is added
by the new count review path.

#### Recover source traces on rotated JPEG uploads

The three quarter-turn controls above reproduce two distinct
line-discovery errors. At 90° and 180°, the candidate's shared rail
ends about 181 native pixels before the lower crossbar. At 270°,
the outer rail incorporates both the paper edge and the album edge.
It predicts the upper endpoint roughly 19 pixels too far out and
the lower endpoint roughly 16 pixels too far in.

The second search runs when the straight-rail search yields no
supported trace set. It permits a short, locally corroborated
extension of the shared rail only if the exposed part past the
known print shows a signed pale-backing-to-print transition.
For the far rail, the top and bottom crossbars each need a unique
termination of the light-to-dark step in the native source. Local
line support and the three free-side traces are then verified
again. The shared side supplies context for the search. The
overlay shows only measured spans on the three free sides.

| Quarter-turn before JPEG75 encoding | Active proposals | Deferred left outlines before → after | Right trace runs before → after |
| --- | ---: | ---: | ---: |
| 0° | 0 | 1 → 1 | 4 → 4 |
| 90° | 0 | 0 → 1 | 0 → 4 |
| 180° | 1 | 0 → 0 | 0 → 4 |
| 270° | 0 | 0 → 1 | 0 → 4 |

In a subsequent before-and-after replay, JPEG95 gains the draft
and right traces at 90°, 180° and 270°. JPEG55 gains them at
180° and keeps the existing hints at 0° and 270°. In that
comparison the JPEG55 90° case also had no left projective
outline. All eight replays preserve the original active detections.

The left outline in the 90° and 270° rotated runs has approximately
0.9937 and 0.9935 native mask IoU against the Golden development
annotation. The three recovered trace sets enter count review and
the number of active detections stays the same. Over five warm
one-thread measurements per input, the median search takes
0.50–0.66 s for the three recovered variants and about 0.27 s in
the existing upright case.
Against separately fitted native source curves, the measured
right-print runs have p95 deviations of 0.60–0.78 px on the top,
0.95–1.08 px on the far side, and 1.07–1.28 px on the bottom.
The far-side run stops 129–165 px short of the lower corner
across these variants. The gap remains open in the overlay.

The 90° JPEG went through actual upload and analysis with no ONNX
detector, automatic orientation or OCR. Desktop and mobile show
the violet draft and the four blue spans in the uploaded raster's
coordinates. On desktop the two manual additions bring the count
from zero to two. The second save keeps the left record and its
preview intact and stores the measured curve on the right print.
After reprocessing both rows, whole print, motif and the legacy
trimmed alias each return 404. The untrimmed review JPEGs return
200. Both geometry approval requests return 409. The diagnostic
100 × 70 mm entries and the Golden annotation are development
inputs. Release-level aspect, landmarks and content-loss scores
still need independently verified references for both modes.
Separate 180° and 270° uploads show the matching overlays in the
mobile UI. Inspecting them writes no new photo row.

With the known left outline fixed, eight right-side negative
controls across the four turns give zero traces in 32 runs. They
include an exterior holder, an interior pale stripe, two corner
covers and an empty leaf with the album edge still present.
In the full pipeline a dark cover over the left corner suppresses
the deferred outline at all four turns. An almost registered
layer offset by two native pixels can still show a draft at all
four turns. Each of those runs has zero active proposals. The
visible traces do not establish whether two layers share a
physical sheet.

The frozen model run retains 18/18 matches. With seeded geometry,
11/11 recoverable whole prints and 12/12 recoverable motifs pass.
The Golden runs retain 29/29 mixed-model matches and 14/29 classical
matches with 15 extra classical proposals. The frozen classical run
retains 13/18 matches and six extras. Detection, rendered crop and
mode availability metrics match the preceding runs.

#### Show only the measured neighbour edges at JPEG55

For the remaining JPEG55 90° variant, the 1280-pixel working
raster has 13 viable joined lines. The shared rail spans about
47.5% of the relevant side, below the half-side coarse minimum.
The first bank completes enumeration with zero geometric seeds.
A 1024-pixel bank joins the observed rails differently and gives
one fit under the existing native paper, side and terminal checks.
A budget veto, a rejected native fit, a competing hypothesis or
existing detection keeps its original decision.

The 1024-pixel fit supplies temporary context for another
original-pixel edge search. Only the four supported open spans
of the right print go to review. Against separately fitted native
curves, their p95 distances are 0.59–1.05 px. The long right span
ends near y1687 while the lower corner lies near y1834 in the
upright original. A 1–2 px near-registered layer on the left can
produce the same contextual fit. It leaves the right trace
locations on visible source transitions, so only those locations
are shown. The original source review still decides physical
ownership. The 19 paired development checks preserve the active
rows and fusion evidence. Covered or removed right prints and
fabricated holders gain zero traces. The alternate source search
takes about 0.62 s median in five warm one-thread samples.

The model-free JPEG55 upload shows the blue spans with zero
active rows and zero deferred outlines. Desktop and mobile show
the same points on the uploaded page. Opening or closing the
traces leaves the review record untouched. Manually marking the
right print increases the count to one, consumes the displayed
spans and stores the curve verification in the photo record. After
reprocessing, the two named exports and the legacy trimmed alias
return 404. The untrimmed review JPEG returns 200. The two
visible prints on this page come from one development capture.
The full matrix of JPEG55, JPEG75 and JPEG95 across four quarter
turns now shows measured neighbour edges in 12/12 runs. The other
11 retain their previous active counts, deferred outlines and
trace lengths. The five frozen and Golden replays preserve
detection and both crop-mode metrics.

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

### Train a detector from architecture without inherited weights

The procedural stock corpus already has an auditable split. The
two local builds reproduce all 144 JPEG and polygon pairs in
`detector-stock-experiment-sources.csv`. There are 96 training
images, 24 validation images for checkpoint choice and 24 separate
challenge images. The inherited synthetic-control checkpoint used
a COCO-derived base and included two legacy camera pages in its
validation split. The new trial uses only the procedural stock
training and validation images.

Load the architecture with `YOLO("yolo11n-seg.yaml", task="segment")`.
Train on CPU for 12 epochs at 512 pixels and batch size 8, with
`workers=0`, `seed=0`, `deterministic=True`, `pretrained=False`,
`resume=False` and `amp=False`. This trial uses Ultralytics
8.3.0, Torch 2.7.1+cpu, NumPy 2.2.6 and OpenCV 4.10.0.
The architecture digest is
`13215703e646f21dff3eb159b421fac6657903b40bd9981f6e3583b1d291c07f`.
The initial tensor digest is
`e3fd8231d60829127a0243b13fd21eff4393ba837f9596d645bdc17d999bcba4`.
The first attempt was killed before the first batch under memory
pressure. After closing completed QA processes, restarting from
the architecture reproduced the same initial tensor digest.

The best checkpoint digest is
`dd4813f9a3c1d87d2d894e472c88602716a12a4f047b543973286906122c76b8`.
The fixed 512-pixel ONNX export uses opset 17 and has digest
`fd656186dd486283486533ac24a704d92c5ae9375f2ba29a5379d7cd94107b35`.
At the default proposal threshold of 0.25, the exported model
finds zero of the 46 expected prints on the separate stock
challenge. Inspecting the exported outputs on all 24 challenge
pages gives a maximum raw score of 0.11763.

| Diagnostic | Correct / expected | Extra |
| --- | ---: | ---: |
| Stock challenge, model only | 0/46 | 0 |
| Stock challenge, with fallback | 11/46 | 21 |
| Frozen detection, with fallback | 13/18 | 6 |
| Golden development, with fallback | 14/29 | 15 |
| Original camera contact sheet, with fallback | 1/1 | 1 |
| Domoritz development, with fallback | 4/4 | 0 |

The candidate falls short of the existing frozen and Golden
baselines. Its frozen result matches the model-free control in
both crop modes. The training and export can be repeated from
documented inputs. The candidate stays outside the product
registry. A future model selection needs the complete inventory
of all sources used in fitting or selection and the independent
calibration and release checks for the selected configuration.

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

Another scratch trial allows one trusted flank to face an independently
detected neighboring print. The new side has to extend outwards, leave
all existing source pixels in the crop and add only measured matching
stock. On the frozen left `touching` print, maximum corner error falls
from 5.469 to 0.624 native pixels, omitted centres beyond one pixel fall
from 1,320 to zero, and foreign source contributions fall from 99 to zero.
Both modes remain in review. The trial preserved all other 56 rendered
outputs and the detector assignments. No independent real print was
improved. In 160 blind layouts the trial made no corrections. Twenty
reached the final pixel checks and failed on insufficient outward gain,
neighbor ownership or incompatible stock colour. This branch remains
exploratory.

After unblinding, a subpixel-phase stress test found several successful
warps on one generated layout. A follow-up placed a distinct stock
strip along the added edge with a faint visible seam. The same rule
accepted 28/48 derivatives and retained 4,946–7,979 foreign strip
centres in those results. The trial remains outside the crop path.

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

#### Offer editable edges for distinct adjacent paper stocks

For one held crop with a single nearby proposal, each physical
paper hypothesis needs two or three independently supported sides.
PhoSca learns the two pale paper tones from their own fitted edges
and looks for the third, local album tone. The rim-aware source
search tests the uncertain sides. At least part of each outline
must border the other print's independently measured stock.
Both final outlines need supported flanks around their full lengths
and at both ends of each side. The hypotheses must be disjoint.
The candidate preserves the existing outline within a native
pixel tolerance and clears other active masks and protected pixels.
An invalid peer or an existing ownership review takes priority.

The resulting outline is editable in desktop and mobile review.
The committed warp, original count, competing-edge spans and
untrimmed JPEGs are unchanged until an explicit paper confirmation.
In the frozen differently toned pair, the two editable candidates
have maximum corner errors of 0.624 and 0.522 px. They recover
1,655 and 1,158 native source centres in the known synthetic
references, with zero added foreign or neighbour centres there.
The model-free detection with just one row and geometries already
ready in the mixed and seeded controls keep their original paths.

The normal upload with the synthetic-control model returns two
independent held rows. Opening and loading the proposed outlines
writes nothing. General approval and a corner request without the
paper confirmation are rejected. Saving the left with the known
166 × 114 mm test dimensions opens its named exports. The right
record and its untrimmed JPEG are byte-identical before and after,
and its exports stay in review.

| Left after explicit correction | Footprint IoU | Maximum corner error | Landmark p95 |
| --- | ---: | ---: | ---: |
| Whole print | 0.99844 | 0.624 px | 0.523 px |
| Motif | 0.99187 | 0.093 px | 0.065 px |

The manual renders have zero measured aspect error, zero
content cutoff beyond one source pixel and zero foreign
interpolation contributors in this test. The isolated paired
fit takes 28 ms median and 33 ms p95 over 20 warm measurements
on one OpenCV thread.

On the 66 paired development renders, the candidate output changes
only two editable cue records. Existing previews and both mode
statuses reproduce. Occlusion, same-tone stock, competing ownership
and malformed-peer checks abstain. In the same-palette strip
controls, eight of 16 inputs can still offer an outline. These
include synthetic examples with 331–993 centres belonging to a
separate strip. A second visible colour may locate an edge while
the reviewer still has to assign the added area to a physical print.
The tested Golden ambiguous cases also have other held reasons
and do not meet this path's entry condition.

#### Preview an inboard motif beside a disputed paper edge

Once the adjacent-paper check produces an editable outline for a
held print, PhoSca can inspect the motif in the stored, slightly
smaller warp. Four physical seams must trace continuously in the
original. The three-pixel native search must support every sampled
side and corner, and the four motif corners need at least 3 px
clearance inside the stored outline. The candidate must be rectangular
in that plane, with a continuous paper rim and no conflicting visible
neighbor or protected content. The motif renderer samples only the
verified inner quadrilateral.

The marking guard examines both the stored warp and the possible
extended paper. The extended projection is used for evidence and
adds no whole-print preview. A thin dark line between two similar
paper shoulders can also block the crop. In development, a faint
185-pixel printed rule and a three-row date with 58 marked centres
entirely in the added strip both passed the ordinary edge checks.
The corresponding margin evidence now withholds the tight motif
preview. Fainter and smaller added marks can still lack enough
contrast to be detected. Four of 60 synthetic illumination controls
with narrow symmetric shadows also lose the optional preview. Both
crop modes were already in review for those inputs.

| Frozen `touching` motif preview | Left print | Right print |
| --- | ---: | ---: |
| Output size | 308 × 204 | 264 × 182 |
| Source footprint IoU | 0.991867 | 0.990738 |
| Maximum corner error | 0.089 px | 0.077 px |
| Landmark p95 | 0.064 px | 0.065 px |
| Relative aspect error | 0% | 0% |
| Omitted centres beyond one source pixel | 0 | 0 |
| Output pixels with foreign contributions | 0 | 0 |

The clean multipart upload preserves both stored whole-print
JPEGs, paper outlines and detection count. Both clients show the
new motif PNG with the instruction to confirm the external paper
outline first. Opening the comparison performs no write. Normal
reprocessing reproduces the PNG bytes. A later intersecting claim
retires the affected motif preview, including its old URL. The
disjoint peer retains its own PNG. Both named exports and the trimmed
alias remain held during this workflow.

Eight warm A/B samples per print on one OpenCV thread put median
crop time at 59.1 and 53.7 ms before the preview path and 171.1
and 148.8 ms with the source checks and render. The new p95 times
are 200.2 and 189.2 ms. The provisional fit contributes no
border-trim confidence to the existing crop decision.

Same-colour touching and overlapping controls keep their previous
mode results. The real transfer run yields no new motif preview
among 37 proposals across seven native views and two half-resolution
Willett views. An original-size Willett view is included in both
its native and reduced form. The other Willett native run was
terminated by the execution environment and replayed at half
resolution.
The Domoritz control has four plausible motif edges in the
rectified image and a successful native straight-line fit, but
only two continuous source spans. It remains a negative example
for accepting a closed motif based on straight-line support.
`--score-review-previews` reports the two new controlled results
separately from the delivered-mode scores. All eight paired benchmarks
retain their detection, delivered crop and review metrics. The frozen
detector keeps 18/18 matches and the seeded run passes all 23
recoverable mode cases.
The independent source controls and browser receipts are under
`runs/crop-quality/adjacent-motif-preview/final/`.
[Crop and source checks](../../backend/app/pipeline/perspective.py) ·
[Marking checks](../../backend/app/pipeline/motif_margin.py) ·
[Pixel regression](../../backend/tests/test_pipeline/test_precise_crop_modes.py)

#### Locate a diagonal detection cutoff on Golden 05

In the model-free replay, two proposals match four draft reference
prints at mask IoU 0.85, with three extra proposals. On the
upper-left print, the lower-right vertex lies about 289 pixels inside
the reference. The original detector already places it there. The
renderer changes that vertex by less than two pixels. Against the
development outline, the original render omits 144,277 of 641,447
visible source centres. The mixed run detects all four prints and
omits 2,109 centres on the same upper-left print.

The nearest upper pair has roughly 263–266 native pixels of
backing between the prints. The proposed paired-stock check lacks
the shared paper evidence in this case. An isolated source-fit
diagnostic starts from the truncated proposal. The retained
lower-left corner and upper direction define a bottom search
window. The short right side gives a direction for the second
window. At original resolution, a light-to-dark transition
supports 225/256 bottom stations and 256/256 right stations.
Both arms near the recovered lower-right corner show the same
local transition. The existing top contour is close to a separate
dark album rail.

The diagnostic alternate reduces the maximum corner error against
the draft from 289.3 to 3.68 px. It reduces omitted centres to
913, includes 2,733 foreign centres relative to that draft and
has no measured intersection with the other three proposals.
The native source and four compressed or blurred development
variants produce supported fits. A covered corner, removed paper
and a dark added strip fail the combined side and corner checks.
This trial uses the same already observed proposal as the
starting geometry. The production path and instance count
remain as measured above. The metric aspect and final ownership
need separate proof.

#### Offer a reviewed preview for the inset corner on Golden 05

On the model-free upload path, the target is a classical four-sided
edge-contour prediction. Its two opposing sides have full source
support. Its lower diagonal has 0.232 support and its upper edge
has two plausible traces near a dark album rail. PhoSca searches for
the missing corner only when the truncated side is substantially
shorter than its opposite side and both already observed directions
support a diagonal truncation. Both new lines must follow a signed
light-to-dark source transition in every quarter of their search
windows. The two transitions have to reach their intersection and
end there. The preview also checks that the expanded outline retains
protected content and clears the other active print claims.

| Whole-print source footprint against the Golden 05 development draft | Stored crop | Review preview |
| --- | ---: | ---: |
| IoU | 0.771689 | 0.993180 |
| Omitted centres out of 641,447 | 144,277 | 825 |
| Centres outside the draft | 2,815 | 3,574 |
| Maximum corner error at source scale | 289.304 px | 4.069 px |

The 996 × 658 preview uses the measured lines and the retained
opposite sides. It includes all previously retained draft-owned
centres. The table measures the projected source footprint against a
development draft. The disputed upper edge, physical aspect and
interpolation accuracy of a finished crop still need a verified
reference. A printed rectangle on wider paper, or a holder flush
with the observed transition, can generate the same evidence. The
editor asks the reviewer to verify that the added area belongs to
the same physical print before saving a paper correction.

The normal upload, desktop and mobile editor and image endpoints
were exercised with the tracked JPEG and no detector model. Five
proposals remain. Exactly one offers the second preview alongside
the stored result, and the two existing upper-edge traces stay
visible. Loading the draft performs no write. A save without paper
confirmation and generic approval return HTTP 409. Both named
exports and the legacy trimmed route return HTTP 404. When a new
manual print claims an area wholly inside the added corner, the
PNG and its editable outline are retired, including after
reprocessing. The stored preview and the original traces persist.

Independent source controls preserve the ordinary output arrays and
both modes' review states. The extra preview appears in the native,
JPEG75 and lightly blurred fixed-geometry replays. It is absent in
all six visible-gap trials and all nine crossing, matte and holder
trials with interrupted edge evidence. A flush transition can still
produce a preview in all 17 same-edge trials. Across 16 complete
model-free reanalyses at four quarter turns and four encoding
conditions, eight retain a nearly complete upper-left proposal
already (polygon IoU at least 0.949 on the 256-pixel comparison
raster). Five get this additional outline.
Three retain a weak or incomplete proposal without a correction.
The fixed-seed matrix isolates the corner measurement and produces
14 previews across 16 replays. In 16 paired one-thread runs of the
original case, measuring the corner took 15 ms at the median and
17 ms at p95. The full preview adds 26 ms at the median and 38 ms
at p95 to the affected crop before asset encoding. The frozen mixed
detector retains 18/18 matches, the seeded crops retain 11/11
whole-print and 12/12 motif passes, and the mixed Golden replay
retains 29/29 matches.
The classical Golden replay still has 14/29 matches with 15 extras.

#### Isolate the missing lower-right instance on Golden 05

The model-free detector emits an oblique left fragment and a
rotated rectangle that extends to the bottom of the capture. Their
mask IoUs against the lower-right print draft are about 0.45 and
0.39. Both originate in edge contours, and both survive the later
selection stages. The expected outer contour is already missing
from the prepared-image candidates. On the unprepared capture it
is present. In component isolation a BGR-to-Lab-to-BGR round trip
alone lowers the best initial proposal from 0.969 to 0.533 at a
1,024-pixel comparison resolution. Bilateral filtering alone
preserves the 0.969 contour. The combined production preparation
and classical selection leave the strongest fragment at 0.450.

A shadow test fed original pixels into the existing classical
detector, keeping its options and all later stages. This makes
Golden 05 a perfect detection page with four matches and zero
extra proposals. Across Golden it adds two matches and reduces
extras from 15 to nine, while losing a detection on Golden 11.
The frozen model-free run falls from 13 to 12 matches and from
7/11 to 4/11 whole-print passes and 8/12 to 5/12 motif passes.
The shadow configuration is rejected. The frozen mixed model
still has 18/18 matches and 6/11 and 7/12 passing crops.

A source-only diagnostic searches the padded union of the two
observed fragments. Canny on the original grayscale crop followed
by the existing contour closing and quadrilateral extraction
finds an outline with 0.991 mask IoU. The same four points appear
at four paddings from 40 to 144 source pixels and under each of
the three tested threshold and closing combinations. The ROI
pass takes about 20–24 ms on one OpenCV thread in four trials.
Neither the annotated outline nor the detector model is used as
input. An independent signed fit supports the top, left and
bottom along their length. The upper part of the right side
has weaker evidence, and covering the lower-right corner can
leave the long fitted lines in place. The original corner pixels
must independently support the proposal.

The candidate contains almost all of the first fragment and about
91% of the frame-touching second fragment. Both are required to
explain the count error. The existing fragment review requires
at least 93% of each fragment after fitting and a separate
continuous-scene divider test, which also fails here. A distinct
clipped print in the same crossing geometry has the same overlap
but reveals two supported, intersecting arms in its visible part.
This control rules out choosing the replacement by overlap or
requiring a complete competing perimeter. The real lower-left
neighbor and a separate narrow print with a gap to the right
have zero overlap with the recovered candidate. A third print
crossing the proposed boundary has its own supported perimeter.
The source search and ownership controls are development trials.
The next product step is a single editable group suggestion that
keeps both underlying observations until the reviewer resolves
the physical count. No count or crop change from this diagnostic
is enabled.

#### Review the lower-right fragments together

The precise model-free path can now offer one shared outline on
Golden 05. It only searches when one weak classical rectangle
reaches the capture boundary and a separate partly supported
contour overlaps it. The first fragment must have three supported
side lines and corners and one weak internal edge. The second
must have enough observable span to check every edge. Its own
supported or ambiguous edge stops the group search. The two
observations still count separately at this stage.

The search starts in a padded union of the two raw proposals.
It extracts coarse quads from the original grayscale image and
uses the supported rails of the first fragment to initialize an
independent fit. The native pass samples a signed album-to-image
step on each side. Three sides need at least 231 of 256 supported
stations with coverage across all quarters. The candidate's weak
side needs 181 stations and reliable support across three quarters.
Both arms at each newly located corner must present a transition
and terminate. A substantial third claim, a protected cut or
competing outlines block the group. The short-dashed review edge
marks the weaker side. A printed rectangle on larger pale paper
can still produce the observed transitions, so the reviewer must
confirm the physical count from the original.

The original upper-left corner preview remains separate. For the
lower-right print the two current mask IoUs are about 0.449 and
0.388. The shared draft reaches 0.99341 native footprint IoU with
5.20 px maximum corner error against the development outline. It
omits 2,161 of 700,861 draft-owned centres and includes 2,474 centres
outside the draft. The complete suggestion takes 98 ms at the median
and 110 ms at p95 over 14 warm runs on one OpenCV thread.
When the original fragment shapes are reused, the source fit
produces a proposed outline on all 16 tested rotations and
encodings. Across 16 complete reanalyses, only the upright original
and upright JPEG95 produce the required group. The other encodings
and quarter turns can produce different fragments. A disjoint narrow
paper remains separate. A planted clipped print with the same
apparent overlap, an independent third claim and a covered corner
all suppress the group in controlled replays. A small independent
print entirely inside the candidate can still evade this ownership
check if the original fragment shapes are preserved and the
independent print has no active claim. Fresh detection of the tested
overlays changed the fragment graph and produced no shared proposal.
The added small prints had detection overlap below 0.13 IoU.

Desktop and mobile load the proposed corners into the count
editor. The two purple outlines identify the exact rows being
replaced. Saving needs explicit confirmation and a current page
revision. The server checks both fragments against the stored
group, tests the submitted corners against the measured outline
and refuses overlapping claims. The original JPEG upload has five
active rows before the save and four after it. The three unrelated
rows are preserved. An identical retry returns the same replacement
and a failed render rolls back the entire edit. Editing or rejecting
a member or adding an intersecting manual claim retires the offer.
Both named cropped exports and the legacy trimmed route stay held
for the replacement, including after repeat processing. An explicit
geometry confirmation clears that hold only when the native boundary
checks pass.

The five paired detector and crop regressions retain their prior
metrics, including 18/18 frozen mixed matches, 11/11 and 12/12
seeded mode passes, 29/29 mixed Golden matches and 14/29 automatic
classical Golden matches. The two targeted Golden 05 fragments were
already in review. The group adds a shared reason to both modes.

### Compare the lower paper edge on Golden 05 with the model enabled

The registered mixed run finds all four prints. The lower-right
proposal descends about 38 original pixels past a strong dark-to-light
edge near its right corner. The pale material in that wedge continues
into the stacked album leaf. The left end is already near the observed
transition. The crop has four existing hold reasons for the uncertain
outer rim, unsupported boundary, uncertain corners and unverified aspect.

The comparison first requires exactly one side with a partial stock
fit. It searches that side on the original within 6.7% of the shortest
side, up to 80 pixels, using the existing normal-gradient line fit.
The line needs 97% support, 90% corner support and at most 1.25 px
residual. It must differ substantially at one end and stay near the
stored edge at the other. The signed lightness drop must hold over
94% of samples and 88% in each quarter. The transition must stop
beyond both ends, and the two adjoining sides must have transverse
contrast near their corners. A competing bright-paper transition
repeated outside the candidate, visible margin marks, other active
print claims and the normal protected-content check can veto the
alternate render.

The coarse left-neighbor mask shares 3,428 native centres with both
the stored crop and the comparison. The shared centres are identical.
Two original-facing transitions bound a pale gutter 15.8–21.3 px
wide across that span. The lower corner of the left print also has
observable contrast. The comparison tests only one narrow automatic
claim and requires membership in the previous crop over the entire
intersection. A manual claim, a wider overlap, an interrupted gutter
or preview growth into a neighbor retains the renderer's veto. The
underlying claim remains in force for both crops. The caption asks
for a separate look at the neighboring edge.

On the original the measured line ends near (2,331.1, 1,829.8) and
(1,310.0, 1,851.2). The comparison uses a one-native-pixel outward
guard along this line. The other two corners keep their positions.
The source footprint preserves every draft-owned centre in the
current crop, regains 50 more and reduces centres outside the draft
from 25,370 to 7,956. The draft is an approximate outline from the
original capture. It does not establish the paper's physical aspect.

The source line and both corners hold through fixed-proposal
JPEG55, JPEG75 and blur replays. A continuing dark rail, an obscured
corner and a simulated pale extension carrying a date abstain.
Some quiet pale paper extensions still produce the same inner
transition. The original remains available beside the comparison.
Both final modes keep their hold independently of the new preview.

| Rendered comparison against the Golden development reference | Stored crop | Review preview |
| --- | ---: | ---: |
| Source footprint IoU | 0.956684 | 0.980317 |
| Draft centres omitted beyond one pixel | 6,003 | 5,980 |
| Output pixels with foreign contributions beyond one pixel | 24,223 | 6,526 |
| Output samples outside the source | 0 | 0 |

The 1,022 × 700 PNG has an estimated aspect. Verified corners, dimensions
and landmarks are still needed for those acceptance measures. The normal
multipart upload retains four active rows. Both clients display the
original trace and the two rectified versions with their comparison
caption. The previous images and both mode geometries match the baseline
through reprocessing. The comparison URL serves the PNG and both named
downloads stay held. Adding an overlapping manual claim retires the offer,
including its old image URL and through another reprocess. All eight
benchmark pairs reproduce their delivered crop, detection and review
metrics. The frozen detector matches all 18 prints and the seeded run
passes all 23 recoverable crops. The added comparison step takes 55.6 ms
median and 67.2 ms p95 across 30 warm one-thread runs. That timing covers
the source fit, guards and preview render with the original crop already
available. The source, browser receipts and reports are under
`runs/crop-quality/golden05-paper-edge-comparison/final/`.
[Native fixture](../golden_fixtures/page_05.jpg) ·
[Measurement and rendering](../../backend/app/pipeline/print_outline_suggestion.py) ·
[Source and covered-corner test](../../backend/tests/test_pipeline/test_slanted_paper_preview.py)

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

#### Keep the interrupted-edge check across manual edits

On `062feff`, saving the same four Domoritz corners erased the source
spans and the unsupported-edge reason. Without measurements, aspect
uncertainty still held both modes. Entering test dimensions of 100 ×
140 mm made the unchanged whole print downloadable. Its lower-right
corner was still about ten native pixels outside the observed edge.
Reprocessing reproduced both results.

The editor now records the original warp and source spans on the first
manual save. If the edit still belongs to the same print, one side
must cover all measured endpoints within two native pixels. The check
persists through repeated edits and reprocessing, and the download path
checks the current receipt. A manual orientation or aspect change
leaves the disputed side in review. Deliberately moving the edge onto
the observed spans clears this specific hold. Aspect, motif and source
sampling still run normally. An undo restores the hold if the edge
again misses the spans.

The normal upload, edit and reprocess replay confirms both holds and
both corrections with and without test dimensions. Across the four
measured endpoints, maximum deviation falls from 9.657 native pixels
to less than 0.001 pixel after the explicit edit. The untouched blur
between the measured spans still needs inspection by the editor.
Against the unblurred source-line development model, the whole-print
footprint IoU in the no-size arm rises from 0.9756 to 0.9875 and foreign
samples beyond one native pixel fall from 1,484 to zero. Omitted source
centres beyond one pixel increase from six to nine. The motif remains a
manual result without an independently verified motif reference here.
The stored-reference association and side agreement took a warm median
of 0.124 ms and p95 of 0.192 ms across 100 one-thread calls.

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

#### Save a reviewed mounted print from its fragment proposals

For the possible mounted whole, the desktop editor shows the
fragment polygons captured with the candidate. The checkbox
starts unchecked. Selecting it and saving the four corners
explicitly replaces those fragment rows in the active count.
The server locks the page, checks the count revision and exact
fragment polygons, and requires each fragment to lie inside the
reviewed outline. Already edited or downloadable rows cause
a conflict. The new photograph and the excluded snapshots are
written together, with an audit link from each fragment. A
failed render rolls back all changes. Repeating the exact request
returns the same record. Restoring a fragment reopens conflicting
downloads for review.

On the original Willett page 6, the action turns six active
proposals into one manually saved whole and the same unrelated
proposal. The five fragment images and masks stay in history.
The original model output and automatic count remain the same.
The initial rectified JPEG is still useful for review. It shows
black photo corners and an unwanted sliver of album backing.
The count correction keeps both named modes and the legacy
trimmed export held. A separate geometry save must confirm
the paper outline, with all four native side fits at 90%
support and corner support at 80% or higher. The maximum
change to the marked corners in that fit is one native pixel.
The existing paper coverage and motif checks still apply. The
regression can export both modes for a source with four clear
edges after this confirmation.

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

#### Review a disputed paper join

The pinned native Willett page 109 is 3491 × 2325. At mask IoU 0.85,
the model-only and classical-only arms each match both photographs and
emit three extra proposals. The mixed arm matches both and emits two
extras, one from each detector. The folded vehicle photograph has
strong scene detail around its actual crease. The butterfly card has
more fine texture than that photo in the sampled interior. Paper
continuity and the physical medium need separate review decisions.

Multipanel revision 10 checks the weak exterior evidence of two
complementary active proposals. At opposite ends of the cut it samples
native collars across the proposed outer edge. Eight adjacent sections
must agree with the measured transition out of local backing. An
aligned strip of backing through the interior or conflicting third
contour vetoes the hint. Only the two supported spans are drawn. Both
proposal masks, the automatic count and the crop geometry stay intact.

The hint appears in all eight tested quarter-turn and JPEG75 views
of the native card with its source proposals. The intentionally
split folded-car photo abstains in all eight corresponding checks.
Rearrangements of two separate photographs from the same page with
1, 4 or 12 native pixels between them abstain in all six lossless and
JPEG75 checks. An integrated backing slit in the same card still
passes at 0.25 and 0.5 pixels in 8/8 checks and at one pixel in 7/8.
The two- and four-pixel slits veto the hint in all eight checks at
each width. As a medium control, photographic reproductions of a
printed page and a lithograph can contain the same line art and
lettering as direct ink on paper. [Talbot plates IX and XI](https://www.gutenberg.org/ebooks/33447)

Both clients open the measured spans and highlight only the two
linked masks. Opening, cancelling or switching to individual geometry
leaves the count at four. Once the reviewer confirms both belong to
the same non-photo object, one page-revision-checked event records
both rejections. In separate desktop and mobile replays through
ordinary multipart upload and processing, this reduced the active
count from four to two. Both photos retained byte-identical review
JPEGs and their existing per-mode geometry and export holds. The
untrimmed previews returned HTTP 200. Both named exports and the
legacy trimmed alias returned HTTP 404 before and after the decision.
The dismissed rows remain available for geometry review.

The measured-spans stage took 8.0 ms median and 10.5 ms p95 over
40 warm calls on one OpenCV thread. The frozen detector, seeded
geometry and Golden replays retained 18/18 instance matches,
11/11 whole-print and 12/12 motif passes, and 29/29 real development
instance matches respectively. Detection, rendered-crop and mode
availability summaries were identical to the previous replays. The
configuration records the new multipanel revision.

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

The left print on Golden page 33 has a larger overrun. Its top-left
preview corner is 129.62 native pixels from the source line and includes
album handwriting. A scratch fit based on terminating adjacent paper
edges moves only the left side. Against the source-only development
mask, whole-preview IoU rises from 0.9107 to 0.9954. Foreign contributions
beyond one pixel fall from 72,458 to 1,564 and omitted centres remain
three. The source fit repeats on the original, JPEG70, blurred and
mirrored variants. A nearby, similarly pale cover can also pass the
source test, so the application keeps its existing geometry.

#### Compare an overextended corner with the measured edge

The left print of Golden page 33 now has a separate full-print review
PNG. The source search requires one unsupported side and three strong
sides. It looks for the two adjacent colour transitions to start near
the proposed endpoints, one far from the current corner. A connected
dark-to-light edge in native pixels has to agree with both local backing
samples. Continued edge signal past the endpoints, a strong step at the
old edge or different material in the wedge vetoes the comparison. The
overlay draws only supported observations. A straight four-line fit
with a half-native-pixel allowance at the measured side supplies the
separate rectification.

The comparison retains previously covered protected regions. It tolerates
only a matching, already-disputed automatic face region from the original
review warning. In this image the existing detector flags a handwritten
album label as a face. Both the warning and original crop stay visible.
Giving the algorithm a separate trusted subject box in the wedge vetoes
the alternate. A new overlapping print on the same page invalidates the
alternate link and the measured overlay.

On a standard multipart upload, both counted prints, masks, mode
statuses, protection reports and original review JPEGs are identical
to the disabled comparison arm. The alternate PNG is 748 × 1021 and
available through the inline private review path. Both named variants
and the legacy trimmed alias return HTTP 404. Geometry approval returns
HTTP 409. Desktop and Mobile show the two previews together, with a
possible-content-loss warning and estimated aspect. The Gallery keeps
the current JPEG. The alternate adds no editable corner preset.
At `e0a9dbf`, manual saves of the old corners and of the exact
comparison corners both return `422 protected_region_cut` for this print.

Against the independent source-line development mask, the 879 × 1033
current preview has footprint IoU 0.9107, 8.14% foreign samples and
72,662 foreign contributions beyond one native pixel. The comparison
has IoU 0.9951, 0.39% foreign samples and 2,154 beyond one pixel. Three
omitted source centres exceed one pixel in each view. Maximum corner
deviation drops from 129.62 to 1.88 native pixels. The fit and protection
check took 8.95 ms warm median and 10.96 ms at p95 over 20 CPU calls.
Rendering took 11.76 ms median and 15.99 ms at p95 on the same thread.

An independent 144-view blind set and a fixed polarity reversal test
the same 36 layouts. Seven of the reversed views, from two fully
visible dark print layouts, get a comparison with zero lost owned
centres. The original views and the cover, sleeve, partial-edge and
printed continuation controls yield no alternative. A separate follow-up
extends the real paper into the wedge with a low-contrast marked strip.
The plausible alternative still appears and can omit roughly 8,200–9,000
owned rectified sample centres containing the mark. The current preview
remains available for this reason.

#### Review one album page mark during a corner edit

Golden page 33 has one disputed frontal-face region at source pixels
`(263, 1231)–(320, 1288)`. The outline surrounds the album inscription
and a small part of the print's sky. The photographed face lies well
inside the print, near `(627, 1078)`. The same Haar model also finds real
faces. The reviewer can now inspect the marked source area at a larger
scale before identifying it as part of the album page.

PhoSca offers this review only when an existing held comparison has
exactly one disputed advisory page-face or page-head detection with a
unique matching region record. Manual, pinned and verified regions
stay outside that offer. The review ID belongs to the current server
correction. The submitted corners must move the centre of the marked
box from inside the old crop to outside the edited crop, at least two
native pixels from each outline. The renderer compares protected
overlap with the prior crop so a wholly omitted unselected region can
still veto the edit. It excludes only the matched detection and keeps
local crop protection. The automatic detector uses the original set.

The recorded decision binds the exact region, source SHA-256, digest
of the page's protection list and four submitted corners. An edit to
other corners requires its own protection check. Reprocessing checks
the same binding. If source evidence changes, the photo detail, album
and review queue suppress named URLs, previously issued URLs return
404, and geometry approval returns 409. An overlap invalidating the
comparison also removes the old review offer. The validation profile
binds protected-region identity revision 2, which preserves separate
advisory and trusted records when merging overlapping detections.

The standard upload and browser tests show the selected mark only on
the left print. Selecting it performs no write. The normal save with
the original corners returns 422. An explicit corrected outline saves
and preserves the upper-level aspect requirement. On Golden 33 the
no-size edit keeps both named exports held with `PRINT_ASPECT_UNCERTAIN`.
With test dimensions of 100 × 140 mm, the exact manual outline can
produce both named JPEGs. Those dimensions demonstrate the contract
and supply no physical aspect reference. A second pinned text region
omitted by the new crop still returns 422. A different print's review
ID and a stale source digest each return 409.

The source-line development check for the exact manual outline with
unknown dimensions gives maximum corner deviation 1.88 native pixels
and three omitted centres beyond one native pixel, the same strict
cutoff as the original. Foreign contributions beyond one pixel fall
from 72,662 to 2,464. On a common 750 × 1050 comparison canvas, the p95
projected error of five source landmarks drops from 68.06 to 1.69 px.
Independent motif and physical dimensions are still pending.

#### Protect visible source content in ordinary edits

On Golden page 10 the existing protection pass identifies an actual face
at native pixels `(1833, 580)–(1961, 708)` on the upper-right print.
At `f2e5ca2`, moving the corners above the entire box bypassed the
edge-crossing test and produced both named JPEGs and the legacy trimmed
JPEG. A synthetic control with pinned lettering reproduced the same
failure for the independent outer and motif edits. In a second control
a concavity and a long protected text strip exposed the ownership
problem when all four corners of the strip lie outside the old crop.

For an ordinary corner edit, the server intersects each protected
source box with the previous rendered whole-print outline, or the
stored outline if no render supplies one, and any stored visible
polygon. It compares that area with the part retained
by the candidate, in square native pixels. This keeps a supported
concavity and distinguishes adjacent material outside the old
visibility. The explicit motif compares against the whole-print
outline. With no recorded visible polygon, a partly intersecting
advisory page-face box continues to use the existing boundary check.
The individually reviewed album mark follows its source-bound decision.
The difference threshold is one square native pixel for numeric
tolerance. The geometry check takes a median 0.038 ms and p95 0.099 ms
over 5,000 calls on a 640 × 480 development crop with two regions and
a concave visibility polygon. This measures the new check alone.

The product rejects a destructive edit with 422, the affected mode
and normalized source bounds. Desktop and mobile show those bounds
on the source. Mobile confirmation is disabled while that rejected
draft needs a valid save. In a Golden 10 multipart replay, the rejected edit
preserved both prior geometry records and the exact existing JPEGs.
A valid correction can be saved again. The replay used test dimensions
to exercise both exports. It establishes no independently measured
aspect ratio. Reprocessing an older explicitly narrowed motif holds
that mode when it drops known content. The whole-print JPEG can still
be downloaded. The locked detection, seeded geometry and Golden
development benchmarks kept their previous detection and crop results.

A legacy overextended crop can contain a false face marker with no
separate visible polygon. Such a repair can still require review.

#### Check ordinary corner edits at neighboring prints

The previous ordinary manual path allowed the edited rectangle to
double as its own visibility mask. The neighbor check therefore
ignored most pixels shared between that rectangle and a neighbor.
An independently raster-labelled control exported 8,569 foreign
source centers in the edited crop. The same geometry was held when
the neighbor was a manually added print. An attempted correction
that moved the edge inward instead dropped 7,599 exposed source
centers of the rear print and still exported both modes.

The ordinary manual path now checks the complete rectangle against
active neighboring proposals. It uses the detector's visible
polygon for an automatic neighbor and current corners for a
manually edited neighbor. A conflicting save or explicit reprocess
also reopens a neighbor whose existing rendered rectangle shares
material area. The inherited overlap check ignores a thin boundary
band and applies an area threshold. Both edited modes stay in
review, the source-corrected preview remains available, and the
original shows the competing outline. Ambiguous contained proposals
receive the same review until the count or geometry is corrected.

When an automatic render already reported possible occlusion, the
first manual correction records that evidence with the prior warp.
An inward edit clears the overlap check only after a separate check
compares that previous warp and its visible mask against the new
rectangle and current neighbors. The native raster subtracts the
new rectangle, neighboring footprints and a one-pixel outline
band. Material remaining areas appear as closed outlines. Both
named downloads stay held. Reprocessing uses the same original
prediction snapshot. If a later manual edit introduces the conflict
after a ready crop, PhoSca keeps the last ready whole-print outline
for either affected print. That outline also remains an active
neighbor claim until the conflict is resolved. The first control's
outline marked 6,935 of 7,599 missing source centers. The later
manual-conflict control marked 24,026 of 24,319 missing centers.
Both outlines excluded all labelled foreign centers. The native
subtraction alone took 2.12 ms
median and 6.56 ms p95 in 300 calls on a 2560 × 1920 development
frame containing a roughly 930 × 930 work area.

Golden page 05 has two lower prints with a measured median gap of
17.49 source pixels. A 31 px overrun produced 9,920 px² intersection
with the neighbor. The normal registered-model multipart upload,
then desktop and mobile review, preserved four active detections
and held both named exports and the legacy trimmed alias. Old
versioned crop links returned 404, while the original and the
source-corrected preview remained available. The exact safe
correction restored the previous JPEGs byte for byte. Both
separate prints could then be saved and reprocessed independently
with both named downloads available. The test used sample
dimensions solely to exercise the exports.

The independently measured correction on Golden 10 retains the
separate prints across an 8.01 px median album gutter. A 4–5 px
overrun in that case can meet an inflated automatic neighbor claim
before it reaches the other physical print. The control already
includes album backing and is routed to review. Another synthetic
case with an inflated automatic neighbor holds a geometrically
correct crop. A prior visibility overrun can also flag pure album
backing as possible loss. The original pixels remain available
to judge both cases. Golden
11 has no confirmed separating strip across 887 sampled source
rows. Automatic decisions about contour ownership require more
source evidence there. Frozen detection, seeded geometry and
Golden development metrics matched `6ff54ee`.

#### Inspect possible paper loss in the original

The browser shows the marked loss regions in the original at their
native coordinates, with up to 24 source pixels of surrounding
context on each side. The detail is opened on demand in the
whole-print editor. It reads the ordinary original page image.
Its source size uses the same oriented coordinates as the warp.
Opening it leaves the chosen corners, the review task and downloads
as they were. Existing held rows get the source size on reprocessing.

For a disputed first manual correction based on an automatic
visibility mask, the source comparison tests whether the marked
pixels all lie on one side of the new outline. The deepest eligible
strip is 28 native pixels. It compares the original Lab colors with
samples from the page perimeter. At least 99.5% must fall within
eight Lab units and all must fall within 13. The source fit must
support all four sides and both corners around the disputed edge.
A close color match adds a hint to
look at the surrounding shadows and paper edges. The prior
manual-ready reference still uses its stored geometry. Both crop
modes and the legacy trimmed download remain held while loss is
unresolved. A 480 × 320 JPEG replay with 2,140 disputed pixels
took 9.44 ms median and 24.29 ms p95 over 100 calls for the
native comparison and outline generation.

Two source-only waivers were tried and rejected. A paired gutter
test accepted 25 of 70 correctly separated variants and rejected
all 72 ordinary physical overlaps. A continuous end-cap check
rejected all 18 previously accepted bridge variants. It still
accepted 42 of 266 counterexamples containing printed dividers or
camouflaged overlaps. Measuring a clear gutter over much of a page
also missed narrow occlusions in specific disputed spans. Golden 05
has paired evidence on 555 of 555 native rows, while Golden 10 has
around 81% supported rows with the model outlines as search seeds.
Golden 11 has no supported continuous gutter.

The backing-color comparison also matched a dark part of the
labelled print footprint in six of eight simulated acquisition variants.
The hypothetical waiver would remove 1,930 visible owned source
pixels. A faint outer shadow exists beyond the mistaken mask.
The regression keeps that kind of control, plain album backing,
an obvious mark and genuinely omitted print texture in review.
The color observation can be read in the source detail while
the export and approval checks stay in place.

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

The Kreis family archive supplies seven full published JPEGs across
physical volumes `SGV_10A_00031` and `SGV_10A_00050`. The paired views
are of the same pages with and without protective sheets or loose
inserts. Six loose positives have separately catalogued recto and
verso files. The standalone files were unreachable from this workspace.
The page publications are CC BY-NC 4.0, and the rights and proof of
independence required for customer release validation remain open.
These files did not enter training, calibration or release evaluation.
The volume records, capture dimensions, source URLs and SHA-256 values
are in the [Kreis source receipt](sgv-kreis-album-intake.json).

The Australian War Memorial publishes two related public-domain
items. Its catalogue describes the loose object as a hand-coloured,
cropped version. That object lacks the same paper and complete
motif needed for a reference. [Mounted print](https://collection.awm.gov.au/C2119201)
· [Loose object](https://www.awm.gov.au/collection/C2119189)

The Finca Vigía restoration temporarily lifted and digitized
original prints in the Hemingway albums. Whole-page capture IDs,
matching loose capture IDs, downloadable originals and rights
for both are unconfirmed. [Restoration report](https://resources.culturalheritage.org/conservators-converse/2014/06/27/42nd-annual-meeting-photographic-materials-session-may-30-preserving-ernest-hemingways-photograph-albums-and-scrapbooks-at-the-finca-vigia-by-monique-fischer-an/)

The Hine/Kirtland digitizer documents two distinct captures of the
same cards, mounted and temporarily removed. A separate local
publication confirms the family album as the source of a named
portrait. Native raster downloads and commercial reuse rights
are unverified. The Hook collection supplies attributed album
images and individual versions with published CC BY 2.0 records.
Distinct files and timestamps leave physical recapture uncertain.
The Met object 1997.249 already supplied the 48-print development
page above. It remains outside independent calibration and release.
[Intake receipt](hine-hook-album-intake.json)

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

### Confirm a negative sleeve in count review

The JPEG75 contact-sheet upload shows five active proposals. Three
proposals on the left share the normalized film-grid divider. On the
right, an outer paper proposal and an overlapping fragment belong to a
separate review. These counts describe proposals before any decision.

The native image shows wood grain through a clear lip to the left of
the three masks. Several film frames match the artwork on the right
in inverted tones. With fixed proposals, an opaque perforated negative
print, a positive print and unrelated patterned paper under a simulated
clear protector all trigger the same grid warning. The divider marks
what to inspect on the original. The camera view also retains the lip
outside the proposed crops.

Desktop and mobile count review let the reviewer highlight the three
linked polygons and confirm their removal from the active count in a
single operation. Opening or cancelling this selection leaves the count
at five. The API checks the current page revision, the exact active
group and its held crop variants. It stores one review event with the
previous page and photo geometry, records a wrong-object decision for
each selected row and advances the count revision. The count stays
open with two proposals. Each excluded row retains its explicit source
preview for geometry review. The two survivors keep their crop
geometry, render records and existing export holds.

In the normal JPEG75 upload, both interfaces leave the two surviving
source preview JPEGs byte-identical. The two named exports and the
trimmed alias return HTTP 404 for both survivors before and after the
group decision. In a separate mobile follow-through, the reviewer
saves the measured outline with diagnostic dimensions of 203 × 264 mm,
rejects the right fragment explicitly and reprocesses the remaining
print. The active count becomes one. The whole print returns HTTP 200.
The motif and trimmed alias stay held with `IMAGE_MARGIN_MARKS_REVIEW`
and `IMAGE_BOUNDARY_CURVED`. The measured outline scores 0.9886 mask
IoU against the same-source development annotation. Detector scoring
still uses the five proposals before manual review.

After a separate intentional motif edit on the same mobile upload, the
Gallery delivered both JPEGs. The 609 × 792 whole print was byte-identical
before and after the edit. The 538 × 783 motif was byte-identical to the
earlier native raster check that retained 35 of 35 panels, 315 of 315
landmarks and 1,938 of 1,938 sampled heading pixels. The lowest row and
the heading remain visible. These are manually confirmed development
crops. [Earlier native comparison](#require-an-intentional-motif-outline-in-review)

A fresh local Chromium replay measured 51 ms from confirmation to
the API response and 59 ms to refreshed count controls. The measurement
covers one desktop update on a local API and database. No detector or
renderer is invoked by the group decision.

The frozen detector replay retains all 18 instances. The seeded
geometry replay retains 11/11 whole-print and 12/12 motif passes. The
Golden development replay retains 29/29 instances. The detector,
geometry, per-mode and availability summaries match the previous runs.

### Review a clear sleeve rim on the native contact page

The native 4080 × 3072 SteveKM development capture shows a 35-frame
paper contact print on the right and a negative sleeve on the left.
The mixed-legacy detector with classical fallback returns four active
proposals. The three left proposals are the sleeve carrier contour and
two model masks for groups of film strips. The carrier has boundary
score 0.9616, the children have scores 0.7016 and 0.6737, and their
footprints cover 81.48% of the carrier. The previous weak-contour path
missed this group. The real paper print has draft mask IoU 0.99472.

Multipanel revision 11 adds a separate path for a strongly supported
classical contour with exactly two substantially contained model
claims. The claims must be disjoint and account for at least 78% of
the carrier. The existing grid test verifies spacing and perforations
across the disputed divider. The new path also samples 384 stations
along each eligible carrier side perpendicular to the divider. Eight
blocks must agree on a weak outward lightness step at a common offset.
Across it, backing texture must correlate at least 0.85 and keep
55% to 93% of its outer contrast inside the possible lip. Wider
plateaus must keep the same lightness step. The pale interior and
exposed exterior need distinct chroma. A known neighboring mask in
the measurement corridor vetoes that side. An exposed backing band,
or a uniform midtone band centred on the divider, vetoes the group.
Only one eligible rim can supply the hint.
The marker is added after ownership and backing checks on the final
proposal list, so it cannot keep an otherwise deferred candidate active.

On the native sample the dashed divider joins `(353.5, 1470.5)` and
`(1953.0, 1470.5)`. The dotted rim joins `(179.08, 530.24)` and
`(161.38, 2598.89)`. At the rim the wider lightness step is 5.88
OpenCV Lab L levels, texture correlation is 0.940 and the inner-to-outer
contrast ratio is 0.862. An independently fitted trace confirms this
optical transition. The gap to the neighboring paper is 36.61 native
pixels at the median. The dashed and dotted lines locate separate
features on the original.

Bare backing, a narrow surface reflection, a supplied neighbor and
five-pixel visible backing gaps are regression controls. With fixed
native proposals the colour veto catches 3- and 6-pixel wood gaps on
the decoded and JPEG75 images. A two-pixel wood gap still passes
after JPEG75. The added lightness test catches all eight decoded and
JPEG75 treatments of a 2, 3, 6 or 11 pixel grey underlay. All eight
matching pale bands preserve the cue. A simulated clear protector
with two real paper prints and a pale seam produces the same witness.
Across 144 fixed-proposal controls, the marker appears in all six
textured sleeve treatments and abstains in all six untextured ones.
It abstains in 72 amber, grey and tinted gap treatments. It still
appears in all 60 pale, pale-with-glare and printed-divider controls.
The group remains a review decision. All four detections keep their
geometries, and the paper on the right keeps its own nested suggestion.

Desktop and mobile show the same three proposals and source lines.
Cancelling the selection preserves the current count. Confirming the
sleeve leaves one active paper print and records the three exclusions
in one review event. The detector comparison is still 1 true positive,
3 false proposals and 0 missed prints before a human decision.
The right whole-print footprint IoU stays at 0.99450 against the
development outline and its motif still needs a separate boundary.
The two named downloads and legacy cropped alias stay held.
Paired evaluation against `9e39961` preserves detection and crop metrics
in seven runs. This includes the frozen 18/18 detections, 11/11 seeded
whole-print and 12/12 seeded motif recoveries, 29/29 Golden matches, and
48/48 matches at the 12 rotated and encoded Golden 26 views. Only the
three sleeve proposals gain a review reason on the native contact page.
The added review stage measures 162 ms median and 218 ms p95 on
25 warm calls for this source, with one OpenCV thread and concurrent
verification load. Other pages skip the pixel measurements unless a
strong carrier and two disjoint model claims qualify.
Paired source, benchmark and review receipts are local under
`runs/crop-quality/native-film-sleeve/`.

### Inspect the original before excluding a negative sleeve

Interior-only feature comparisons on the pinned SteveKM development
capture matched 53 native features with opposite brightness polarity.
The 33% JPEG55, JPEG75 and PNG variants matched 39, 55 and 67 with
separately measured grid locations. All four equal-polarity runs had
zero inliers. Matching with the production quads gives 55 opposite
inliers on the native capture and zero on the reduced JPEG75 and PNG.
The reduced variants need independent grid registration. An opaque
negative-tone print under a transparent protector also preserves the
rim witness and can produce even more opposite-polarity matches. A
separate control matched 35/35 frames from opaque negative-tone paper.
On the real sleeve, 551 punched openings are much closer to the pale
insert (median ΔLab 9.0) than exposed wood (47.94), so their colour
does not establish transmission of the exterior tabletop. Matching
artwork and apertures leave the physical material ambiguous.

When a reviewer opens **Negativhülle prüfen**, Desktop and Mobile show
an enlarged source window at the midpoint of the measured divider.
If all three proposals share a measured rim, the editor shows a second
window at its midpoint. The source is the same authenticated original
used in the page view. The windows follow the native pixel-centre
convention. The width is 13% of the shorter source side with a minimum
of 48 pixels when available. The 3:2 windows shift at the page boundary
and retain the original aspect. Only the corresponding source area is
shown in each detail. Both details fit in one row when there is room.
Two short ticks locate the measured line in each detail and keep its
middle available for reading the source pixels.

The API requires `film_carrier`, all three currently linked automatic
rows, both named crop modes held for every row, and the current count
revision. An already completed count at that revision rejects an open
group dialog. Missing or mismatched confirmation returns 422. Changed
geometry or a count already confirmed returns 409. Opening or
cancelling the details leaves the active rows and revision intact.
Explicitly confirming the native case changes 4 active rows to 1.
The JPEG75 case changes 5 to 2. Both named exports and the trimmed
alias stay held for the survivors. The existing source preview
bytes, masks, transforms and deferred proposals reproduce their
previous values. The paired captures describe this development case.

### Review partial film groups on the rotated contact page

The 1014 × 1346 SteveKM development view is the 33% capture rotated
270° and encoded at JPEG quality 75. The JPEG SHA-256 is
`a3582cdda8f0c3dc0b5d17b528b6f4035d30b07ad293f36f2080a274b7e3501f`.
The mixed-legacy checkpoint SHA-256 is
`1e95a1a90ff897df7099ff80675186ec1d8145f7bb07559fe918b3dd3a6db748`.
It returns four proposals. The two narrow ONNX strip masks are invalid
solely because of their aspect ratios. Together they cover 51.2% of the
broad contrast proposal and 45.5% of the measured carrier. They omit too
much of the physical sleeve for the earlier strip-union or
strong-carrier paths.

Multipanel revision 12 tests for one convex contrast proposal around
exactly two disjoint aspect-only model claims. Each claim needs model
confidence ≥0.45 and boundary score ≥0.67. The contrast proposal needs a
boundary score from 0.40 up to 0.70 exclusive, at least 85% containment
of each claim and combined coverage from 46% up to 77% exclusive.
A four-side chroma refit must contain at least 98% of each claim. The
evidence is measured across the full refitted carrier. The children must
span a common width of at least 76% of the carrier and face one another
near a single repeating pale divider. The grid test requires at least five
regular rail pairs. At least 80% must have negative film polarity and
every rail is checked again in the original pixels. The nearby rim must
preserve backing texture with reduced contrast. Visible backing or a
consistent dark lane at the divider vetoes the cue. The narrow-lane test
uses quarter-pixel sampling across the divider and agreement along six
sections. Overlap of more than one square pixel with any other active or
deferred outline vetoes the cue as well, including the deferred raw mask
and its visible or warp outline.

This view supplies seven periodic rail pairs and a divider measuring
5.75 source pixels across. The measured divider endpoints are
approximately `(628.4, 138.1)` and `(629.2, 609.8)`. The rim endpoints
are approximately `(152.6, 50.7)` and `(839.4, 58.7)`. A separate
source measurement finds a median gap of 11.12 pixels between the
carrier and the paper print. The stage takes a median 141 ms CPU time
over nine warm calls with one OpenCV thread on this qualifying source.
Through normal segmentation of 16 development views, this JPEG is the
only new group in revision 12. Its lossless 270° twin passes the
proposal checks but fails source support at one side of the chroma
refit in that revision. The 90° and 180° JPEG75 views each contain
only one qualifying strip mask.

In 90 same-source optical controls with fixed detector records, the
marker is withheld for all 12 wood gaps of 1–5 pixels and all 18
neutral and muted-grey gaps of 2–5 pixels across three offsets. It
appears in 49 controls, including all pale and near-stock lanes and
all half-pixel treatments. An opaque negative-tone print under a clear
protector can also produce the cue. These cases require an assessment
of the physical objects in the original. The divider and rim guide the
inspection and do not decide the count.

The ordinary multipart JPEG upload, processing worker, review API and
image download path retain four active proposals before confirmation.
The count changes from four to one only when the reviewer
confirms the exact three sleeve rows. Missing confirmation, a substitute
row and a stale count are rejected. The rejected rows remain in review
history and replaying the event is idempotent. The separate paper's mask,
two crop modes and untrimmed preview stay identical through the decision
and reprocessing. Both named exports and the legacy trimmed alias stay
held. The initial detector result remains one true positive, three
false proposals and zero missed prints. All crops stay at their original
geometries, so the group itself changes review availability alone.
All eight paired evaluation reports match `31df6d1` in their
non-timing results apart from the multipanel revision. This includes
18/18 frozen detections, 11/11 seeded full-print recoveries and 12/12
seeded motif recoveries. None of these views adds an independent
release reference.
The source comparison and paired local receipts are under
`runs/crop-quality/rotated-film-group/`.

### Recheck one uncertain sleeve side on the lossless contact page

On the 1014 × 1346 lossless 270° view, the two ONNX fragments cover
53.1% of the contrast proposal. The source PNG is pinned by SHA-256
`487af95500bcb3d3c396b30a965fd80b35da0e2161dc22754c32d620d1281cc3`.
At one long side, support for the measured chroma plateaus falls
to 81.7% across the edge and 62.2% in the weakest longitudinal section.
The minimum is 86% across the edge and 65% in every section. The
other three sides pass the original test.

Revision 13 repeats the existing source-pixel line measurement at
outward offsets of one, two and three pixels, only for the sleeve and
only when one of the four original measurements is missing. All
three fits must pass. At both endpoints of the initial line, their
signed distances must differ by at most 0.60 native pixels across
the fits. Each fit must lie between zero and 2.50 pixels outward.
More than one square pixel of overlap between the original carrier
proposal and any other active or deferred outline disables the retry.
The two-pixel measurement supplies the line for that side. The
resulting quad still goes through the previous shape, corner,
containment, peer, divider and rim checks. This also rules out an
inward transition where all three scans agree but the fitted contour
would leave part of a competing claim outside.

The largest spread in this lossless view is 0.33 pixels and the
largest displacement from the initial line is 1.58 pixels. The
output lines are approximately `(628.5, 138.1)` to `(629.1, 609.6)`
for the divider and `(152.6, 50.9)` to `(839.3, 58.6)` for the rim.
Both strip masks lie entirely within the refitted carrier. The
independent paper mask has zero overlap with the carrier. The
lossless source gains the group in normal segmentation. The other
15 recorded development views retain their existing grouping.
On this lossless source the complete sleeve stage takes a median
167 ms CPU time over nine warm calls with one OpenCV thread. The
four-side-only stage declines in 77 ms under the same conditions.

All 90 JPEG75 divider controls retain their earlier results. On 60
lossless source edits with fixed proposals, all 12 wood gaps of 1–5
pixels and all 18 grey or muted-grey gaps of 2–5 pixels suppress the
cue. The cue stays visible on six one-pixel grey or muted-grey gaps,
12 matching pale gaps and six near-stock gaps of one or two pixels.
The editor still needs a person to distinguish physical pieces in
these cases.

In a separate 177-case challenge around the exterior edge, the
outward rule excludes 13 of 54 earlier advisories. The largest
accepted displacement from the original source edge falls from
2.442 to 1.123 native pixels. The outward and raw-overlap checks
block all 27 authored overlapping pending claims that passed the
earlier retry. The raw-carrier check may also withhold the group
when a corrected neighbor extends into an overlong raw proposal.
The known paper in this capture is clear of the fitted sleeve,
while its independently measured outer outline overlaps the raw
sleeve by 338.22 native square pixels. Keeping both proposals in
that circumstance leaves their physical ownership available for
manual inspection.

Multipart PNG upload with the same mixed-legacy checkpoint reports
four proposals. Explicit confirmation of the three sleeve rows
reduces the count to one. Missing confirmation, an unrelated member
and a stale revision are rejected. Replaying the same decision is
idempotent. For the independent paper print, both named crop states,
the geometry, the two checked protected regions and the untrimmed
preview JPEG match revision 12 through confirmation and reprocessing.
Both named crops and the trimmed alias remain held because the paper
rim and aspect remain uncertain. The initial detector result on this
view is still one true positive, three false proposals and zero
missed prints. All eight paired reports preserve detection metrics,
both sets of crop metrics and review availability from `54cb64f`.
The frozen replay still has 18/18 detections and the seeded geometry
replay still passes 11/11 whole prints and 12/12 motifs. Timings and
the multipanel revision are recorded separately. The source comparison
and local paired receipts are under `runs/crop-quality/lossless-film-group/`.

### Show native paper edges after the lossless sleeve review

The separate paper print in the 1014 × 1346 lossless 270° development
view is a base YOLO mask with boundary score 0.8105. Fusion selected
it from a pair containing one model and one classical proposal.
Both crop modes are held. The whole print reports `PRINT_OUTER_RIM_UNCERTAIN`
and `PRINT_ASPECT_UNCERTAIN`. Revision 13 of the print outline suggestions
allows that exact combination to run the positive contact-grid
check used by the classical contact print. The protected-region
checks, minimum 0.78 boundary score and existing conflicts still
govern eligibility.

The full-resolution check measures four chroma transitions and
verifies pale paper around the perimeter and positive perforation
contrast in the grid. Native station support is 99.72–100% on these
four edges. Its scope has mask IoU 0.99145 against the development
outline, compared with 0.78752 for the stored warp footprint.
The complete shape contains the previous footprint and preserves
both protected regions. This is a measurement for review on one
development capture.

The server sets a required paper-outline check when the source
agrees. One raw sleeve claim overlaps the paper scope by 338.22
native square pixels in the initial four-proposal view, so the
scope and its lines stay hidden. After explicit sleeve exclusion,
**Papierkanten erneut prüfen** runs the usual crop processing
on Desktop and Mobile. With the ownership conflict resolved, the
editor shows four open cyan edge segments. A new print drawn inside
that scope hides the lines even when it crosses none of the four
segments. The verification requirement persists when the lines
are hidden or the crop is processed again.

The ordinary upload and crop endpoints retain the automatic paper
mask, warp, protected-region checks and untrimmed JPEG compared
with `4bb1450`. All unconfirmed or implicit geometry writes in
the replay return HTTP 409. An explicit outer quad and a separate
confirmation are required before the API accepts an edit. The
correction revision records the confirmation and runs the existing
native manual verifier. Confirmed old, measured and oversized
quads in this replay still leave both named crops and the trimmed
alias held. The measured quad retains `PRINT_BOUNDARY_AMBIGUOUS`.
The existing JPEG75 classical guide retains its prior output.

Across 16 same-capture transformations, all 74 successful renders
retain both core crop modes and the exact arrays. The four protected
content aborts match. Only the lossless 270° paper proposal gains a
new check marker with its original neighbors present. The paired
crop call takes 606 ms median CPU time on the qualifying lossless
source after exclusion, compared with 426 ms on `4bb1450`, over
nine warm calls in each arm at one OpenCV thread. All eight
benchmark reports retain their prior detection statistics and crop
metrics, including 18/18 frozen detections and 23/23 seeded crop
recoveries. The measurement adds no release examples.

On 105 source edits, the optical check could also emit a combined
scope on 31 of 54 separated-paper controls. Some neutral backing
gaps up to 4.5 native pixels preserve the perimeter and grid tests.
In a separate paired replay of 13 variants, the source hints leave
the core crop metadata and image data identical in both modes.
The unchanged native verifier also holds explicit measured-aspect
edits on the genuine lossless source and two 4.5-pixel gap aliases.
Physical ownership remains a review decision. The raw proposals,
original and local comparison receipts are under
`runs/crop-quality/lossless-paper-guidance/`.

### Distinguish the previously reviewed sleeve edge

The current native manual verifier reports two parallel lines at the
top of the separately saved paper contact print. On the genuine
lossless 270° view the outer line's endpoints are within 0.66 native
pixels of the independent sleeve refit. The counterpart JPEG75 line
is within 1.19 pixels. The sleeve remains visible in the source after
the count decision. Narrowing the original search to five pixels
would also remove ambiguity on the 4.5-pixel neutral split controls,
a pale layer and an opaque common-mat reproduction. It does not
establish physical ownership.

Print outline revision 14 can identify the far line in this held pair.
It requires a recent matching rejection snapshot on each of three
automatic proposals. Their latest decisions must share one server
event and explicitly record a film carrier. The base contrast
proposal and its two strip masks must still share the measured
divider and rim. Refit the carrier from the original pixels and
compare those traces with the recorded lines. The paper quad and
carrier must occupy separate areas. Exactly one alternative line
must align with the refitted sleeve, and its companion must align
with the submitted paper side.

The shared backing strip is measured in 256 native cross-sections
along 85% of the paper side. The strip must be between 7 and 18
pixels wide. Both adjacent stocks must differ in chroma from the
middle of the strip, with a native half-height crossing within
1.5 pixels at each side. At least 90% of the stations and 75% of
each eighth must agree. Four consecutive failed stations veto
the hint. All current active and deferred claims are checked against
the measured strip and the sleeve. A new overlapping claim clears
the distinction when the existing pair is displayed.

The independent native profiles pass all 256 stations for both
formats. Full and partial bridges across the strip, a short 10-pixel
bridge, a darkened strip and a two-pixel misregistration fail at
least one required check. Pale tape on the sleeve side can preserve
the two transitions, so the editor identifies the side of the
reviewed sleeve. It leaves an independent question about the
physical material at that line. Neither the corridor nor the
existence of an adjacent sleeve resolves possible splits within
the paper contact print.

In both multipart upload tests, the orange line appears after
explicit grouped exclusion and saving the measured paper. Three
independent wrong-object decisions, even with the same reason,
leave both lines in their ordinary style. A late active crossing
claim withdraws the distinction before and after reprocessing.
An added claim anywhere inside the measured sleeve also withdraws it,
including a claim that touches neither line nor the backing strip.
The two native traces, saved warp, crop reasons, protection state
and untrimmed JPEG are identical to `f2731d8`. Both named modes
and the trimmed alias continue to be held. Rechecking the witness
takes median CPU time of 157.0 ms for PNG and 129.5 ms for JPEG75,
over nine warm calls per format with one OpenCV thread. All eight
paired benchmarks preserve their prior detection and two-mode crop
metrics, including the 18 frozen detections and 23 seeded geometry
recoveries. Local evidence and the comparison are under
`runs/crop-quality/film-neighbor-edge/`.

### Show an open inner edge with one clear corner

Golden page 10 is the 2560 × 1920 development capture with JPEG
SHA-256 `02466bf77cf62be45ed4d4b6d3d3c2827b783c7448daba85842dc66998755206`.
Its fourth active proposal is a dark print from the registered
mixed model. The stored lower side is flat at native y=1856 and
includes some pale album backing. Both named modes are held with
`PRINT_BOUNDARY_UNSUPPORTED` and `PRINT_ASPECT_UNCERTAIN`.
The former source guides abstain.

Print outline revision 15 can add an open span for a plain base YOLO
mask with those two held reasons, protected content intact and
the existing alternative guides absent. It starts with the stored
geometry and uses the boundary measurements from the current crop.
The weak side needs the opposite side and at least one adjacent
side to have strong support. The inward search fits lines in
six bounded windows at offsets scaled to the short side. It
requires a unique pair with a strong scene transition inside a
line whose pale band agrees with the backing. On this capture the
outer line follows the transition into the album backing.
The displayed inside span runs approximately between native
`(2229.05, 1809.96)` and `(1379.03, 1832.45)`.

The left terminal probes would use pixels claimed by another real
print. The right corner supplies its own evidence. Local fits from
6–56 and 12–76 pixels along that adjacent side independently meet
the inside line within 0.27 pixels of each other. The measured
transverse edge starts at that corner and the inside line loses
its support beyond the corner in the same direction. A separately
supported enclosing perimeter withholds the hint. The scope covers
both source lines, the search strip and the verified terminal,
with two pixels around their sample footprint. All active raw,
visible and warp claims and all pending proposal outlines are
checked against that scope. Protected regions in the possible
removed area suppress the line as well.

The multipart upload, worker and review endpoints retain all five
existing masks, warps, held states, JPEGs and protected-region
decisions compared with `f5a8d31`. In the exact case a newly added
manual claim at x2320–2368 and y1797–1850 touches the measurement
scope alone. It lies outside the saved crop and the displayed
segment. It removes the hint immediately and on reprocessing.
A pending proposal with only an outer polygon in the same area
also withholds the hint on reprocessing.

The fixed-proposal replay emits the line at all eight full-size
rotation and JPEG75 combinations and abstains at all eight
75%-size combinations. Full-size endpoints remap to within 0.08
native pixels. In a 74-proposal independent screen on 22 real
development views, only this row gets the additional guide.
The native stage takes 132 ms median CPU time over nine warm calls
with one OpenCV thread.
Across the eight paired benchmark reports, all detection, crop and
review-availability metrics match the prior revision. The new scope
and line are the sole fixture metadata changes, on this Golden10
proposal. The frozen and seeded checks retain 18 detections and
23 passing recoveries.

A diagnostic warp keeping the other three sides fixed improves
footprint IoU from 0.94688 to 0.99554 and reduces foreign
interpolation samples beyond one native pixel from 35,974 to 466
against the development outline. Visible omission at that tolerance
is unchanged in the matched diagnostic. This warp is an offline
comparison. Some warm bordered prints and reproduced scenes on
pale stock produce a similar source transition after JPEG encoding.
The product shows the single open line for inspection and leaves
the saved geometry and both named exports held. Receipts and the
source comparison are under `runs/crop-quality/golden10-inward-edge/`.

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

### Confirm a shared paper grid in count review

On the pinned 1,346 × 1,014 Stevekm camera derivative at JPEG75,
the registered 512-pixel mixed legacy model and classical fallback
produce three sleeve proposals and two overlapping proposals on the
35-frame paper print. The input SHA-256 is
`a3c019249af7ce646813de4e5ec6555699cc1785730b5016994d13206d70988e`.
The registered model SHA-256 is
`1e95a1a90ff897df7099ff80675186ec1d8145f7bb07559fe918b3dd3a6db748`,
with proposal threshold 0.25 and acceptance 0.70. The counted parent mask reaches 0.832222 native
IoU against the paper draft. Of the 473,946 draft pixel centres, it
omits 78,152. The already measured review outline reaches 0.988729
native IoU, with 4,619 draft centres omitted and 731 outside. Before
review, both modes on all five proposals require inspection. These
counts and metrics describe one development source. This source has
no independent mode or album-scale release reference.

Sending the unprepared full frame to the same model lowers the best
final mask IoU to 0.82211 at comparison grid size 1,024 and loses
the paper grouping. Sending the right-hand region to the unchanged
model with 12% padding returns no proposal. With 24% padding the best
source-registered IoU is 0.0714. In a separate shadow replay, reusing
the looser perforation witness for deferral reaches 0.98847 mask IoU
and puts the right fragment in count review. That shadow also takes
a separate claim out of the active count on three of four fixed-claim
controls that simulate separate pieces of paper. The aliases are
the 0.65- and 1.25-pixel backing slits and a flush same-stock overlay.
The 2.25-pixel slit keeps both proposals. JPEG55 retains its already
deferred fragment. The lossless derivative still has six active
proposals because its stricter pair geometry fails before the rail
check. The two input trials and the phase relaxation remain development
experiments.

Desktop and Mobile offer **Zusammengehörigkeit prüfen** on the existing
paper grid and mark the parent and its active fragments. Examine the
original for a second paper edge, gap or layer at each proposed cut.
**Als einen Abzug zählen** confirms that all linked fragments belong to
that physical print. The API checks the count revision, exact parent
and fragment IDs, matching source outline and held status of both modes
for every participant. It shares the page lock and idempotency checks
with the other count decisions. An already completed count at that
revision prevents an older open group decision from changing it. The
event captures the pre-decision page and all active photo records. It
marks the one or two confirmed fragments as having no separate print,
retaining their prior masks and
source preview for a later correction. The parent keeps its ID, mask,
mode records and preview, and the count remains open on the same page.
Restoring a fragment with new corners refreshes the page revision. If
its new footprint overlaps an already corrected parent, both parent
exports return to review.

**Aussenumriss prüfen** remains available for the retained parent.
Opening the measured outline loads an editable draft. The normal paper,
motif, overlap and sampling checks still govern the two exports. The
count decision leaves both named downloads and the legacy trimmed alias
held until their respective geometry is accepted. The production
detector, checkpoints and automatic release profile remain the ones
used in the preceding comparison.

An independently replayed native upload removes one right-hand fragment
on JPEG75 (5 → 4 active) and both on the lossless capture (6 → 4).
The retained paper row and the three sleeve rows have identical
correction records and byte-identical previews before and after the
count decision. Selecting the wrong parent, a sleeve row, stale
metadata or only half the lossless group leaves the page intact.
An identical retry returns the same event. Both named exports return
404 after the count decision and the untrimmed preview returns 200.
Local Chromium measured 83 ms to the response and 165 ms to the updated
count on Desktop. Mobile took 69 ms and 121 ms on JPEG75 and 77 ms and
131 ms on the lossless upload. These are individual local interaction
measurements. The count decision involves no new crop rendering.

After a separate sleeve decision, saving the measured paper corners
with diagnostic dimensions of 203 × 264 mm opens only the whole-print
download. Saving the reviewed motif opens the second download. The
whole-print bytes stay equal through the motif edit, and reprocessing
reproduces both exports. Each manually corrected motif retains all 35
panels and 315 sampled landmarks. The JPEG75 replay retains 1,938 of
1,938 sampled header pixels and the lossless replay retains 1,940 of
1,940. The source reconstruction reports fully owned motif samples.

The whole-print exports both render at 609 × 792. The table compares
the delivered transform to the same-source paper draft and uses the
entered diagnostic size for the aspect comparison. The draft is still
too approximate to establish a verified paper boundary or foreign-pixel
count for automatic release.

| Whole-print check against the draft | JPEG75 | Lossless |
| --- | ---: | ---: |
| Source footprint IoU | 0.98873 | 0.98963 |
| Maximum corner difference | 5.71 px | 5.71 px |
| Draft paper area omitted | 0.975% | 0.898% |
| Foreign interpolation contribution | 0.208% | 0.188% |
| Output pixels with foreign taps beyond one source pixel | 272 | 307 |
| Relative aspect difference | 0.038% | 0.038% |

The locked detector, seeded geometry and mixed Golden metrics match
the preceding comparison for both modes. The comparison reports, native
API replays and visual receipts are local in
`runs/crop-quality/shared-paper-grid-count/`.

#### Save the measured paper outline before resolving the count

The editor can save the measured outer outline before the reviewer
resolves the paper count. The server reads the original detector group
from the recorded prediction. A manual parent remains eligible when its
saved warp and visible outline match that group, allowing normalized
rounding up to `1e-6` per axis. Both modes must carry the current
overlap hold. The other proposals must be the active automatic fragments
of the same group. The count endpoint still validates the current
revision and exact IDs under the page lock. A different saved warp or
visible scope invalidates the previous group for that edit. The check
also applies to earlier saved geometry that has the same audit record.

On the pinned JPEG75 upload, Desktop and Mobile each saved the measured
outline and then confirmed the paper and sleeve groups. The count moved
from five to four to one. The saved geometry is within `3.4e-9` normalized
units of the recorded outline. The count actions preserve its reviewed
mask, warps, protection and JPEG bytes. Both named exports and the trimmed
alias return HTTP 404 in that replay. The reviewer entered no physical
dimensions. In separate API runs, moving the warp or narrowing only the
visible scope causes the former paper command to return HTTP 409 and
retains the disputed fragment. An identical successful command retry
returns the original event. The browser and API receipts are under
`runs/crop-quality/contact-paper-edit-first/`.

The lossless derivative of the same capture starts with six proposals.
After saving the measured outline its count moves from six to four to
one, confirming two paper fragments and then the sleeve. The same mask,
both mode records and the review JPEG survive the two count decisions.
The named and trimmed exports return 404. Multipart upload and processing
took 8.19 s in this single local run. See `lossless-edit-first.json`.

All eight frozen and development replays reproduce the non-timing
reports from `42c2bed`. The locked detector still matches 18/18. The
seeded crop run still passes 11/11 whole prints and 12/12 motifs. The
manual count change uses the existing holds and adds no new automatic
proposal or crop. The comparison is in `exact-benchmark-receipt.json`
in the same local directory.

The cut at approximately `(724, 500)–(1242, 500)` was also tested with
native slit and flush-stock controls. The aligned margin pulse detects
11/12 simulated 0.65-pixel slits and 12/12 1.25-pixel slits at an
exploratory threshold of two Lab units. It also triggers on 12/12
backing-coloured marks printed on one continuous sheet. Low-contrast
flush layouts overlap the real source score. The cue does not support
an automatic physical ownership decision. The 480 source-only probes
are in the `native-separation` subfolder.

#### Inspect the proposed cuts in the original

The shared-paper count dialog offers three square source windows for
each active proposed cut. They are centred on the first endpoint, the
midpoint and the second endpoint in the original raster. The window
size is 27% of the line length, rounded and clamped to 64–144 native
pixels or the available raster size. Each view is bounded by the
original image. Two short ticks sit on the normal to the cut, between
30% and 39% of the window side from its centre. The paper and any
visible mark between them stay unobscured. The window uses the same
authenticated page image as the count editor and leaves the count
and both crop modes in their current states.

On the pinned JPEG75 source the one active cut yields three windows.
The lossless derivative yields six windows for its two cuts. Desktop
and Mobile Chromium load the original image with its exact uploaded
SHA-256 in each window. The centres are within 0.71 native pixels of
their target. The nearest ticks stand at least 42 native pixels from
the cut on these inputs. Opening with the keyboard or mouse, closing,
reopening and cancelling make no write requests. The complete count
review payload stays equal across those operations. Browser receipts
and screenshots are in `runs/crop-quality/contact-paper-cut-zoom/`.

### Compare the disputed whole contact print

The normal print proposals in the 33% JPEG75 and lossless PNG both cut
the heading of the 35-frame contact print. At a native mask IoU cutoff
of 0.85, both miss the physical print. The best raw model candidates have
IoUs of 0.8223 and 0.8183 before suppression. Classical proposals reach
0.8317 and 0.8286. Fusion does not remove a higher-scoring source
outline. Changing suppression or acceptance would still leave the short
upper edge. The full-size source already reaches 0.9947 native mask IoU.
Each encoding derives from the same capture.

The crop editor now straightens the existing measured outer quad as a
separate held PNG beside the primary preview. The parent keeps its
editable four corners. The comparison assumes one physical sheet.
It marks one proposed cut on JPEG75 and two on PNG. The lines are
projected through the comparison homography, including its output
rotation, and drawn by the client over the unmarked PNG. The editor
directs the reviewer to inspect the marked places on the original page.
The aspect shown is estimated when the geometry records lack a physical
aspect reference. Count confirmation leaves the existing images and
review records available. An explicit reprocess after confirmation
updates the comparison to reflect the remaining active source claims.

The renderer accepts a fragment cut only from an automatic YOLO mask
with exactly the same measured group outline. It checks at least 99%
containment, two finite endpoints inside the outline, and five stations
within two native pixels of the current fragment contour. Duplicate cuts
and groups larger than two fragments are withheld. All other claimed
areas, including explicit reviewer quads and stored pending source
masks, remain in the overlap check. A raw classical contrast contour can
cross by a short sliver only with a corroborating original-pixel edge.
Its intersection must be no larger than the greater of 6 square pixels
and 0.035% of the paper area, and it must span no more than 10% of the
shortest side. Over that span the local line must have at least 82%
support, no competing line, and end offsets between 5 pixels outside
and 1 pixel inside. At least 85% of original-pixel stations must show
the same stock immediately inside and a darker exterior. Protected
regions already in the primary crop must stay inside the comparison.
The crop also passes the protected-region, homography and sampling checks.
An independently saved manual quad or polygon crossing the comparison
by more than one square native pixel retires the comparison asset.

The exact PNG source had one 125.74-square-pixel intersection with a
coarse sleeve proposal. Original colour profiles show an 8–16-pixel
wood gap along the same span and a continuous outer paper edge. The
three JPEG75 sleeve claims are disjoint from the measured paper outline.
Independent controls with a real short foreign strip recovered the
dark and pale overlaps as vetoes in 64 probes across two positions,
four depths, two re-encodings and the two source derivatives. A 0.65-
or 1.25-pixel backing slit inside the print and a flush separate overlay
can still produce the outer comparison. Those cases retain all active
instance claims and both modes in review. The named JPEG exports require
their normal count and geometry decisions.

| Draft measurement | JPEG75 primary | JPEG75 comparison | PNG primary | PNG comparison |
| --- | ---: | ---: | ---: | ---: |
| Rendered footprint IoU | 0.8295 | 0.9870 | 0.8246 | 0.9875 |
| Draft area cut off | 16.77% | 1.22% | 17.27% | 1.17% |
| Maximum corner difference in native pixels | 68.56 | 5.71 | 69.19 | 5.71 |
| Landmark coverage over 35 fields | 312/315 | 315/315 | 315/315 | 315/315 |
| Header ink pixel centres | 1,429/1,938 | 1,938/1,938 | 1,509/1,940 | 1,940/1,940 |
| Grid ink pixel centres | 248,755/251,331 | 251,331/251,331 | 249,392/250,457 | 250,457/250,457 |
| Foreign interpolation weight | 0.515% | 0.102% | 0.410% | 0.113% |
| Output samples with foreign taps beyond one source pixel | 1,928 | 156 | 1,489 | 220 |

The 609 × 781 and 609 × 782 comparison PNGs reproduce pixel for pixel
from the native sources and their recorded homographies. Each file
uses owned bilinear sampling. These measurements use a shared draft
paper reference. Independent aspect and motif references are absent.
The main whole-print and motif records, proposal counts (five and six),
and ordinary JPEGs match the previous revision. The comparison adds no
review tasks. The motif stays held without a supported rectified motif
on these two exact inputs. Normal upload and rendering replayed on both
Desktop and Mobile. Revealing the editor and moving the draft corners
are local actions. Without manual geometry approval, the named export
routes and the trimmed alias remain held. Reprocessing reproduces the
same comparison PNG until the applicable source claims change. With
one OpenCV thread and 31 warm measurements, median comparison-renderer
CPU time was 9.15 ms for JPEG75 and 11.44 ms for PNG. The p95 values
were 13.41 and 15.76 ms. Test reports and screenshots are local in
`runs/crop-quality/disputed-contact-outline/`.

### Preserve camera stills before page analysis

The built-in camera formerly encoded its live video resolution on both
Desktop and Mobile. The browser can negotiate a video size below the
requested ideal, and re-encoding cannot restore the missing source
detail. Both clients now use `ImageCapture.takePhoto()` on the active
track if supported. A valid JPEG or PNG is transferred as returned by
the camera. The browser and camera select the actual image dimensions.
When a still is unavailable, the existing canvas path uses the live
frame dimensions and JPEG quality 0.92 on Desktop or 0.96 on Mobile.
The app identifies a video-frame fallback. The camera views keep the
whole video frame visible for alignment. Mobile can open the device
camera picker, releasing the preview camera first. Files from the
picker or the gallery use the same upload and offline queue.
[MediaStream Image Capture](https://w3c.github.io/mediacapture-image/#dom-imagecapture-takephoto)

An ordinary Chromium upload replay used the pinned SteveKM camera
JPEG as the stand-in for a camera still. SHA-256 was
`41a5650e9d9c6e220829ec128682d94dc9005fffb311776e638cdeb7b25f8cda`.
The live test track showed the same capture at 1,346 × 1,014. On both
clients the camera path sent the 4,080 × 3,072, 2,677,318-byte JPEG
without re-encoding. Server upload verification, persisted original
bytes and the analysis raster identified the original image. The
previous code persisted a JPEG at 1,346 × 1,014 despite the available
still API. Desktop then counted five proposals and Mobile six.
With the original bytes both clients count four, including the actual
contact print and three sleeve proposals. The mixed detector and renderer
replay used the existing development outline for comparison. A paper
detection needs mask IoU at or above 0.85 in this replay.

| Source from browser replay | Physical print matched | Mask IoU | Whole-print footprint IoU | Draft area cut off | Foreign sample weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous Mobile video, 1,346 × 1,014 | 0/1 | 0.8194 | 0.8159 | 18.41% | 0.003% |
| Previous Desktop video, 1,346 × 1,014 | 0/1 | 0.8233 | 0.8207 | 17.91% | 0.042% |
| Preserved camera still, 4,080 × 3,072 | 1/1 | 0.9947 | 0.9945 | 0.46% | 0.116% |

The maximum corner deviations from the draft are 68.50 native px for
the old Mobile video, 67.95 for the old Desktop video and 7.50 for the
preserved still. The previous previews retain 1,425 and 1,490 of 1,940
projected header ink landmarks. The still retains all 1,940 and all
315 panel landmarks. Its rendered width-to-height ratio is 0.781.
The fallback crops estimate 0.670 and 0.661. The development reference
has no independent physical aspect measurement.

All three inputs hold both crop modes and the trimmed download alias.
The proposed paper geometry yields a whole-print preview. The image-only
mode lacks a supported motif warp on these inputs. The stand-in test
simulates the camera API. Actual sensor output size depends on the device.
An offline Mobile capture in the same browser kept the still bytes in
IndexedDB and persisted the exact same JPEG after reconnect and retry.
The two browser still replays took about 32 seconds from upload to the
analysed page with the local registered model. The older reduced captures
took 6–8 seconds in that environment. These are single local measurements.
Local development receipts are under `runs/crop-quality/native-still-capture/`.

The separate native endcap experiment found no separation on the
51 authentic-source variants or on 72 flush overlays. It detects
77/150 planted 0.65-pixel gaps and 128/150 planted 2.25-pixel gaps
across the tested backgrounds and codecs. In an independent adversarial
check, backing-coloured lines printed across a single sheet trigger the
same cue in 18/18 variants. Two disconnected marks at the ends also
trigger it. The true separate strips in the clear negative sleeve can
lack the cue. Colour at the ends alone therefore cannot settle physical
ownership. Those cases still require inspection of the original.

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

### Locate native contact-print lettering while the paper is held

The native 4,080 × 3,072 SteveKM source holds the whole print with
`PRINT_BOUNDARY_AMBIGUOUS`, `POSSIBLE_CONTENT_CUTOFF` and
`PRINT_ASPECT_UNCERTAIN`. The existing protected pipeline defers the
motif seam until the paper decision. In a separate pixel measurement,
the unguided seam along the film grid would lose all 1,940 mapped
development heading samples while retaining all 315 anchors inside the
35 fields. The current untrimmed preview retains all 1,940 heading
samples, 250,457 sampled grid pixels and 315 anchors. These samples
were registered from the same capture. The scene has no verified
physical paper or motif reference.

`motif_margin_review` revision 7 measures the strip next to the four
supported physical picture transitions when the border diagnostic
finds a curved seam and the ordinary protection path holds both modes
for those three reasons. The diagnostic supplies the search area. It
requires a convex visible polygon and at least 900 pixels on the
shorter side of the existing rectification.
Both the existing rectification and its 800-pixel-long resampling
must find the mark using source-pixel corroboration. The projected
polygons must have source IoU of at least 0.80. The mark must lie at
least four native pixels or 1.2% of the short paper side from both the
current projected paper and the visible polygon, using the greater
distance. Another active proposal claiming more than one square
pixel of the mark suppresses it. A new manual claim also removes the
overlapping annotation and, if it was the last mark, its review reason.

The native replay annotates one real region around `(2511, 490)` to
`(3020, 539)`, including “ASSIGNMENT” and “NYC, Brooklyn”. It covers
351 of the 1,940 mapped heading samples, so the hint asks the reviewer
to include the full relevant lettering. A single-resolution response
at the white paper-to-wood edge was rejected. Independent controls
included another printed title, blank paper, a uniform shadow, a long
rule, dots and a title owned by a separate proposal. Only the two title
examples without a competing claim produced a mark. Each of the three
33% derivatives kept its existing annotations and review statuses.
The new annotation affects an already held case, so it adds no cases
to review. All 45 rendered images in the four-source comparison matched
byte for byte.

In the ordinary upload and review workflow, both Desktop and Mobile
display the annotation on the original in **Ganzer Abzug**. On a
412-pixel-wide phone the highlight locates the label and the overview
requires magnification to read it. The annotation persists after the
explicit 4-to-1 sleeve count decision and after two photo reprocesses.
The original, untrimmed preview, mask and whole-print homography match
the baseline at each step. An unrelated manual claim preserves the
annotation. A competing manual claim removes it in both views. The two
named exports, the legacy trimmed alias and the motif preview stay
unavailable through the review flow. The untrimmed preview remains
available. The marker does not supply a motif warp.

A separate native draft retaining the heading and the lowest film
rail was visually inspected. At this revision the guarded paper edit
returns `protected_region_cut`. Two advisory face
boxes straddle the left paper edge near `(2023, 2695)` and `(2014,
788)`. Their centres are in the exposed tabletop, and native close-ups
show wood and paper edges. Existing review can resolve a single
qualifying page mark. Review of the two boxes is covered below.
Over 15 warm calls with one OpenCV thread, the new conditional
diagnostic took a median of 517 ms and p95 of 634 ms of process CPU on
the native source. The four-source replay report and rendered surfaces
are in `runs/crop-quality/contact-motif-envelope/`. The comparison also
preserves 18/18 locked synthetic detections, 11/11 seeded whole-print
and 12/12 seeded motif crops, 29/29 Golden development detections, and
48/48 detections on the six controlled development pages. Detection and
crop metrics, review counts and gate results match the previous run
across all eight regression comparisons.
[Development source](contact-sheet-camera-development.json) ·
[Marker implementation](../../backend/app/pipeline/motif_margin.py)

### Review multiple advisory marks at a measured paper edge

On the native contact print, the centres of the two advisory Haar
boxes are 22.83 and 18.76 native pixels outside the stored paper
outline. Their box cores overlap the paper by 8.50% and 15.20% and
the padded boxes by 19.25% and 24.41%. Source close-ups show a wooden
table between the real print and the adjoining sleeve. The active
sleeve proposals initially overlap both padded boxes. The normal
4-to-1 count review retires those claims only after an explicit
sleeve exclusion. Reprocessing the remaining photo then offers
both native source views in geometry review.

The new offer only examines held paper outlines with the advisory
cutoff reason, with optional paper-boundary and aspect uncertainty.
It can expose one to four advisory page-detector regions. A trusted
or pinned region, active or deferred competing claim, or uncertain
paper edge vetoes the offer. For each box, the centre must be at
least twelve native pixels outside one side. Core overlap may be
at most 20% and padded overlap at most 28%. The local edge fit must
have at least 90% support, with both endpoints within two native
pixels of the current side. The median inner-to-outer Lab lightness
step must be at least 20. At least 90% of the samples must show a
pale, near-neutral inner band. These measurements locate a place for
human review. An independent trial produced 13 pale portrait cases
out of 24 in which a similar optical witness still passed.

The editor opens at the current measured paper corners and shows a
native zoom for each field. Each field needs its own checkbox.
The usual crop edit records the ordered set of explicitly chosen
regions. A partial set or changed offer is rejected. The server
matches the source SHA-256, region digest, count revision and
corner geometry, with a one-native-pixel corner tolerance. It repeats
the pixel and claim checks before applying the choices. Only the
selected suggestions for this photo are excepted. All other protected
regions and both output modes run through the usual geometry checks.
A different protected region digest or count revision holds the
downloads. A new claim touching even the marked padding also
withdraws the offer or returns the saved crop to review.

On the normal native upload, an unconfirmed edit returns
`protected_region_cut`. An incomplete selection returns
`album_page_mark_choices_required`. Confirming both with the stored
corners still holds both exports for `PRINT_ASPECT_UNCERTAIN`.
The separate orange lettering mark stays visible during this step.
Supplying 203 × 264 mm as diagnostic dimensions opens the manual
whole-print export and keeps the curved motif and its lettering in
review. The earlier independent motif outline was held as
`IMAGE_RECTANGLE_INCONSISTENT`. Enclosing that outline in a rectangle
in the saved paper plane and projecting it back onto the original
allowed the deliberate motif edit to export. The 1,853 × 2,410
whole-print and 1,645 × 2,386 motif downloads then stayed byte-identical
through reprocessing. The motif edit kept the whole-print file exact.

Both exports retain all 28,044 measured dark header centres, all
2,730,535 dark grid centres, all 70,839 low-rail centres and 315 anchors
in the 35 fields. The whole-print footprint IoU against the visual
paper draft is 0.9949. The largest corner difference is 7.50 native
pixels and 4,368 output samples read beyond the draft's one-pixel
buffer. The motif has zero samples outside that same paper buffer.
The JPEG luma mean absolute error against the owned replay is 0.77 for
the whole print and 0.82 for the motif. These dimensions and sample
areas are development inputs. The source still lacks an independent
physical reference. The automatic release gate remains held.

The narrow advisory check took 8.7 ms median and 24.3 ms p95 CPU
over 30 warm one-thread calls on this source. All eight existing
regression runs preserve detection, crop quality and review counts.
The four-encoding comparison also preserves all 45 rendered surfaces
byte for byte before a deliberate edit. The product receipts and
exports are in `runs/crop-quality/native-protected-paper-review/`.
[Development source](contact-sheet-camera-development.json) ·
[Review geometry](../../backend/app/pipeline/off_print_advisory.py) ·
[Upload and export regression](../../backend/tests/test_api/test_off_print_advisory_review.py)

### Resize the motif in the saved paper plane

On the native contact print, the separately measured motif draft
failed `IMAGE_RECTANGLE_INCONSISTENT` when sent directly as four source
corners. The same content envelope rendered when aligned to the saved
paper plane. The server checks opposite lengths and right angles in
that plane with a two-output-pixel tolerance. Four independently
dragged corners in the skewed source view made alignment difficult.

With a ready whole print, Desktop and Mobile use a single homography
from the saved paper corners to the unit square for a motif already
aligned to those axes. Each edge must deviate by at most 0.0001 along
its fixed coordinate in the square. The pointer destination is
projected into the square. Two rectangle edges follow it, with the
opposite source corner held at its original value. Destinations are
bounded by the paper and a move through the opposite corner keeps the
last valid position. The projected corners shown and submitted to the
existing editor API refer to the original. An existing deliberately
rotated rectification uses the free-corner controls. A first motif
starts at the paper corners. Save requires an intentional gesture or
explicit selection, subject to the current whole-print review.

The geometry regression uses the exact stored paper and previously
tested motif from the 4,080 × 3,072 source. Two opposing drags reproduce
the diagnostic motif within 0.001 original pixels under all four
cyclic corner orders and both windings. Pure geometry is only one
witness. The earlier source seam, even enclosed in its smallest
paper-plane rectangle, would still remove all 28,044 measured dark
header centres, 8,128 dark grid centres and 267 dark lower-band
centres. The editor keeps the original, the marked lettering and a
separate intentional choice in the workflow.

In the normal native workflows, two opposing touch drags on Mobile
and two mouse drags on Desktop each produce a 1,640 × 2,375 motif
after an explicit save. The measured border is
slightly farther from the outer paper edge than the previous diagnostic
rectangle. The native result contains all 28,044 dark heading centres,
2,730,535 dark grid centres, 70,839 dark lower-rail centres and 315
anchors across 35 frames. The sampler finds full support and zero
foreign contributions beyond the visual paper draft's one-pixel
buffer. The 1,853 × 2,410 whole-print JPEG and its crop metadata are
byte-identical to the state just before the motif edit. Both named
downloads and the old trimmed alias return 200. A Desktop prep run
observed a form refresh while paper dimensions were being entered
immediately after reprocessing. The successful run entered the
dimensions after the refreshed editor loaded. Both tests use the
same diagnostic paper dimensions of 203 × 264 mm as the prior replay.
The locked automatic metrics still match, including 18/18 detections,
6/11 automatic-seed whole prints and 7/12 motifs. The seeded geometry
run still passes 23/23. The source measurements and browser receipts
are under `runs/crop-quality/paper-plane-motif-editor/`.
[Source and reference limits](contact-sheet-camera-development.json) ·
[Geometry regression](../../frontend/tests/cropPreview.test.js)

### Check a second renderer seed against separate paper

On the locked detector run, five recoverable prints match the
right physical instances and fail one or both output modes. The
renderer still starts at the raw detection polygon. In a shadow
run with the registered synthetic model `c31317f`, rerunning the
renderer from the refined `visible_paper_rim` turns `white_margin`,
`camera_tilt` and the first `touching` print into two-mode passes.
Their full/motif maximum corner errors are 0.493/0.083,
0.255/0.083 and 0.631/0.094 native pixels. All three have zero
strict content loss and foreign samples in both modes. The second
`touching` print and the recoverable `overlap` neighbor stay held.

A separate flush paper strip falsifies the apparent gain. Of 80
stock controls, 40 satisfy the narrower proposed rule with one
ambiguous edge, a corroborated stock fit and a passed protection
check. All 40 become ready even though they sample between 237 and
1,167 output pixels from the additional stock beyond the one-pixel
buffer. An inboard colour-step probe finds all eight high-contrast
strips and only 22 of the 40 narrow-rule cases. It also fires on one
of four clean JPEG derivatives. The production renderer uses the raw
detection polygon for this path.
The shadow replays are under
`runs/crop-quality/paper-plane-motif-editor/renderer-seed-controls/`.

### Show a lower exposed strip next to a held print

On Rudge `12REDA.JPG`, none of the six raw right-hand candidates
at the 0.25 proposal threshold reaches the lower visible island.
The classical detector also supplies no proposal there. Holding its
two crop modes therefore correctly prevents an incomplete export,
while the existing source outline does not direct the reviewer to
the lower visible area. A separate scan fitted on the upper field
alone independently matches 17 lower motif features within three
native pixels, 13 on JPEG75 and 10 after moderate blur. The opaque
gap supplies zero matches. This paired evidence establishes the case
for development. The app's hint uses only the page photograph.

For an already held model crop with an ambiguous or unsupported
boundary, uncertain aspect and possible content cutoff, the renderer
searches outside a side with at least 90% line support
and 85% corner support. A near-black patch supports the outer extent.
A lightness step of at least 20 in the upstream textured band must
match one continuous line at at least 69% of the sampled stations.
Two lateral steps and the similarity of the gap's colour to both
external flanks support the source location. The marker is inset
within the midtone band, before the black backing. A nonunique strip
or one covered by another active proposal supplies no hint. The
existing geometry editor displays the marker and an original-pixel
detail with the surrounding gap. A competing manual claim retires
the hint immediately.

The native marker covers 25,698 of 30,160 pixel centres in the
approximate lower motif reference and 450 centres outside that
draft. The edit route receives no geometry from the marker. The
normal native evaluator still has three proposed objects and both
modes held for the right one. The source marker is the sole change
in the three crop records, and all seven rendered images match the
baseline byte for byte. Normal replays of JPEG75, 75% and 50% scale
each display one right-hand marker and keep all affected tracks in
review. Those encodings yield five proposals each. Scope remains
the source review of the strip.

On Desktop and Mobile the detail includes the visible lower field,
the gap and the bottom of the existing crop. The named whole-print
and motif downloads and the trimmed alias remain unavailable.
Reprocessing the original reproduces the marker and original JPEG.
An unrelated manual claim preserves it. A competing claim covering
3,600 native square pixels of the marker retires it in both Photo
Detail and the geometry queue, including after reprocessing.

The same-page left print, left page outline, black-only, flat-card
and unadjusted different-work controls produced no marker in the
final source check. Three separate motifs inserted with matched
lightness and preserved aperture edges each produce the marker in
both the native and JPEG75 versions. This confirms the ambiguity of
ownership from the single photograph. Median CPU time for
the narrow final check was 9.7 ms and p95 was 11.8 ms over 30 warm
one-thread calls. The regression comparisons retain 18/18 locked
synthetic detections and 23/23 crops with the seeded geometry. The
hint operates on an already held crop and adds no cases to review.
The current source, test replays and draft masks
are development evidence. The Red Album is excluded from release
and calibration. The replay files are in
`runs/crop-quality/rudge-visible-strip-review/`.
[Paired photographs](https://github.com/rudged/home/tree/9422452820380ca089db3e0baf0cfaf3dc6c7cdc) ·
[Detector and pixel checks](../../backend/app/pipeline/source_strip_review.py)

### Review a book gutter and the contained print

The Rudge 12 left false proposal is a single classical `edge_contour`
covering 35.60% of the original. Its confidence is 0.829. The model
supports a smaller print entirely within it with confidence 0.583
and overlap IoU 0.354 against the leaf. The general window detector
finds 12.07% pale preview coverage against a 21% minimum. Broader
colour limits still leave the largest pale component below its
area and rectangularity requirements. A second detection is present
on the right with a partly covered subject.

The count cue first limits classical candidates to single-source
four-sided outlines between 18% and 48% of the original area. It
requires one automatic inner model proposal with at least 0.50
confidence, 0.58 boundary score, 0.66 border contrast, 97% containment
and IoU between 0.28 and 0.66 with the large outline. A competing
active or deferred proposal with at least 8% containment vetoes the
cue. The small proposal remains an independent active row.

The source check tracks two opposite Lab-lightness transitions along
128 stations per side. Search offsets and fit tolerances scale with
the side length relative to 1,200 pixels. Both lines need at least
77% fit support. The dark centre and matching pale shoulders need
82% combined support overall and 64% in each quarter. Both ends
need coordinated pale-to-dark caps supported inside the raster, and
at least one cap needs a chroma difference of 6.5 relative to the dark
outer apron. A unique qualifying side produces two visible rails in
original coordinates. The native Rudge witness measures 91% and 98%
rail support and 128/128 joint stations. Offset-scaled reproductions
with a fixed source guide work from half-size through 4× scale.
Over 30 warm one-thread runs on the native source, the gutter check
took 20.1 ms median and 25.0 ms p95 process CPU.

Printed facsimiles and independent prints joined by a similar strip
can produce this optical pattern. All four individually registered
prints on adjacent Rudge 13 abstain, although the broad page outlines
also have gutter evidence. These cases require current instance
evidence and a human decision. Native Rudge 12 and the 75% lossless
view qualify for the count cue. Compressed, rotated or smaller views
can instead have competing or insufficient inner proposals, or an
outer guide that stops before the visible cap.

Opening the count cue on Desktop or Mobile shows the measured fold
in the original and the retained violet inner proposal. The reviewer
explicitly rejects the large row. A matching current count revision,
one active marked automatic outer row, both its crop modes held and
one active automatic inner row with the exact original polygon are
required. The API treats a completed count, stale revision, shifted
child, manual replacement or nonunique child as stale. An identical
API retry returns the same event without another decision.

The ordinary native upload still has three proposals. In both clients
the explicit review reduces the active count to two. Their stored
crop modes, masks and review JPEGs match before and after the decision.
Both named downloads and the trimmed alias still return 404, and the
right-hand lower-strip cue remains in review. Reprocessing both
survivors preserves the same modes, geometry, JPEG hashes and count
revision. All eight regression runs retain their prior metrics and
add no other leaf hints, including 18/18 detections and 23/23 seeded
crops on the frozen synthetic controls. The native case's two mask
IoUs still miss the 0.85 development threshold. This count correction
adds no independent release group. The native receipts and controls are under
`runs/crop-quality/rudge-album-leaf-count/`.
[Paired photographs](https://github.com/rudged/home/tree/9422452820380ca089db3e0baf0cfaf3dc6c7cdc) ·
[Native gutter check](../../backend/app/pipeline/album_leaf_review.py) ·
[Count action regression](../../backend/tests/test_api/test_album_leaf_count_review.py)

### Measure the curved opening on the left of Rudge page 12

The left model proposal keeps all 317,200 pixel centres in the six-point
visible-opening draft and adds 74,196 outside it. The excess is concentrated
where the proposal follows the album board. The independently scanned
`REDA16.jpg` registers to the original page with 160 inliers and a
2.44 original-pixel p95 on spatially withheld matches. Its lower edge
projects more than 150 original pixels into the backing below the opening.
The motif itself projects to rows 1,109–1,114, below the visible lap at
rows 1,065–1,067.
The registration helps locate the printed material. The projected rectangle
describes the separately scanned stock, including material hidden on the page.

A development replay fits signed lightness transitions around the model
proposal in the 2,048 × 1,536 original. Straight fits on the left, right
and bottom have 93.6%, 100% and 100% sampled support, with p95 positional
deviations of 1.6–1.7 original pixels. A single fitted parabola has only
40.6% support in the right quarter of the top edge. A locally connected
path, with a penalty on step size and endpoint distance from the side fits,
follows the visible transitions at all 128 stations. It crosses between
the upper mount seam and the scalloped material. It depends only on the
original pixels and the proposal. The draft and the registered scan are
used afterwards to inspect the result.

| Mask intersected with the model proposal | Draft IoU | Draft centres removed | Centres outside the draft |
| --- | ---: | ---: | ---: |
| Original model proposal | 0.810432 | 0 | 74,196 |
| Fitted outline | 0.948108 | 1,280 | 16,011 |
| Fitted outline with a 6 px outward guard | 0.936743 | 0 | 21,420 |
| Fitted outline with an 8 px outward guard | 0.932064 | 0 | 23,120 |
| Fitted outline with an 8 px left-side guard | 0.941403 | 0 | 19,744 |

For the mask comparison, rasterize the outlines with OpenCV subpixel shift
4 on the original grid. For the two uniform guards, keep centres within
the specified Euclidean distance of the candidate and inside the original
model mask. The uniform 6 px variant loses 6–18 draft centres after JPEG
re-encoding at qualities 55, 75 and 95 or Gaussian blur at sigma 0.6.
The 8 px experiment keeps all draft centres in these four replays and
scores IoU 0.9321–0.9328. The left-side guard shifts each candidate vertex
left by `8 × clamp(1 − (x − L(y)) / 42, 0, 1)`, where `L(y)` is the fitted
left line. It scores 0.94126–0.94211 across the original and the four
variants and retains all draft centres. It addresses the side with
50,871 of the initial 74,196 centres outside the draft.
These guards measure sensitivity to the approximate outline. The tan stock
next to the fitted boundary still needs its own verified visible reference.

On page 13, the same optical fit accepts the upper-left of four registered
physical prints both at native quality and after JPEG75. It abstains on
the other three and on a dark empty opening. The 8 px left-side guard
still loses 5,161 of 99,403 projected print centres on that upper-left
control. An altered copy of page 12 with a subtle additional outer border
also passes, demonstrating that the
fit can pick an internal graphic line when a plausible outer boundary
exists. A single-thread replay took 275–439 ms across the five page 12
inputs. The result remains a development measurement. The product keeps
the original mask, and both modes remain held in this case. The existing
count review still lets the user exclude the album leaf separately.

Source `12REDA.JPG` has SHA-256 `fc02f806a28c2284a44cfbb8679a75c448c1036293404e45589667f147240be0`.
The tested guide vertices in native pixels are
`(256,404) (256,1041) (326,1067) (829,1064) (850,880) (850,404)`.
The development reference vertices are
`(314,1058) (361,438) (446,419) (690,411) (836,414) (823,1062)`.
The probe, overlay and stage measurements are retained locally under
`/tmp/phosca-rudge-red12-aperture-independent/`.
[Pinned source and independent scan](https://github.com/rudged/home/tree/9422452820380ca089db3e0baf0cfaf3dc6c7cdc)

### Show possible album opening edges in review

The product uses the native transitions as two open source traces. It
considers an automatic ONNX proposal when both crop modes already need
review and the whole-print reasons are unsupported boundary and uncertain
aspect. It requires two adjacent sides with boundary support of at least
0.88 and the opposite two sides below 0.45. An existing edge or outline
suggestion, explicit saved quad or conflicting stock review takes precedence.
The sampler rotates by quarter turns to bring the weak sides to the top
and left, then maps the measured traces back onto the original.

The lateral search ranks short bright-to-dark profiles over 96 stations.
The selected line needs at least 86% overall support and 75% in each
quarter. Against the current mask's intersection at each station, its
inward displacement must reach 8.5% of the proposal width at p10 and
10.5% at the median. The top path needs at least 92% support over 128
stations and 82% in each quarter. Its central median bow must reach 2%
of the proposal width, without abrupt changes or a disconnected left end.
The original gives a bowed trace around x367–850 and y397–432 and a
left trace from (361,497) to (319,974). The visible-material boundary can
be traced without access to the separate scan.

Four source quarter turns and their JPEG75 encodings retain the two traces
when given the same proposal and corresponding measured boundary support.
The four independently registered print guides on Rudge 13 abstain,
including the genuinely arched upper-left print. The full mixed-detector
replay on that page adds no cue. Artificial outer-seam controls show that
similar lightness transitions can still arise on a reproduction. The
caption asks the reviewer to distinguish print stock and backing and to
use an unobstructed capture when the album covers part of the subject.

On the native multipart upload, only the left whole-print review receives
the two traces and their print-edge hint. Desktop and mobile render the
same open paths and caption. The underlying masks, warps, mode statuses
and JPEG bytes match the paired baseline. Explicit count review reduces
the active list from three to two. The left opening traces and the right
lower-strip cue survive ordinary reprocessing of both remaining photos.
Both named downloads and the trimmed alias return 404 throughout this
case. The trace took 27.6 ms median and 32.0 ms p95 over 30 warm
single-thread runs on the 2,048 × 1,536 source. All eight benchmark
pairs match their previous instance, crop and review metrics on every
fixture. That includes 18/18 frozen synthetic detections and 23/23
seeded crops. Both native Rudge 12 masks still miss the development
matching threshold. Receipts and screenshots are under
`runs/crop-quality/rudge-album-opening/`. Neither Rudge page contributes
to the release split.
[Source trace](../../backend/app/pipeline/album_opening_review.py) ·
[Source rotation and negative controls](../../backend/tests/test_pipeline/test_album_opening_source_evidence.py)

### Explain a held whole print in mobile geometry review

In revision `9b78009`, Mobile can offer **Korrekt & vollständig**
on the Golden 04 270° JPEG75 neighbor with an uncertain corner,
possible occlusion and unverified aspect. The server rejects the
event. The earlier client checks cover specific source-outline
and motif states and leave this combination enabled.

The geometry queue serializes `variant_urls.full_print` from the
same variant-path check used by the geometry event. When a
nonempty whole-print mode has an explicit null URL, Mobile holds
the action and associates it with a source correction hint. After a
successful edit, it reads the updated modes, review reasons and
`crop_download_urls` from the saved photo. The rendered whole
print appears through its named URL when available. A held state
keeps the local editor preview and the associated guidance.

The hint prioritizes complete visibility and recapture for possible
occlusion. For unverified aspect it asks for measured paper width
and height together with the actual corners. Existing measured
outline and motif hints keep their roles. Legacy rows with no
whole-print mode and rows with a downloadable whole print and only
an optional unavailable motif stay eligible for ordinary geometry
confirmation. A separate required motif review can still block it.

On the Golden 04 normal upload the held geometry task connects the
button to its recapture instruction through `aria-describedby`.
Pointer and keyboard attempts send no approval request. Saving
unchanged corners without physical dimensions returns HTTP 200 and
updates the hint to the newly measured neighbor overlap. Both named
crop downloads and the trimmed alias return HTTP 404. The preview
remains accessible, and count review retains four active proposals
and one pending proposal. In a separate normal legacy upload, a
mode-less geometry task stays enabled and keyboard confirmation
returns HTTP 201.

A separate locked synthetic upload with a measured 165.5 by
113.5 mm print tests the available whole-print path. Saving its
corners renders the named whole-print JPEG and enables geometry
confirmation with the optional motif unavailable. Adding an
overlapping manual outline to the same page causes the old JPEG
URL to return HTTP 404. Attempting to confirm from the stale queue
returns HTTP 409. On reload the same photo shows the overlap hint
and live editor preview, with the saved JPEG retired and the
confirmation disabled. The API review suite has 65 passing tests,
and the frontend build and 16 tests pass. Browser and source
receipts are local under `runs/crop-quality/mobile-geometry-approval/`.

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

### Hold the inner cut when pale stock continues into a neighbor

The frozen `touching_same_color` page is 840 × 640 before a quarter turn.
Rotating it 270° counterclockwise before saving as PNG, JPEG75 or JPEG55
can make the original left print's inward ink edge look like its paper
edge. On the source before rotation, the selected fitted edge lies at
x403.62–403.75. The annotated visible paper continues through x416. The
outward search follows a different ink boundary around x424.6, on the
neighboring print. The outward observation has support 0.842–0.867 and
one strongly supported terminal. Its other terminal is missing. The
previous 0.88 support cutoff made the check skip these observations.

The new review path retains the original 0.88 branch. A partial outward
line can also initiate the check with at least 0.84 support, incoming
line support at least 0.90, and end support at most 0.20 at one end and
at least 0.80 at the other. The native check expects pale Lab stock
(L ≥185 and a and b within 21 of neutral), with contrasting image
inside the current cut and matching stock against backing on two
other sides. At least 10 of 16 stations must continue to the same
neighbor with every sampled point within 11 Lab units of that stock.
For this partial case, sample outwards up to 10% of the
shorter side, clamped to 6–24 native pixels. A continuous strip may
approach within one native pixel of the original neighbor polygon.
This covers the case where the original neighbor mask begins at its
ink. A visible strip of album backing breaks the continuity. The
neighbor geometry used by the decision comes from the original
proposal. Reprocessing remains independent of which neighbor already
has a derived outline.

PhoSca assigns `PRINT_SHARED_RIM_UNCERTAIN` to both modes. The dashed
overlay is the central 76% of the current fitted cut on that side. It
shows where to verify the missing margin in the original. The farther
observation supplies no new warp and does not claim a paper corner.
The raw detector polygons, crop transforms and rendered buffers are
preserved. For an affected row, the inward fit can no longer be stored
as a confirmed visible paper boundary. The original proposal stays
available as evidence.

The saved cut evidence is independent of which overlay has priority. It
survives reprocessing, including when a reviewer rejected the original
neighbor candidate. Earlier records with the same review reason can
reconstruct the trace from their saved quadrilateral and per-side
evidence. The reviewer submits explicit paper corners. Each endpoint
of the disputed cut must lie at least one native pixel inside that
quadrilateral. Reusing the cut, changing only dimensions or moving an
unrelated side returns a correction request with no writes. The
validated edit retains the previous trace in its provenance, so a
later edit cannot restore the same cut.

This inward clearance measures whether the edit moved past the disputed
cut. In an adversarial manual experiment, shifting the contested side
by 1.02 native pixels clears that check while 2,508 reference paper
centres still lie beyond the one-pixel cutoff tolerance. Accurate
manual export still depends on the reviewer's paper boundary decision.
A reviewer can use a separate capture to establish that boundary.
In a positive native replay, submitting the known paper and motif
corners of this fixture made both named exports available. Their
source sampling had zero cutoff beyond one pixel and zero foreign
interpolation contributions.

The four independent API setups used archived `b7a8465` and the
candidate, with each of the two registered checkpoint digests below.
Each replays the three pinned derivatives through a multipart upload
with source protection enabled. The model input is 512, the proposal
threshold 0.25 and the acceptance threshold 0.70. Photo orientation is
disabled and OCR is stubbed during the crop check.

- Synthetic control `c31317f31356140346fe421ead102e2bce1fa06f85be409d12d5b3309ee54f60`
- Mixed legacy `1e95a1a90ff897df7099ff80675186ec1d8145f7bb07559fe918b3dd3a6db748`

On all six target uploads the internal geometry changes from ready to
review in both modes. Detector counts are equal. Both named downloads
and the trimmed alias return 404 while the untrimmed preview returns
200 with identical JPEG bytes, including after reprocessing. A direct
crop approval fails. Desktop and Mobile show the measured location of
the current cut and retain the same active count. Mobile keeps the
approval control disabled with the review reason. Native target uploads
took 0.71–1.50 seconds in the local candidate runs. The independent
comparisons retain their timing and exposure records.

The values below measure the exact same affected transforms on either
side of the change. The raw motif buffers are included in this check.
These results explain the review decision and measure preservation of
the previews.

| Source-reprojection check across six target rows | Full print | Image only |
| --- | ---: | ---: |
| Footprint IoU | 0.95009–0.95802 | 0.99187 |
| Max corner difference | 11.89–11.99 px | 0.11–0.16 px |
| Relative aspect difference | 3.76% | 0% |
| Landmark p95 | 9.92–10.03 px | 0.07–0.13 px |
| Visible source area omitted | 4.20–4.99% | 0.81% |
| Source centres omitted beyond one pixel | 2,712 | 0 |
| Pixels receiving foreign interpolation contributions | 0 | 0 |

The protected motifs in these controls happen to be intact beyond the
one-pixel tolerance. The single source cannot resolve the physical stock
ownership for either mode. An exact source-alias control reproduces a
distinct-stock pair with one continuous printed substrate and the same
pixel raster. Both possible physical interpretations retain the same hash
through all quarter turns and tested encodings. A separate upload of the
physical prints can resolve the ambiguity.

The independent final comparison covers 166 page and checkpoint
treatments with 360 rendered rows. It finds 10 additional rows routed
to review across both modes, all for the original left equal-stock
print. That includes the three original synthetic-control 270° replays
and seven mixed-checkpoint variants at 90° or 270°. Every affected
whole-print transform omits 2,712 source pixel centres beyond one
pixel. There are zero foreign interpolation contributions. All
rendered arrays and transforms match the archived code. The comparison
adds no hold to the 59 previously ready modes across 48 known-gap
treatments or the 32 previously ready modes on the original frozen
inputs. The 22 runs on 11 real development views also keep their
original states. All of those real modes were already in review.
In 120 warm timed pairs, the active source stock test took 1.20 ms
median and 1.51 ms p95, up from 0.07 ms and 0.18 ms on the original
shorter search for the same proposal coordinates.

The independent occlusion screening tested 106 additional rotated
JPEG75 page and checkpoint combinations. The 55 geometrically ready
modes were recoverable by reference and had zero foreign interpolation,
out-of-source sampling or cutoff beyond one pixel. Those include nine
independently recoverable motifs where the paper corners are covered.
The frozen detection, seeded geometry, mixed Golden and both
classical-only summaries match the baseline for recognition, masks
and both modes. The locked model still detects 18/18, with 6/11
whole-print and 7/12 motif geometry passes. The seeded modes pass
11/11 and 12/12, respectively.

The `print_boundary` profile is now revision 12. Use calibration for the
deployed profile and independent physical release references for both
modes before enabling automatic downloads. The paired API and browser
receipts, native masks and reprojection metrics are local under
`runs/crop-quality/shared-rim-rotation/`.

### Show an inward source transition below an album edge

The original `golden_page_26` source (`133a327cefccdab318574d5e9e00ef1ebaba72cb03ab12b51c97a1f75fe1b17f`)
contains four physical prints. The mixed detector
(`1e95a1a90ff897df7099ff80675186ec1d8145f7bb07559fe918b3dd3a6db748`,
512 px input, score 0.25, NMS IoU 0.70) matches all four. The upper-left
print is already in review in both modes. Its top warp edge follows a
bright album line that continues past the right side of the print. The
estimated top corners sit 37.82 and 31.03 native pixels outside the
development reference line. A strong darkening transition runs inside
that line and ends at the observed side edges.

For an automatic model proposal with a weakly supported print boundary,
PhoSca can widen the native-pixel search on a single disputed side. The
existing line must brighten at the band and continue beyond at least
one end. The candidate inward line must darken consistently along its
length and have a pale, uniform outer strip. Two independent cross-edge
fits locate the ends. At each end the darkening must appear on the
print side of the intersection and disappear on the other side. The
test samples the outside band at each native-pixel depth to reject thin
intervening lines, including tape. A competing proposal at a
supported point vetoes that side. PhoSca emits the hint only when
exactly one side qualifies and there is no existing correction hint.

The source view draws one open, 556.78 px segment from approximately
`[337.31, 174.13]` to `[893.70, 153.04]` in **Ganzer Abzug**. The
separate native Lab lightness measurement over 53 stations locates the
transition within 0.50 px median and 0.75 px p95 of the segment. Its
ends stop about 35 and 34 px, respectively, short of the development
reference corners. The hint asks the reviewer to determine whether
the transition belongs to the paper or the printed motif. The rest of
the boundary still requires inspection.

| Upper-left whole-print preview against the development annotation | Before and after |
| --- | ---: |
| Detection mask IoU | 0.92451 |
| Source footprint IoU | 0.93601 |
| Maximum corner difference | 45.88 native px |
| Visible source area omitted | 0.6614% |
| Source pixel centres omitted beyond one pixel | 2,634 |
| Foreign interpolation weight | 5.9265% |
| Output pixels with foreign contributions | 38,373 |
| Output pixels with foreign contributions beyond one source pixel | 36,481 |

These figures use a development annotation. Physical aspect ratio,
landmark correspondences and independently verified **Nur Bildmotiv**
geometry are unavailable on this source. The review cue leaves the
active count, masks, both transforms, rendered buffers and default
JPEGs identical to `68c2b66`. No new mode entered review. The normal
multipart upload and reprocessing produced the same four active
records. The named crop exports and the legacy cropped export remained
held. Desktop and mobile displayed the hint alongside the original.
On the tested mobile viewport the existing number badge crosses a
short section at the start of the dashed line.

With fixed proposal evidence the helper reproduced the transition in
24 quarter-turn, JPEG and blur controls with up to 0.75 native px of
normal displacement. A preliminary sparse-band check accepted 23 of
44 tape controls. The per-depth band check rejected all 44, as well as
the additional printed line, tinted strip and lettering controls.
Nearly identical pale material can still display a visible transition.
With fresh inference on 12 source encodings, the target matched and
stayed in review in both modes on all 12. Only the original produced
the new hint. Nine variants already had a competing hint, and two
lacked the conditions for this one. One 180° JPEG75 replay produced
three unrelated extra proposals, including two fragments against the
neighboring print. The original 29 detected Golden rows received one
new hint in total. The helper took 61.89 ms median and 69.36 ms p95
across 25 warm runs with single-threaded OpenCV on the target.

All five frozen reports preserve recognition metrics, mask overlap,
per-mode geometry and review availability. The locked detector still
matches 18/18, with 6/11 whole-print and 7/12 motif geometry passes.
The seeded geometry passes 11/11 and 12/12. The `print_outline_suggestion`
profile advances to revision 3. The paired API, source overlay, browser,
benchmarks and control receipts are local under
`runs/crop-quality/golden26-inward-edge/`.

### Review two fallback contours inside a detected print

The Golden 26 development image from the preceding case has four prints.
Reversing both image axes and encoding with OpenCV JPEG75 gives
`5f047fdc4abdf9c6d1d7fed3bee636b8a019364e5f55dab40ba35f1a725815c9`.
The same mixed ONNX checkpoint still proposes all four prints. The
original lower-left print falls to 0.726 model confidence and triggers
classical fallback. In `53d9b84` three additional singleton classical
proposals survived fusion. One crosses the frame and backing. Two
outline parts of the original upper-right photo and close through its
motif. They contain 35.58% and 45.61% of the model parent's area, with
98.49% and 99.82% containment by the parent and 0.0037 mutual IoU.
Disabling classical fallback removes all three extras on this derivative
and reduces the mean mask IoU of the four matches from 0.98225 to 0.93449.

Count review first requires a strong, active model-only outline with
exactly two eligible classical contour children. Each must occupy
25–58% of its area and be at least 97.5% contained. Another model
claim of at least 15% child coverage vetoes the group. The children must
occupy 65–95% of the parent together, with mutual IoU at most 0.035.
The parent is then fitted at native resolution and must have a
supported quadrilateral with at least 0.90 IoU to its original mask.
For each child, at least 85% of stations on each of two distinct fitted
sides must lie within five native pixels of different measured parent
sides. The other sides must lack a strong independent fit. One side
must run inside the measured parent, with a weak line fit and no
coherent narrow transition across its tested original-pixel band.

The search also tests for a gap through the parent in both directions,
independently of the two proposed interior sides. It preserves native
sampling depth across the gap. The vote requires opposed Lab contrast
at both shoulders for pale or dark material, or a coherent bilateral
color step toward a neutral centre. A straight, oblique or connected
gently curved path must have original votes at 87% of stations and at
least 76% per quarter. The curved path uses a bounded cubic fit to a
connected band and checks the original votes within two raster rows.
A reconstructed curve may bend at most 28 rows from its end-to-end
line and must fit the nominated centres with p90 residual at most 2 rows.
A detected gap or an independent supported interior edge keeps both
children active. A thin overprint with identical content can still
produce the same pixel evidence as a single sheet.

The independent control set preserved all 54 straight and oblique gaps
under lossless, JPEG75, keystone and skewed capture. The combined
color step was needed for three compressed grey rails. A straight-line
search detected 12 of 40 one- and three-pixel bowed controls. The
connected-band check preserves all 40, including centre deflections
of 2–12 source pixels. Six single-stock optical controls still show
the two contours in review. The real neighboring lower-right print
and the visible same-art overlay controls stay separate. An exact
same-raster overlay alias stays ambiguous. On the real positive case,
median warm one-thread process CPU time was 248 ms across eight runs.
The whole-parent gap check accounted for 73 ms median in isolation.

On the rotated source the two contours become separately selectable
count proposals. Their original quadrilaterals are preserved. The
turquoise lines mark 446.89 and 584.42 native pixels to inspect inside
the parent. Four correct detections and one independent frame-crossing
false proposal stay active. At IoU 0.85 precision on this derivative
increases from 4/7 to 4/5 and recall stays at 4/4. The 12 encodings of
this one development page contain 48 true matches and one active false
proposal after the change. The original source remains at four of four.

Multipart upload and reprocessing preserve the masks, perspective
matrices, shapes, rendered full previews and preview JPEG bytes of all
five retained records. Both named crop downloads and the legacy cropped
download are held. The parent loses the earlier occlusion reason for
the two removed active claims. Its content cutoff and aspect concerns
persist, and the segmentation reason records the disputed count.
The parent mask IoU is 0.95576 and **Ganzer Abzug** source-footprint IoU
is 0.99188 against the development annotation. Its maximum corner error
is 6.62 native px, omitted source area 0.0868% and foreign interpolation
weight 0.9540%. There are 152 omitted source centres and 4,529 output
pixels with foreign contributions beyond the one-pixel band. Independent
motif geometry, physical dimensions and landmark references are unavailable.

Desktop and mobile require a deliberate confirmation of an independent
paper print when adding either proposal. In an isolated browser replay,
one test addition consumed only its proposal and kept both crop exports
held. The other proposed outline intersected protected content and the
API kept its proposal open with a correction message. Current count
acknowledgement clears the open proposals, preserves all five active
photos and stores the shown evidence in the review record.
All five frozen reports agree with the baseline in both modes and
all non-timing results. The synthetic detector matches 18/18, with
6/11 whole-print and 7/12 motif geometry passes. The seeded geometry
passes 11/11 and 12/12. The original Golden model run stays at 29/29.
The separate contact-sheet camera replay retains its one correct
print and three existing extra proposals, with both modes in review.
The `fusion_ownership` profile moves to revision 3. Paired API receipts,
browser views, source masks, controls and benchmarks are local under
`runs/crop-quality/nested-fallback-contours/`.

### Trace scene detail across a nested flatbed proposal

The photoScanner repository publishes an album page with six physical
prints mounted on ribbed orange backing. The original embedded raster
is 4,960 × 7,014 RGB pixels at 600 dpi. At the pinned revision, the
author's six `truth` quadrilaterals give coarse motif positions. On the
top-left print the annotation starts 108–113 pixels inside the paper
and 12–19 pixels inside the visible motif. The author annotations
therefore support count and location checks. The paper and motif
coverage measurements below use separate native edge drafts.
[Pinned source and author's annotations](https://github.com/ndipatri/photoScanner/tree/1000a550af25257e53af921ba61b72b65fe54657/tests)

```bash
git clone --no-checkout https://github.com/ndipatri/photoScanner.git /tmp/phosca-photoscanner-source
git -C /tmp/phosca-photoscanner-source show 1000a550af25257e53af921ba61b72b65fe54657:tests/reference/scan-83.pdf > /tmp/phosca-scan-83.pdf
pdfimages -j /tmp/phosca-scan-83.pdf /tmp/phosca-scan-83-native
sha256sum /tmp/phosca-scan-83.pdf /tmp/phosca-scan-83-native-000.jpg
```

The PDF SHA-256 is
`0d1df3d0ad8e65661c6fa703b7664813da75554187a3e8976aaf82a99032f107`.
The JPEG SHA-256 is
`8fcd332f56b583759ef65f0982f9b45ca198109733d98a880c533c3fd95e6b48`.
The mixed development replay used the same registered
`1e95a1a9…` ONNX detector at proposal threshold 0.25 and confidence
threshold 0.70. The second replay resizes the decoded original to
2,480 × 3,507 with OpenCV area interpolation. These are two views of
one page. Source reuse rights and separate rectified references are
unverified. The scans and drafts stay in local development.

In the native baseline the model supplies all six observed locations.
The classical detector proposes six regions, including two containing
only part of the image scene. Fusion retains both fragments as separate
regions. The upper-right fragment omits 23.78% of its draft motif
and 35.91% of its paper. Its lower proposal edge is 494–503 pixels
inside the true motif's lower edge. The middle-right fragment omits
35.73% of its motif and 47.45% of its paper. The larger proposals
contain both draft motifs and still omit about 9% of the paper in
each. At half resolution the upper-right extra drops out, while the
retained upper-right mask omits 5.27% of the draft motif. The
middle-right fragment remains.

The count check considers a singleton contrast-region quad with one
model-supported owner. Their IoU must be 0.55–0.78, at least 96% of the
fragment must be inside the owner and every other claim must cover
less than 8%. Two adjacent sides must have supported source lines.
The other two must lack a supported or competing line in the native
search band. Their middle segments must lie inside the owner. On each
segment the check follows several separated scene edges through the
proposed cut at two or more of three rectification widths. It keeps
both measured segments as review marks, moves the fragment to the
count review and holds the containing print's crops for review.
Rejection or addition of the fragment remains a reviewer decision.

| View | Visible prints | Active before | Active after | New count offers |
| --- | ---: | ---: | ---: | ---: |
| Native | 6 | 8 | 7 | 1 |
| Half-size | 6 | 7 | 6 | 1 |

The middle-right case triggers on both views. The native upper-right
case needs a different test because its lower cut crosses a smooth
area of the scene. The change leaves the case open. It preserves the
segmentation geometry of the six larger regions. A paper overlay and
a borderless overlay with an observable intensity step keep separate
rows across both tested quarter-turn orientations. The generated
scene with both continuous cuts moves to review in the same tests.
On this page the additional count check took 84 ms median CPU time
at native size and 23 ms at half size over eight warm one-thread runs.
Multipart upload at half size preserved the larger record's mask,
warp quad and preview JPEG. Reprocessing kept both crop downloads
and the legacy crop download held. The deferred polygon matches the
removed contrast region at raster precision.
Desktop and mobile display both traces and the original fragment
outline. Opening or cancelling the correction leaves the count
unchanged. Explicit mobile count confirmation withdraws just that
fragment offer, preserves all six active rows and leaves the
containing print's two crop downloads held.

All eight paired benchmark reports agree with the previous revision
on detection, geometry, crop modes and availability. The frozen
detector recognized 18/18 prints. Its automatically found proposals
passed 6/11 whole-print and 7/12 motif geometry checks. The seeded
geometry replay passed 11/11 and 12/12. The count change advances
the `fragmented_print_detection` validation profile to revision 4.

```bash
python -m pytest -q backend/tests/test_pipeline/test_fusion_ownership.py -k nested_scene_fragment
```

The same intake supplies a white embossed album pocket with one photo
and an empty pocket above it (`scan-67`), and another page with two
unobstructed photos and a previous upstream false contour spanning a
112-pixel strip at the scanner edge (`scan-66`). The embedded JPEG
hashes are `45c4f72fb886fcf84ca914b4c02fe8c774d31035bfdf28f0b4dad65b5527a691`
and `9ae5f44a33eb5dce124e4efd5573f7e7b267881330eee1dde4c03f2182487534`.
On the former, the mixed baseline has four active proposals in the
native view and three at half size. The remaining pocket and overlap
errors are tracked in the pocket analysis below. The page, native
source traces and draft masks are recorded locally under
`runs/crop-quality/nested-scene-fragment/`.

### Review continuous embossing across an album pocket

The source for scan 67 is the 4,960 × 7,014 JPEG embedded in the
pinned PDF. Use `pdfimages -j scan-67.pdf scan-67-native` to extract
it. The source shows one sepia print and a pale upper pocket. The
upstream print quad is a coarse page-space development annotation.
It supplies no separate print capture, verified motif corners or
training independence for release. [Pinned source PDF](https://github.com/ndipatri/photoScanner/blob/1000a550af25257e53af921ba61b72b65fe54657/tests/reference/scan-67.pdf)

The upper pocket has a singleton model proposal at confidence .590
native and .586 at half size. Independent spectral measurement finds
two common embossing directions in the interior and on exposed
material beyond the pocket. At native size, the lateral crossing
median correlations are .935 and .974. The real lower print has
median correlations below .24 on all four measured sides. The
upper welded lip interrupts the pattern. Across a pale region of
the real print, the source phase breaks even though the
nearby pocket and the print are similar in lightness.

The deployed check considers weak model-only proposals with a single
well-supported model peer in the same column. It uses the original
pixels and a pale, low-chroma interior check. On a 144 × 108
area-averaged thumbnail of the interior, the central residual after
subtracting broad illumination must span at most 3.6 OpenCV
eight-bit Lab lightness units from the 5th to the 95th percentile.
The scan stayed at 2.65–2.72 across five registered source and encoding
variants. Two faint scenes over the same film measured at least
4.45 and 5.10 at the weakest clearly distinguishable tested levels.
A second check finds smaller details using overlapping 22 × 24
windows. The largest 10th-to-90th percentile span must be at most 6.
On five versions of scan 67 the local span was 3.06–3.29. A hard
control whose detail occupied roughly 1% of the pocket area reached
7.78–9.23. The rule then measures two
approximately perpendicular periods in an interior patch after
subtracting a Gaussian-smoothed copy. It then compares native
patches separated by the measured horizontal period at 23 stations
on each lateral side. Each side requires at least 15 consecutive
stations with a residual standard deviation of 3 or more and
normalised correlation of at least .70. The measured intervals
become the two traces in count review. The initial check applies
to nearly upright acquisitions and abstains when the geometry or
source signal is insufficient.

A pale genuine insert under the same transparent embossed layer can
produce the same signal. A test with visible content under the layer
stays active, while the near-blank insert is available for explicit
count review. Unpatterned pale surfaces and one-directional stripes
also stay active in the controls. A pending offer retains its
original claimed mask in neighbouring crops. A linked manual save
requires current count evidence, explicit confirmation of a paper
print and at least 75% of the entered outline inside the offer.
The actual print boundary and the two crop modes then remain
in review.

| Multipart input | True prints | Active proposals before → after | Active precision before → after | Open pocket proposals added |
| --- | ---: | ---: | ---: | ---: |
| Original JPEG | 1 | 4 → 3 | 25% → 33% | 1 |
| Half-size lossless AREA raster | 1 | 3 → 2 | 33% → 50% | 1 |

The lower print remains detected in both views. The retained rows,
both mode records and existing preview JPEGs matched the prior build
byte-for-byte where applicable. The same measured traces were found
with registered JPEG75 and JPEG55 recompressions of both raster sizes.
Desktop and mobile show the two measured traces on the original and
require explicit confirmation to add a print. Cancelling leaves the
count and original crops intact. In a deliberate test, confirming a
print adds a held row. Both export modes and the legacy trimmed route
return 404, before and after a normal reprocess. The ordinary preview
remains available. Exact-image helper checks on both registered source
sizes preserve the two traces with the localized content veto enabled.

| Existing lower print versus provisional visual draft | Original | Half-size |
| --- | ---: | ---: |
| Mask IoU | .923 | .924 |
| Ordinary preview warp IoU | .938 | .951 |
| Draft footprint omitted by the preview warp | .36% | .13% |
| Corner error in pixels of the input raster | 30–135 | 3–54 |
| Ordinary preview aspect ratio | 1.218 | 1.236 |
| Estimated bilinear weight outside the draft | 6.0% | 5.2% |

These are development estimates for the existing ordinary preview.
A wider contour around the native upper pocket, a lower carrier
contour at native size and an interior motif fragment at half size
remain open detection errors. Neither named crop mode qualifies on
this scan. Separate verified motif geometry and landmarks are still
needed for release scoring.

All eight paired regression reports match the previous revision
on detection, geometry and both crop modes. The frozen detector
recognises 18/18 physical prints. The seeded crop run passes 11/11
whole prints and 12/12 motifs. The validation and calibration
profiles include `embossed_album_pocket_review` revision 1.
The local source, benchmark and product receipts are under
`runs/crop-quality/embossed-album-pocket/`.

```bash
python -m pytest -q backend/tests/test_api/test_embossed_album_pocket.py
```

### Trace paper under an embossed carrier

The lower print in scan 67 illustrates a different issue from the
empty pocket. In the original product replay its stored crop includes
embossed foil and misses some picture-side pixels. Only the top side
has a supported standard line fit. A wider search on the original
proposes all four sides. At approximately 132 native pixels search
reach, source support was 83–98%. Two sides moved about 110–137
pixels at one end compared with the stored corners.

The product uses this cue for held model crops with an unsupported
paper boundary, at least one well-supported current side and at least
two weak current sides. It fits four source lines within 7% of the
shortest side, capped at 180 native pixels. Each proposal needs a
well-supported fit and a plausible convex intersection. On two
opposing sides, separate exterior patches have to agree on both
directions of the repeating embossing. At 49 stations on each
candidate side, pairs of narrow original-pixel bands are matched
along the edge. A station needs exterior correlation at least .70,
interior correlation at most .45 and a local Lab transition at least
10. Each fitted side needs at least 72% supported stations and at least
50% in every quarter. PhoSca displays consecutive supported intervals
on the previously weak sides. The end intervals and breaks remain
open in the editor.

At native size the lower scan 67 row gains eight open segments on
three sides. The half-size raster gains nine segments on the two
sides whose standard fit remains weak. Direct controls with a plain
carrier or with repeating material still visible on both sides of
the edge produce no traces. A separately tested pale image margin
can lie beneath the same embossing. In that control a strictly linear
hypothesis sits 20–37 pixels inside genuinely owned image content.
The measured phase break is valid as an optical transition. The
editor prompts for paper ownership and uses the saved crop geometry
and the existing exports policy.

The registered native turns and JPEG75/JPEG55 copies show eight
intervals per input. The registered half-size turns and JPEG55 copy
also show intervals. The half-size JPEG75 derivative abstains. These
are transformations of the same development capture. The direct
check took 85–265 ms of CPU time per view on one OpenCV thread.

An independent native audit estimated coverage on three sides with
a two-pixel uncertainty margin. The four-line development draft
reduces estimated omitted photo-side centres from 12,527 to 137 and
backing-side centres from 285,136 to 344. For the same three sides the
estimated output pixels affected by backing fall from 305,701 in
the existing preview to 514 in the trial owned-bilinear warp. The
upper dark seam is excluded from these measurements. Its broad and
chromatic transitions disagree by more than 2.5 pixels at 778 of 2,192
usable source stations. Another active outline covers nearly the
whole lower print. Exact paper and motif ownership and the treatment
of that overlapping proposal remain open.

Ordinary multipart uploads at both sizes recover the same proposals
and keep the open pocket offer. The lower row gets only the source
trace and its source type. The other rows, both mode records apart
from that annotation, and the ordinary preview JPEG hashes match the
prior build. The desktop and mobile editors show the same native
intervals. Service reprocessing keeps the traces and keeps both
named downloads and the legacy trimmed download held. All eight
paired benchmark reports match round for round in detection, geometry,
and both crop modes. The validation profile advances the
`print_outline_suggestion` revision to 10. Local receipts are under
`runs/crop-quality/embossed-carrier-edges/exact/`.

```bash
python -m pytest -q backend/tests/test_pipeline/test_embossed_carrier_paper.py
```

### Review an embossed carrier around a detected print

In the native scan 67 baseline a classical `edge_contour` surrounds
97.1% of the active lower model mask and has IoU .583. Its polygon
accounts for roughly 9.64 million native pixel centres and contains
the entire independently fitted visible photo draft. About 43% of
the broad footprint lies outside that draft. Phase measurements across
the broad top and right edges support continuous embossed material.
At the smaller print, the phase breaks on all four fitted sides.
The broad bottom and left do not have equivalent measurements.

The count check compares single classical outlines with one
well-supported model child. The child must account for 45–68% IoU
and at least 95% of its own mask must lie in the broad outline. The
separate original-pixel fit must lie at least 98.5% inside it. A
competing active, visible or pending proposal with at least 8% of
its footprint inside the broad outline vetoes this path. The
source bands around the broad contour exclude samples that can
reach the original or measured child footprint. At least two
sides need 40 of 97 phase-supported stations in long runs. A
separate check looks for dark or chromatic content in the exposed
ring after averaging the embossing cells.

Three source intervals support the original native count hint.
The upper interval spans about 2,020 pixels. The right intervals
span about 1,260 and 420 pixels. A nearer stock boundary and a
separate print beneath the film remain possible. In hard controls a
second pale image can retain the crossing phase and pass the ring
check. A more plainly visible separate print, and declared adjacent
claims including a pending source mask, veto the count change.
The larger proposal stays selectable and retains its raw source
mask in neighbour checks. A linked addition requires explicit
confirmation of another print. Using the known child's outline or
a subregion of that print cannot consume the larger proposal.

In the native multipart A/B comparison the active rows go from three
to two, with one additional open count proposal. The actual lower
print remains active. The wider upper proposal still needs review.
The upper pocket offer, active masks, both crop-mode records and
ordinary JPEGs match the prior build. The original plus all three
registered right-angle turns and JPEG75/JPEG55 copies show three
source intervals in the direct check. It took 273–376 ms of CPU
time on one OpenCV thread. Across eight paired benchmarks, detection,
geometry and both crop modes match the prior build. The count
validation profile advances `embossed_album_pocket_review` to 2.
Local receipts are under `runs/crop-quality/embossed-outer-carrier/exact/`.

Desktop and mobile display the count hint and let the reviewer
cancel without saving. The ordinary reprocess preserves the
existing child and its two held modes with the carrier open. The
unedited wide template crosses a protected region and the API
rejects that save. On a separate test record, a manually marked
outline in the exposed ring goes through the normal linked save
and keeps both named downloads and the legacy trimmed route held.
After the count decision the real child's old occlusion reason can
clear. Its paper-boundary review, geometry and preview stay in place.

The extra half-size proposal is a different failure. It lies
within the lower print, with .186 IoU and about 99.95% containment.
Scene texture crosses two internal sides. The remaining sides follow
the print perimeter and continue beyond the small proposal. Nearly
aligned copies of the same print can produce this relation. This
was the open half-size case in that baseline.

```bash
python -m pytest -q backend/tests/test_pipeline/test_embossed_carrier_paper.py -k carrier_count
```

### Review a small contour sharing two paper edges

The false half-size scan 67 proposal is a singleton classical
`edge_contour` at .805 confidence. The surrounding model-only
print is .880. The child occupies about 18.5% IoU and over 99%
of its footprint lies in the parent. This is distinct from the
larger contrast-region fragments. The new path requires
14–26% IoU, at least 99% containment and a single owning
model proposal. A substantial intersection with another raw,
visible or pending claim vetoes the deferral in either containment
direction.

The source check needs two adjacent weak sides with continuous
scene detail across the proposed cut, and two adjacent strong
sides. On the strong sides a signed print-to-backing transition
must continue beyond the child on the same fitted line, within
the larger detected footprint with its geometric tolerance. The
measured rail extensions are about 168 and 232 input pixels on
scan 67. On the weak cuts a signed narrow transition is compared
with the broader illumination ramp. A coherent step on both
weak sides vetoes the hint.

A separately visible print and a broken outer rail keep their
rows in the controls. The production helper rejects 42 tested
raised or stepped rim variants. One-level brightness differences
and 1–2 pixel shifts of the same motif can still give matching
tracks. Lossless and aligned-region right-angle controls retain
the cue. Three of the 16 unaligned encoded-region controls
abstain. These are registered transformations of one source.

The original child mask remains available for neighbouring crop
checks. The parent needs review while the count offer is open.
Linked editing requires a current count revision and explicit
confirmation of an independent print. The entered outline must
cover at least 55% of the offer and lie at least 75% inside it.
The added print enters a separate paper check. Confirming the
count preserves the source offer in the decision snapshot and
retires its parent count reason. The user's reprocess then
evaluates the current paper and motif with the current claims.

In the ordinary half-size upload the active list drops from two
rows to one. The earlier pocket remains open with the same ID and
evidence. The larger print's mask, mode records, warp and ordinary
preview JPEG match the prior build. Desktop and mobile both display
four trace intervals on the original. Cancelling writes no decision.
A deliberate diagnostic linked addition requires the checkbox,
and both of its named downloads and the legacy trimmed download
remain held after reprocess. On a separate test instance, accepting
the count preserves both pending records in the event snapshot.
A fresh submission using those retired IDs returns 409. The active
print keeps its distinct boundary review after the count reason
clears. Local receipts are under `runs/crop-quality/nested-edge-count/exact/`.
The original-size upload keeps both earlier offers and both active
rows. All eight paired regressions match the previous detection,
geometry and crop-mode reports. The source stage check took
approximately 45 ms on the half-size scan. The validation profile
uses `fragmented_print_detection` revision 5.

```bash
python -m pytest -q backend/tests/test_pipeline/test_fusion_ownership.py -k nested_edge
python -m pytest -q backend/tests/test_api/test_mobile_review.py -k nested_edge
```

The separate scan 66 replay showed no overlap with the 112-pixel
historical scanner-bed proposal at either size. With the registered
mixed detector and classical fallback, both views found the two
coarsely labelled exposed prints. They also kept two additional
candidates spanning stacked pockets and partially exposed prints.
On the upper print, a signed source transition with median
Lab contrast 76.46 lies about 900–1,100 native pixels inside the
current left contour. The lower beach print has competing left
transitions and shares its coarse outline with parts of other prints.
The historical scanner strip and these unresolved ownership issues
have different causes. [Pinned scan 66 PDF](https://github.com/ndipatri/photoScanner/blob/1000a550af25257e53af921ba61b72b65fe54657/tests/reference/scan-66.pdf)

### Show a measured image transition inside a held detection

This guide applies when the automatic ONNX crop has an unsupported
outer boundary and both variants already need review. An existing
edge or outline review takes precedence. The shortest side of the
working quadrilateral must be at least 420 source pixels. A side
with outer support of at least 0.45 or a competing outer edge is
excluded. A paper claim overlapping the working quad vetoes the
guide.

For each remaining side, a projective grid samples 120 stations and
256 inboard depths. It looks for a bright local field followed by a
signed drop into darker pixels. The exterior must agree with the
local field along the way to the proposed step. At least 87% of
stations must support the same robust line, with at least 70% in each
of six sections. The source fit then needs at least two unambiguous
search radii with at least 0.85 edge support. Fitted endpoints agree
within the larger of one source pixel and 0.18% of the measured
length. Further signed Lab measurements count an optical step only
where both its near and far collars support it. Their coverage must
reach 81% overall and 63% in every section. The guide draws contiguous
supported spans with at least 12 measurements.

The outer field is also tested for detail at a separate sampling
scale. A closed four-sided perimeter around the transition and its
collars can veto the guide. Its search uses a bounded image of at
most 1,100 pixels on the longest side, then checks each proposed side
and its ends in the source raster. Intersections with protected
regions withhold the corresponding guide. More than one supported
side leaves the decision open. A later overlapping paper claim
retires the guide immediately and prevents it from reappearing while
the claim remains. The editor retains its paper corners and shows
the source spans as an open guide. Both modes remain in review.

The exact 4,960 × 7,014 Scan 66 JPEG (`9ae5f44a…87534`) yielded
one guide between `(1975.64, 1263.49)` and `(1997.05, 2636.41)`,
about 1,373 source pixels along the upper photo. The desktop canvas
and mobile overlay use the same native endpoints. Across the normal
upload and reprocess path, the four active detections, mode reasons,
warp geometry and four untrimmed JPEGs matched the baseline. Both
named downloads, the cropped alias and the alternative crop preview
routes remained held. The beach photo and the partial prints lower
down have insufficient or conflicting evidence for this guide.
On this source the guide measured a 144 ms median and 160 ms p95
across 21 warm single-threaded runs. All eight benchmark reports
matched the previous milestone on detection, crop metrics and mode
availability. The frozen runs retained 18/18 detections and passed
23/23 recoverable crops with the supplied starting contours.

In development controls the guide also appeared after each quarter
turn. It abstained on the tested film and contact-print layouts,
montages, penned text, reflective stripes, and separate registered
overlapping claims. With ten authored wider reproductions on
different backing colours, four well-supported outer perimeters
vetoed the guide. The six near-colour reproductions still showed the
same visible transition, so the guide does not identify the physical
sheet. A second, independent source audit found that the first
sustained change from the bright field may be several pixels from
the sharpest optical step, particularly near the bottom glare.
These measurements establish a source location for review. Physical
paper ownership, corner geometry and aspect remain unconfirmed.

The before-and-after receipts, renders and controls are under
`runs/crop-quality/scan-66/visible-image-edge/`.

### Compare overlapping held model outlines

The comparison is calculated from current page records when the
album list, count queue or geometry queue is read. It uses active
automatic ONNX masks with both crop modes held. Manual or refined
geometry and other proposed paper outlines take precedence. The
two candidates must have admissible initial geometry. Between 68%
and 83% of the larger mask and at least 96% of the smaller mask
must overlap on the normalized 1,024-pixel comparison raster. There
must also be a single current automatic candidate with unresolved
geometry overlapping 10–23% of the larger and at least 55% of
its own area. Its IoU with the smaller mask may be at most 0.005.
Conflicting pair assignments are withheld.

The API returns the two exact original polygons and their row IDs
in `competing_photo_outline`. The desktop and mobile views clip the
fills directly to those polygons. Violet is the intersection, orange
is exclusive to the larger proposal, and cyan is exclusive to the
smaller proposal. The source remains visible through the fills.
Both rows retain their own geometry controls. The comparison is
request-local, so the next read after a correction or rejection
reflects the current page. The focused geometry queue reads the
associated sibling records as well.

The original Scan 66 pair shared 7,782,018 native pixel centres.
The larger candidate contained 2,696,094 centres outside the smaller
one. This included a 109,503-centre visible landscape strip sampled
inside a conservative 44-pixel band that was outside both other
candidate masks and their held warp quadrilaterals. The wider
candidate also overlapped 1,672,162 centres of the lower proposal.
At half resolution the same pair qualified with nearly the same
containment. The pair qualified under all four quarter turns. A
single printed reproduction or a physical stack can show the same
local source structure, leaving the physical count unresolved.

The original upload and reprocess path kept four active detections,
their source masks, both held crop modes and byte-identical untrimmed
views. Rejecting the overlapping lower candidate in a disposable
test album changed the count through the existing action and
removed the comparison. Across 31 warm single-threaded checks on
the registered page geometry, the comparison took 3.3 ms at the
median and 7.9 ms at p95. The independent replay included 64
direct cases, checks against 14 further real development views,
and a 37-case archive stress with an invalid companion forced.
The comparison matched Scan 66 and abstained on the archive stress.
The eight frozen and development benchmark reports matched the
previous milestone on detection, crop quality and mode availability.

Source measurements, upload receipts and both review views are
available under `runs/crop-quality/scan-66/competing-outlines/`.

### Review a fallback outline crossing the album edge

The same Golden 26 180° JPEG75 development raster has a third false
proposal from the contrast-region fallback. In the original orientation,
its four corners are `(378, 0)`, `(1310, 0)`, `(915, 1096)` and
`(293, 1120)` in native pixels. It covers most of the upper-left photo
and extends across pale album stock into the tiled floor. The existing
photo covers 68.37% of the proposal's native raster footprint. The
contrast proposal has confidence 0.612 and boundary evidence 0.4182.

The check considers a singleton contrast proposal below 0.55 boundary
evidence and at most 0.68 confidence. Exactly two vertices must touch
the same edge of the capture frame. One active, model-supported print must account
for 55–84% of the proposed area. At least 86% of the existing print must
also fall inside the proposal. One strongly supported inner side of
that print must align to the proposal within 16 native px along at
least 70% of the measured interval. Competing fits on the existing
print can each support that comparison. The crop geometry uses its
existing fit and checks.

The source test searches a shallow band off the frame for a lightening
transition toward pale album material. It verifies distributed support
in five regions across both proposal side crossings and beyond them.
An unsupported run of more than 32 native px in the interior vetoes
the cue. The original pixels across the exposed lateral sides must show
similar Lab color on both sides. Pale material below the transition is
checked past a 16 px clearance from the print and proposal sides.
A connected local difference exceeding Lab 5.5 across
at least 1,000 native pixel equivalents can distinguish an overprint.
The matched transition extends about 1,192 native pixels in the
source. In the rotated upload its endpoints are approximately
`[2311, 1782.42]` and `[1119, 1807.45]`, extending 119.85 and 171.54 px
beyond the sides. A separate 241-station check finds median lightness
gain of at least 19.99 L* across the five regions. Peak-gradient
offsets have p95 values between 1.78 and 3.67 native px.

A loose cross-line search accepted narrow inserted prints straddling
the album line. The maximum-gap check rejects those controls. The
first strong-color check missed an inserted near-pale sheet with an
intact album line. The lower distinct-material threshold identifies
both the lossless and re-encoded instances. A general Hough pass found
partial album lines that missed the left continuation on the exact
source. Source voting over admissible shallow slopes recovered the
distributed transition.

Very narrow separate stock above the transition can leave every
measured part of it intact. The independent floor-only controls used
15–50 px prints between the album edge and the frame. All 16 retained
the album transition, including versions with photographic and pale
matte content. A trial floor model found local terminating edges in
31 of 48 additional placements. Visible grout and tilted stock still
confounded that test, so it remains experimental. The count review
asks the reviewer to inspect that band and identify any separate paper.
The original four-corner proposal is still available to open, correct
and explicitly count if a separate print is present. The turquoise
line represents the source material transition.

At IoU 0.85 the rotated normal upload has four active matches and
zero active extras, versus four matches and one active extra in
`107448f`. The three deferred contours remain available in count
review. The four photo masks, stored transforms, rendered full
previews, JPEG hashes and both crop-mode records are equal to the
baseline. On the existing upper-left print the original-content and
aspect checks still apply, plus the segmentation review reason for
the count proposal. Fresh upload, individual reprocessing, manual
addition and count confirmation were verified. Both named crop
exports and the legacy cropped export stayed held. The count task
accepts the current set of proposal IDs and preserves the reviewed
source evidence in the decision record. Independent motif, metric
aspect and landmark references for this page are still outstanding.
For the upper-left print, the development mask IoU is 0.99628 and
the whole-print source-footprint IoU is 0.99070. The crop omits
0.9183% of the development source area, with 2,924 source pixel
centres beyond the one-pixel tolerance. Foreign interpolation weight
is 0.0416%, including 68 output pixels affected beyond that tolerance.
The largest corner difference is 5.70 native px.

Across the 12 encodings of this development page, the model and
fallback match all 48 expected instances with zero active extras.
The earlier revision has 48 true matches and one active extra. Across
the five frozen reports and the contact-sheet camera replay, all
measurement and crop-mode fields agree with the previous revision.
The synthetic detector still matches 18/18. The independent seeded
geometry passes 11/11 whole-print and 12/12 motif references. On the
exact rotated case the helper takes 54.40 ms warm median process CPU
time and 79.54 ms p95 over 20 single-thread calls.

The `fusion_ownership` profile moves to revision 4. Native evidence,
API and browser checks, benchmarks and experimental negative controls
are local under `runs/crop-quality/frame-backing-contour/`.

### Recover an observed bowed edge in a transparent review preview

On the same Golden 26 JPEG75 upload, the original upper-left print
is cluster 2. Its stored projective warp clips a wedge of content near
the upright right edge. A strict development-mask measurement counts
2,378 of 2,924 omitted source centres on that side. The reference
includes some pale pixels beyond the visible transition. Independent
local Lab comparisons of the original, JPEG75 and JPEG55 locate the
edge without assuming that the development quadrilateral is exact.
On 686 agreeing source stations, contiguous-block holdout gives
p95 residual of 3.57 px for a line and 1.20 px for a quadratic fit.
The confidently classified photo- and album-side centres admit no
single straight side with zero errors. The upper crossing is visible.
The lower crossing remains weak. A larger straight quad recovers
visible image and admits measured backing in the same test.

The optional preview runs on a held classical contour with an
unsupported boundary, no competing outline and at least two strongly
supported sides. Each eligible side is sampled at 228 stations. The
source gradient and a pale, neutral exterior strip nominate the
transition. The fit uses local Lab steps, eight distributed anchor
bands and three robust selection passes. The curve needs at least
87% supported stations, 65% in each eighth and p95 residual at most
1.35 native px. It must improve p95 over the corresponding straight
fit by at least 0.55 px, with chord bow between 2 and 8 px. At least
23% of supported observations must lie more than 1.75 px beyond the
current cut. The entire exterior strip is checked for another layer.
A known neighboring outline intersecting the search corridor vetoes
the preview. Two eligible sides leave selection to the reviewer.

The source overlay shows an open 76-point trace with a 897.21 px
chord. The partial extension uses the nearer of the measured and
fitted depths, inset by 1.25 native px and limited by adjacent
supported stations. Where coverage stops, the preview returns to the
original cut. The extended source mask and premultiplied colors are
sampled with the existing homography. The PNG canvas translates by
6 output px and grows from 637 × 945 to 644 × 945. Physical aspect
and landmark correspondences remain unverified.

| Measured against the development annotation | Stored crop | Transparent review preview |
| --- | ---: | ---: |
| Source footprint IoU | 0.99070 | 0.99203 |
| Omitted source area | 0.9183% | 0.7853% |
| Omitted source pixel centres beyond the one-pixel band | 2,924 | 2,133 |
| Output pixels with foreign contributions beyond the band | 68 | 0 |
| Foreign interpolation weight over visible output | 0.0416% | 0.0149% |

The preview adds 791 source centres and preserves all previously
sampled source centres. In the stricter independent photo-side subset
it recovers 698 of 887 formerly omitted centres on that side. The
other local photometric classifier identifies 749 of the 791 additions
as clear photo content. Neither classified subset measures a new
addition of clear album backing. Their uncertain regions remain
part of the review. A same-art overlapping print without its own
proposal contributes 69 peer centres in one optical alias control.
All 24 tests with a supplied independent neighbor mask veto the cue.
The helper emits a preview in all 28 fixed-proposal rotation,
encoding and blur treatments. These derivatives share one exposure.
Measured warm helper time is about 34 ms on one CPU thread.

In the normal multipart path only cluster 2 gains the optional PNG
and trace. The four instances, masks, stored transforms, both mode
statuses and prior JPEG bytes match `110a29a`. Reprocessing retains
the original and optional previews. Both named crop downloads and
the legacy cropped download remain held. A manually marked area
intersecting the search band invalidates the PNG and trace display.
The curve reference is recorded when the original preview is created.
It survives that invalidation and later reprocessing. Tests cover a
same-corner edit, an edit of 0.35 native px and entering a print size.
The curved-edge review persists for both modes. Desktop and mobile
offer both previews for comparison in the review.

Paired evaluation preserves detection, named-crop metrics and review
counts in all seven runs. The frozen synthetic set detects 18/18
prints and the seeded recovery control passes 11/11 full prints and
12/12 motifs. The original Golden set matches 29/29 prints. Across
12 fresh encodings of Golden 26 the evaluator matches 48 prints,
with zero extra active proposals and both modes held. The film
contact-page run retains one matched print and three extra proposals,
all held for crop review. Only the known JPEG75 case gains the
optional source trace and preview metadata.

The `print_outline_suggestion` profile moves to revision 4. The paired
native masks, PNG, browser receipts and evaluator comparisons are
local under `runs/crop-quality/bowed-edge-preview/`.

### Compare two displaced edges on Golden page 04

The original Golden page 04 (`7666f3ff…`, 2560 × 1920) has three
correctly detected prints and zero extra proposals in the mixed replay.
The left lower print (truth index 1, prediction index 2) keeps the
detector's rectangle. The automatic fit supports top and right, weakly
supports the left side at 0.1964, and reports two competing transitions
at the bottom at 0.7679. The legacy annotation reports 8,794 missed
source centres beyond one pixel. Two disjoint sets of source
measurements both confirm 6,072 of those centres. The measured loss
is printed stone and window detail along the bowed left edge. The
adjacent centre print starts only about 2–6 native px beyond the
right boundary over part of its length.

Revision 5 looks for a weak outward and an ambiguous inward side
beside two well-supported sides. Each disputed side is sampled in
half-pixel offsets at 256 native stations. A signed lightness step
with a pale, locally consistent outer collar nominates source samples.
The robust quadratic fit needs at least 87% distributed coverage and
at most 1.35 px p95 residual. The outward fit needs a 2.5–10 px bow.
The inward fit also needs a source-supported transition that ends at
a corner where the pale material continues past the print. A weaker
outer lightness drop can identify a competing paper rim. Its veto
checks the three-pixel drop and its excess above both adjacent
three-pixel slopes. A detected neighbor intersecting either search
corridor vetoes the comparison.

Four open sections can be drawn over the original. The optional PNG
uses the same homography plus an 11 × 1 output pixel translation.
On the left, it expands only to the nearer measured or fitted edge
with a 1.5 native px margin. At the bottom it retains at least 2.5
native px of the pale side of the strong scene transition. Any
inset strip touching a protected region, its existing padding, or
an extra native pixel returns locally to the existing cut. Open
intervals and the corners return to the existing cut as well.
The comparison adds 4,753 source centres, removes 4,607, and omits
zero previously sampled centres in the padded protected regions.
Both optical fits support 4,752 of the additions and find zero
removed centres in their strict content intersection.

| Measured quantity | Stored crop | Transparent review preview |
| --- | ---: | ---: |
| Source footprint IoU against the old annotation | 0.96641 | 0.97752 |
| Omitted source centres beyond one pixel, old annotation | 8,794 | 4,063 |
| Output pixels with foreign contribution beyond one pixel, old annotation | 20,284 | 14,141 |
| Source footprint IoU against optical fit on the second station split | 0.97087 | 0.98203 |
| Omitted source centres beyond one pixel, second split | 6,177 | 1,425 |
| Output pixels with foreign contribution beyond one pixel, second split | 18,002 | 12,732 |

The optical splits describe one exposure. At the bottom the pale
material reaches the neighbouring album area and a lower layer
edge continues past the print. A pale paper border with the same
appearance remains possible. A separate white-rim ScanCropper print
has two discernible edges, a strong scene-to-stock step and a weaker
stock-to-scanner step farther out. With matching pale material the
second edge can remain unobservable. A narrow color-matched layer
over the left side can also pass the optical test in a constructed
control. These limits govern how the comparison is described in both
editors. Metric dimensions, landmark accuracy and the true motif
boundary remain unverified on this page.

The comparison is served as a review PNG beside the stored JPEG.
Both named modes and the old cropped export remain held through
upload and reprocessing. Upload and reprocessing produced the same
PNG and JPEG bytes in the normal application path. An unchanged
mobile corner submission without physical dimensions received
`protected_region_cut` and left the review open. An overlapping manual claim retires the
comparison, including when it crosses only the inward measurement
band. The new cue does not record a manual paper-curve decision.
The ordinary manual geometry and protected-content checks still
apply separately.

The helper took 39 ms median and 84 ms p95 across 25 warm calls
on one OpenCV thread with the verification suite running. Eight
paired evaluations preserve detections, named crop geometry and
mode availability. These include 18/18 frozen detections, 11/11
seeded whole prints and 12/12 seeded motifs. The new comparison
appears on the original Golden 04 only. In the separate replay of
its original and three rotated JPEG encodings the detector matches
all 12 prints and keeps two existing extra proposals. The rotated
encodings enter different boundary diagnostics and get no new
comparison. On the 180° JPEG55 replay the original model seed reaches
native y = 1886 in the original orientation. The subsequent boundary
fit follows a different album edge out to y = 1906. Beyond the
separately measured album layer and a further 3 px guard are 22,581
output samples with foreign material, including 21,667 wholly foreign
samples. This case remains held. Browser captures, renderer replay
and paired metrics are under `runs/crop-quality/displaced-print-edges/`.

### Compare a continuing album layer near the capture edge

The 180° JPEG55 copy of Golden 04 (`3e3a31d7…`) keeps all three
prints. In original upright coordinates, the raw model mask for
the left lower print reaches y = 1886. The four-corner extraction
still reaches y = 1886. A 31 px native edge fit follows the wrong
album line to y = 1906. The visible photo transition at y ≈ 1842–1850
falls 36–44 px inside the initial edge. The 82 px outward retry
cannot reach it. The held full-print rendering uses the later fit.

Independent source sampling uses 288 native stations between
x = 154 and x = 1302 in the upright orientation. The first distinct
album layer continues from beneath the print onto the neighbouring
page at y = 1869–1883. Four encodings give at most 1 px disagreement
under the print. Eight measurement kernels remain within 2 px there.
Twelve separate sight marks agree within 1 px. The pale strip
between the photo transition and the continuing layer remains
an ambiguous material boundary. The measurement uses no crop labels,
detector masks or output transforms to determine the layer.

Profile revision 6 adds a comparison for a held model outline with
two strongly supported sides and a weak corner at a capture-adjacent
edge. The native profiles look for a light-to-dark scene transition
43–83 px inside the stored line and a shallower dark-to-light layer
13–44 px inside. The layer must precede the scene by 15–44 px,
with a p95 fit error at most 1.55 px and distributed source support.
The layer continues outside at least one end while the scene stops.
There it must show a matching chroma direction and a second, lighter
material step across the dark band. Local line fits on both sides of
the corner must meet within 2.5 px. A known neighbor intersecting the
proof corridor vetoes the comparison.

The optional PNG follows only supported spans and leaves a three-pixel
reserve on the outward side of the layer. It retains protected areas
with their padding, uncertain intervals and the ends of the stored
rectangle. The 768 × 1160 canvas uses the same homography plus a
one-pixel vertical translation. Both the photo colors and alpha from
the ordinary API preview replay exactly. The separate continuation
trace runs 18.5–159.2 native px outside the stored rectangle.

| Measured quantity | Stored crop | Transparent review preview |
| --- | ---: | ---: |
| Native source centres beyond the independent album-layer guard | 19,508 | 825 |
| Visible output pixels receiving material from that zone | 22,581 | 1,428 |
| Source footprint IoU against the old annotation | 0.94526 | 0.96625 |
| Strictly missed source centres against the old annotation | 1,248 | 1,248 |
| Source footprint IoU against the second native optical fit | 0.94551 | 0.96650 |

All 18,683 removed centres fall inside the independently bounded
album zone. The preview adds zero centres and removes zero additional
centres in the optical photo core. The remaining foreign samples
include the locally retained ends. The image still loses content on
another side. Absolute print dimensions, perspective and motif
correspondences remain unverified.

Fixed-proposal tests keep the cue after another JPEG75 encode and
Gaussian blur at 0.8 and 1.3 px. All six short pale-rim controls
keep their modeled paper centres. Two visible, longer pale rims,
six overlapping mount treatments and eight ordinary shadow controls
veto the comparison. A deliberately matched three-phase hard shadow
still passes in the uncompressed control. Printed reproductions of
both transitions can also pass. This comparison therefore asks the
reviewer to identify the material at the original rather than inferring
ownership from color alone.

Normal upload and reprocessing retain the three instances, both mode
records, and the stored JPEG bytes. Both named downloads and the
legacy cropped download remain held. A new source claim across just
the continuation retires the review asset. Saving unchanged mobile
corners without physical dimensions leaves both modes in review.
The cue does not persist a manual curve decision. Warm helper time
was 46 ms median and 49 ms p95 across 24 calls on one OpenCV thread.
Across eight paired evaluations the only added result is the local
comparison on the 180° JPEG55 copy. Detection, regular crops and
review counts match revision 5. The frozen detector retains 18/18
detections and the seeded control retains 11/11 whole prints and
12/12 motifs. The four Golden 04 encodings still yield 12 matches
and two existing extra proposals on the 270° JPEG75 case.
Receipts and the exact sampling comparison are local under
`runs/crop-quality/continuing-album-layer/`.

### Compare a local dark print edge

On Golden 04 rotated 270° and encoded at JPEG75, the old left
cut misses dark image material and the other edge retains a
broad pale strip. The native source audit found 304 of 304
signed transitions on that side, with p95 residual 0.47 px around
the source curve. The lower quarter has median displacement
6.38 px outside the old cut. The bottom photo-to-pale transition
is separate evidence. Its disputed pale strip can either be part
of the photographic paper or album material.

The comparison uses a model mask when both modes are held for
an ambiguous perimeter and uncertain corners. Two sides must
have support at least 0.94. The tested side must have support
between 0.58 and 0.82, at least 0.90 at one end, at most 0.20 at
the other, and an ambiguous adjoining side at the weak end.
The measured side must be at least 580 native px. A peer claim
intersecting the local band stops the cue. The native search
looks from 4 px inside to 14 px outside the cut. The fitted
transition must return near the cut at the strongly supported end.

Two narrow strips along the observed transition must have
correlated high-pass luminance and similar Lab color. A flat
shadow and an adjoining print edge can look like the same
brightness step without preserving image detail. The comparison
uses a high-pass correlation of at least 0.58 and requires median
Lab distance at most 10. A light interruption in the added
region caps that station. The local polygon follows the smaller
of the fitted and observed expansions with a 1.25 px allowance
inside the measured transition. Nearest stations must support
the same expansion. The old crop is the baseline throughout.

The exact API preview is 1123 × 782 RGBA. Its perspective is the
stored transform translated by nine output rows. It adds 1,309
source centres at the left edge, all in the one-pixel-eroded
intersection of two independently measured optical contours.
It removes zero previously enclosed centres. A byte-for-byte
pixel replay of the PNG finds the added source in 1,940 output
pixels. The unrelated pale strip on the other edge stays
visible in the comparison.

| Metric against the native JPEG75 optical fit | Stored crop | Comparison |
| --- | ---: | ---: |
| Source-footprint IoU | 0.97758 | 0.97916 |
| Omitted native centres beyond the one-pixel tolerance | 1,815 | 506 |
| Output pixels with foreign contribution beyond the tolerance | 17,277 | 17,260 |

Against the other optical fit the footprint IoU rises from
0.97740 to 0.97897 and strict omissions drop from 1,842 to 533.
Foreign output counts remain around 17,250 on both renders. The
new portion adds zero foreign source centres in either fit.
The full reference still lacks confirmed paper outside the dark
image, physical dimensions and independent motif landmarks.

Five fixed-proposal source, recompression and blur controls add
1,291–1,354 native centres. Among 28 ordinary neighboring-strip
and shadow controls, 27 abstain. One light softened shadow
allows an addition entirely inside the older optical core.
Thirty-four independently prepared distinguishable clone, shadow
and neighbor tests abstain. A pixel-identical layer and 25 of 28
small palette-shifted, deliberately registered independent prints
can still pass the optical cue. The corresponding known peer
polygons veto those proposals. The source overlay and transparent
comparison support manual ownership review.

Normal upload and reprocessing reproduce the one open trace and
comparison PNG. The four active instances and one pending count
proposal, stored JPEGs, transforms and crop-mode records agree
with the earlier version, apart from the comparison fields.
Manual movement that covers the existing protected region can
retire the comparison without recording automatic curve ownership.
Submitting the old corners with the partial protection claim
returns HTTP 422. Both named downloads and the trimmed alias stay
held. Desktop and Mobile show the original edge hint and the
comparison. `print_outline_suggestion` moves to profile revision 7.
The measured helper takes 16.31 ms median and 21.48 ms p95 CPU
time over 31 warm calls with one OpenCV thread. Across eight
paired reports, the only new output is the local comparison on
this source. Counts, retained geometry and both mode states
match the prior reports. The frozen detector stays at 18/18,
and the seeded test retains 11/11 whole-print and 12/12 motif
passes.
Receipts are local under `runs/crop-quality/local-dark-print-edge/`.

### Review a fallback contour sharing a weakly detected print edge

Golden 04 at 270° and JPEG75 originally returned three model masks
for three physical prints, plus two classical fallback proposals.
The right print's model confidence is 0.432 and its pre-fusion
boundary score is 0.6898. The additional contrast-region proposal
has confidence 0.579 and score 0.4153. Its footprint includes part
of that print, pale album material and at least 42,712 floor pixel
centres in a conservative visually checked region. The older
ownership check excludes the model below confidence 0.70. The
older backing rule requires two frame vertices instead of the one
observed capture-corner vertex.

The new review cue considers a single contrast proposal with score
below 0.55, confidence at most 0.68 and exactly one corner on the
capture frame. It requires one overlapping YOLO segmentation mask
with confidence in `[0.40, 0.70)`, boundary score at least 0.65 and
border contrast at least 0.70. The overlapping areas must cover
70–90% of the proposal and 55–82% of the model outline. On this
source the values are 77.23% and 70.78%. These are nomination
checks for the pixel comparison.

The original must support exactly one side of the proposed
quadrilateral at 0.90 or higher. That side must coincide within
five native pixels of an independently fitted model side at at
least 85% of the stations. The model must have at least three
unambiguous sides at support 0.88 or higher. A second side of the
proposal must cross the model interior with locally changing image
structure and insufficient evidence of an independent paper seam.
A third side must have clear access to the pale material outside
the model and retain the same material across the line. The actual
one-shared-side fit has support 1.00. The remaining three proposal
side supports are 0.1696, 0.1250 and 0.1607.

The two unsupported spans are checked for a faint chromatic band
across both shoulders. The check samples up to 128 stations at
native normal offsets, allows up to 12 px end-to-end drift, and
requires coherent Lab chroma evidence in all four quarters. Only
stations inside the capture can vote. A span with less than 80%
sampleable stations prevents the cue. The chroma veto retains the
simulated separate 6 px rim with 15 intensity levels, 13 px seed
displacement, Gaussian blur 0.85 px and JPEG75. The first version
missed it. Across 50 targeted alternative-material controls the
cue is still ambiguous in four cases, including one identical
raster, two very faint rims and a compressed narrow gutter. Forty
perspective-stamped separate-print gap and overlap controls keep
their active proposals. Twenty-two nominated real-source
perturbations preserve the cue.

The violet proposal remains individually accessible in count
review. Orange marks the shared edge, teal the span inside the
already detected print and blue the span across pale material.
A reviewer can check that a separate print is visible and correct
its outline, or confirm the count with the current set of proposal
IDs. The pending proposal retains its exact native mask as a
neighbor claim during crop checks and reprocessing. Fresh upload
and reprocessing keep the other four masks, crop transforms, mode
records and preview JPEGs. Both crop-mode downloads and the legacy
cropped download remain held. The mobile and desktop editors show
the measured traces. The count action checks the current revision
and can be safely replayed.

The exact case improves from three matches and two active extras
to three matches, one active extra and one pending proposal. The
four tested encodings have 12 matches and one active extra at
IoU 0.85. The retained matched masks and rendered crop records
agree with profile 4. Against the rough, rotated page annotations,
the three whole-print source-footprint IoUs are 0.97297, 0.98963
and 0.98559. The respective maximum corner errors are 30.15, 8.98
and 10.45 native px. Strict content omissions are 4,126, 1,727
and 4,623 source centres. Output pixels with foreign contributions
beyond the one-pixel tolerance number 18,739, 4,727 and 4,538.
These errors still need correction. All three motifs remain held
without a rendered motif transform. Physical aspect measurements
and independent landmark pairs remain unavailable for this page.

An independent native edge audit of the left print reproduces
1,730 of the 4,126 draft omissions against both the unrotated
and rotated optical boundaries. Of those, 1,271 dark image centres
lie more than two px inside both fits. The strongest local
displacement is in the lower quarter of the left side. The pale
strip along the opposite edge is still disputed. Its visible
material affects the reported foreign-pixel count.

The other extra crosses two true scenes on diagonal lines and
borrows a real print side and a continuing album edge. If its
neighbor mask is omitted, both crop modes lose an occlusion
warning on the middle and right prints. A separate partly hidden
print with six visible rows and 1,980 visible source centres evades
the existing independent-content veto. This proposal and its
neighbor claim therefore remain active while the two source
scenes and album continuation are investigated.

The subsequent source probe finds 783 and 1,119 native pixels of
exclusive model interior across the two diagonals. At the stricter
Canny thresholds these contain 9 and 17 connected details crossing
both sides of the candidate line. Four- and nine-row partially
hidden paper controls retain these votes and both observed rails.
The same evidence persists in a single paper reproduction of the
album scene. A separately textured sheet with a physically occluded rear
edge also yields a high vote. This supports an advisory overlay;
the source test alone does not settle the physical print count.

The `fusion_ownership` calibration profile moves to revision 5.
The seven paired control reports retain their detection, scoring
and crop records. The frozen detector remains at 18/18 detections,
and the seeded geometry keeps all 11 whole-print and 12 motif
passes. On the exact case, the gated helper takes 111 ms median
and 123 ms p95 process CPU time across 31 warm single-thread runs.
At profile revision 5 the mobile geometry button on the uncertain
neighbor could be pressed and the server rejected its event.
The [mobile hold explanation](#explain-a-held-whole-print-in-mobile-geometry-review)
uses the same availability state as that verification.
Native, benchmark and browser receipts are local under
`runs/crop-quality/borrowed-fallback-edge/`.
The local reference check covers seven manifests, including Golden,
the paired-reference drafts and both archive calibration drafts.
It reports zero verified crops in either mode and zero shared
verified physical groups. The Adachi archive catalog and this
regression set add no release references or calibration scores.

### Review a diagonal fallback contour on the original

On the same 270° JPEG75 capture, classical cluster 3 remains
active. Its two long diagonals have source fit support 0.2589
and 0.2054. The two supported sides score 0.9821 and 0.8304.
The proposal is a singleton edge contour with confidence 0.768
and a mixed boundary score of 0.625. The cue considers it
where two model outlines cover between 42% and 72% of its
area in total, with at least 10% coverage each, and a third
model outline is available. Those outlines must have raw
confidence at least 0.40 and boundary evidence at least 0.65.
The check runs on the native source after the classical fallback
on pages with four to twelve active detections, for convex
proposals at least 4 native px away from the capture frame.

The two weak sides must each have one continuous span with
at least 26 native px clearance inside a different model outline
and from all other model outlines. The native band measures
connected Canny features reaching both sides of the line.
At least eight features, nine distinct 12 px bins and a span of
40% of the inspected strip are required. The actual source spans
cover about 779 and 1,113 px. They locate details to compare at
the original.

The independently refitted side of a third model print must
agree with one strong line. The other strong line must
continue outside the candidate, with consistent pale-side
polarity across its end. The actual external measured trace
is 180 px long. Known active and pending contours veto the
external measurement where they claim the sampled area.
Broad independent texture in the part outside the model masks
also vetoes the cue.

The first version matched a parallel extension with the pale
material on the opposite side. A matched-polarity requirement
keeps the authentic continuation and rejects the six reversed
polarity controls across shifts and re-encoding. A further
check excludes an actively or tentatively claimed extension
even when the claim comes from the classical detector.
Deliberately hidden 4- and 9-row rear prints retain all four
traces. The larger rear region and the isolated textured
corner veto the cue. A photograph of the whole scene printed
as one matte sheet still yields the same measured evidence.
The count remains a reviewer decision.

Desktop and Mobile show the four measured segments and the one
selected footprint. Opening and cancelling are local actions.
The explicit exclusion looks up the current page-count revision,
the selected UUID, the four traces, and the two held crop modes.
It records the original mask in review history. The existing
pending cluster 4 remains available. In the real round trip
the active count changes from four to three, with three matches
at IoU 0.85. Masks, both mode transforms, and stored JPEG bytes
of the three retained photos remain identical after
reprocessing. The middle and right photos lose the spurious
occlusion reason. Curved-edge, corner and aspect issues still
hold the named and legacy cropped exports.

The raw detector across the four encodings stays at 12 matches
and one active extra before count review. The helper takes 186 ms
median and 229 ms p95 CPU time for 31 warm calls with one OpenCV
thread. The `fusion_ownership` profile moves to revision 6.
Across eight paired replays, the only recorded change is the
review annotation on cluster 3. The frozen detector stays at
18/18, and the seeded geometry retains 11/11 whole-print and
12/12 motif passes.
Browser, reprocessing and native receipts are local under
`runs/crop-quality/diagonal-fallback-review/`.

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

A development replay on the two differently coloured `touching` prints
compared the model's original and refined masks as warp seeds. Using the
refined mask directly makes both modes of the left print geometrically
ready while its outer stock is still disputed. The two motifs can be
measured using only pixels inside the old warp. A diagnostic render with
owned sampling keeps their corners at least 6.2 original pixels inside
that warp. Against the frozen reference the worst motif corner errors
are 0.089 and 0.077 px. Both renders have zero strict cutoff and zero
foreign interpolation contributions. The same-tone and overlapping
controls abstain from the guarded diagnostic. This experiment is preserved
under `runs/crop-quality/golden05-paper-edge-comparison/adjacent-motif-probe/`.

### Show only the exposed edge of an overlapping print

In the frozen `overlap` capture, the ordinary model finds both
physical prints. The front print loses 8.24% of its visible source
footprint and omits 3,753 centres beyond one native pixel. The model
input has competing left and right paper lines. The bottom fit
unambiguously follows an ink line within the paper. Combining the
two physical side alternatives with that bottom still puts both
lower corners about nine native pixels inside the development
reference. A source search shifted outward by 6–12 pixels finds a
bottom line that brings all four corners within 0.61 pixels.

The alternate has zero shared source centres with the exact concave
rear reference. Its overlap with the reference convex hull is
12,491 px². The actual detector rear proposal claims 1,428 centres
within the alternate, including 1,205 in newly added paper. The
existing source-stock verifier corroborates 942 of those added
centres on the original and 754 after light blur. A replacement
based on contour intersections would need an independent explanation
of the rest of the claim.

A separate 29 × 6 px paper tab with the same sampled colour covers
170 centres of the alternate with identical source bytes. The source
comparison was also tested on 156 opaque overlay derivatives. The
combined T-junction and four-rim rule still accepts 96. A separate
small photographic fragment can pass those measurements and add
foreign content inside a complete render. These counterexamples
establish the limit of a whole-print interpretation.

The review code uses the two opposing disputed sides and two
supported sides as its entry condition. It samples the pale stock
from an observed side and fits the four paper lines on the original
to locate a possible outer strip. It requires one current neighbor
claim to grow into the strip. Only stations with a signed native
transition within 1.5 pixels of the line and enough clearance to
every other claim appear in the overlay. Each interval is shortened
by one pixel at each end. The frozen normal-path result adds one
turquoise span about 186 native pixels long. The four original
violet competing traces remain. Both mode states, the committed warp,
the detection results, the existing preview and the download decisions
match the baseline.

The exposed span repeats through four quarter turns and four encoding
and blur treatments when the original detector contours are carried
through the same transform. It measures 185.6–185.7 native pixels.
With the final native fit, 145 of 160 original stress treatments
still have an observable span, with zero centreline samples on the
painted overlay masks. The shortest distance to the detected rear
stays above ten pixels. On 208 additional treatments that put a separate
object directly on the exposed edge, 144 visible disruptions suppress
the measurement. The other 64 preserve the same apparent local source
step and can still put a review trace on a different physical object.
Supplying those objects as active claims excludes their areas from
the trace. The overlay suggests a possible paper edge and leaves both
physical ownership and metric aspect for review.

The helper takes about 13 ms median and 18 ms p95 in 72 warm
one-thread measurements of the original. In 392 paired pipeline
comparisons, both mode records and all rendered assets reproduce
apart from the optional trace metadata. The normal upload and
reprocess serve byte-identical previews on Desktop and Mobile.
Both named downloads and the legacy trimmed route still return
HTTP 404 for both prints.

On the separately examined Willett page 99, the two original
rear edges visibly terminate at the foreground print. In the native
source, the median Lab contrast of one rear edge drops from 183.1
to 1.2 after the junction. The other drops from 168.2 to 1.1. The
normal crop rectangles share 7,247 native centres. The foreground
crop crosses only seven centres of the source-draft visible rear
mask. The example is development evidence from a single album
capture. Independent physical and motif measurements are still
required for release. [Willett page 99](https://historycenter.mld.org/willett_album/items/col003_p99.html)

#### Trace the dark-page extras on Willett 99

The normal mixed replay has two matches and two extra detections on
the pinned 6,016 × 3,819 source, SHA-256
`848005c653f3d38624862ce5d07ebc32b1740de06ef3b320f6c994723b8d9213`.
Both extras are standalone full-page model proposals at confidence
0.549 and 0.486. The two matching detections come from classical
outlines supported by the source. The two extra positions are dark
album quadrants, with median Lab lightness 24 and 31. Their proposals
borrow bright edges of the nearby prints with the paper on the side
outside the claimed shape. The two legitimate crop sets are
byte-identical when the extra neighbor claims are omitted in a
shadow replay.

An original-pixel diagnostic marks both extras when the dark core,
continued backing and reversed paper transition agree. It also marks
nine of 20 truly separate, nearly colour-matched paper controls.
It leaves all 18 planted dark photographic prints unmarked across
three illumination levels and three border intensities. The ordinary
detector replay on three Willett pages, ten Golden views and sixteen
frozen pages supplies only three eligible weak model-only proposals.
Of those, the same two Willett 99 proposals are marked. The ambiguity
for near-matching paper requires a human count decision.

#### Review a dark album backing proposal

PhoSca applies the count hint after precise source-contour refinement
when normal inference used both the model and the classical fallback.
The original pixels must be available. An eligible proposal is a
single full-page model mask with confidence at least 0.30 and below
both 0.70 and the configured acceptance threshold. At least one
eligible neighboring print must have a classical border quadrilateral
corroborated by a model proposal. The check is limited to pages with
2–12 surviving detections and 1–6 eligible candidates. After checking
the Lab distribution in the dark core, it samples both sides of each
proposed edge. It requires at least three sides with dark continuation
or reversed lightness into a corroborated neighbor. Both types must
have visible spans. A strongly signed edge of the candidate or a
narrow seam tracing a separate sheet rejects the hint.

The two suspected areas on Willett 99 move from the active count to
two pending proposals. Against the development outlines, active count
precision moves from 2/4 to 2/2 while recall stays 2/2. The mean mask
IoU of the two matches stays 0.97126. Both actual prints stay active. On the native
upload, the actual prints keep identical mask, geometry and mode
records and byte-identical review JPEGs. The original model mask of
each open proposal stays in the neighbor coverage used for crop
validation. As a result, placing a proposal in review does not give
an adjacent crop permission to claim its pixels. The registered
`mixed-legacy` model SHA-256 is
`1e95a1a90ff897df7099ff80675186ec1d8145f7bb07559fe918b3dd3a6db748`,
with a 512-pixel input, proposal threshold 0.25 and acceptance 0.70.

The count editor shows two types of approximate spans along each
candidate quadrilateral. Turquoise spans show dark material on both
sides and orange spans show proximity to a brighter neighbor edge.
In a signed original-pixel fit, one orange span differs from the fitted
edge by up to 22.75 native pixels. The lines guide inspection of the
original. The four editable corners still need a count decision. The
separate near-matching textured stock test with darkened photo content
marks 5 of 24 deliberately distinct paper controls with fixed proposals.
Willett 9 and 95 and the NerdScan source and darker insert show no
hint in the development replays. The printed-photo controls justify
keeping the candidate available for review.

An explicit count confirmation with the current revision and the list
of reviewed pending proposals dismisses the dark candidates. Editing
one candidate instead requires confirming that a separate paper print
is visible. This adds that print while leaving the other candidate
pending. Both modes require a second, explicit paper confirmation.
The original-pixel fit must locate all four corners within one native
pixel of the submitted outline. Each side needs at least 90% edge
support and 80% corner support. The usual geometry, source-pixel,
protected-content and aspect checks follow. On Willett 99, deliberately
adding the two empty quadrants through Desktop and Mobile keeps both
named exports and the legacy trimmed download unavailable. The inline
untrimmed preview remains available. The same-corner paper correction
and reprocess of an empty quadrant also retain the paper review reason.

The five frozen and development comparison runs show equal aggregate
detection and crop metrics, equal mode availability and no extra
pending dark proposals. They include 18/18 detections in the locked
synthetic replay, 23/23 recoverable cases in both seeded crop modes
combined and 29/29 detections in the mixed Golden replay. The regular
locked synthetic run still has 6/11 full-print and 7/12 motif crops
ready. The native count workflow moves one development page into
explicit review and adds two count offers, with no new crop hold for
either genuine print. Ordinary synchronous upload for that page took
40.89 and 44.35 seconds in two current native runs. The 56.42-second
baseline upload overlapped other jobs, so these times do not establish
a throughput change. The current evaluator measured 10.57 seconds for
segmentation and 41.06 seconds from source decode through JPEG writes.
Receipts and visual development checks are kept
locally under `runs/crop-quality/willett-dark-backing-review/`.

The `dark_album_review` validation profile revision applies to the
precise classical route without a learned visible-mask refiner. Detector
calibration must include that revision when this route is enabled. The
Willett comparisons are development evidence from one physical album.
Automatic release remains gated by the separate calibration and final
physical album references. [Willett page 99](https://historycenter.mld.org/willett_album/items/col003_p99.html)

### Recover a count offer from red photo corners and open paper edges

Willett page 95 contains three physical prints. Red oval holders cover
the corners of the upper-left American Falls dam photograph. The
registered `1e95a1a9…` checkpoint produces no model region on that
photograph in the native raster, its half-size derivative or the
800 × 507 publisher JPEG. Four overlapping model tiles at each of five
coverages also fail to find a complete print. The classical detector
retains a partial region at native size with draft IoU 0.80966. At half
size and publisher size its best regions reach 0.00321 and at most
0.0019. The missed count begins in proposal generation.
[Willett page 95](https://historycenter.mld.org/willett_album/items/col003_p95.html)

Unclosed bright components and signed edges gave useful draft outlines
on the two reduced views. On the Red Album cover, the same optical
test also follows the decorated title leaf across the gutter. The
selected rule uses a more specific physical witness. On a dark neutral
page, four disjoint compact red components must form a plausible set
of print corners. The interior must contain image detail. A fifth
mount inside or just outside the outline vetoes the candidate because
it can belong to a separate print. No existing claim may cover more
than 2.5% of the proposed footprint.
On at least three sides, two separate spans must have a fitted line
and a dark-to-bright transition on the original raster. The stored
quadrilateral joins the mount centres as an editable starting outline.
It exposes neither the hidden tips nor a measured aspect ratio.

| Source | Existing target mask IoU | Count offer IoU against paper draft | Measured spans |
| --- | ---: | ---: | ---: |
| Native 6,016 × 3,812 | 0.80966 | Existing claim vetoes a second offer | 0 |
| OpenCV AREA50, 3,008 × 1,906 | 0.00321 | 0.95276 | 7 |
| Publisher JPEG, 800 × 507 | ≤ 0.0019 | 0.94826 | 7 |

The native JPEG SHA-256 is
`1cf86c7d9c1688a07c156aabc4cb11edbfd4f9860959bd14e56d6ce8fb5d8a73`.
The publisher derivative SHA-256 is
`b72a9d0db344fde822223dce8fe7d4e49217248d8e81ab0c2b5108ca6d563caf`.
The draft extrapolates under the holders. It supports locating the
print for count review. Exact corner, aspect, projected content and
motif measurements require separate unobstructed references.
Willett 80 shows a second copy of the scene with different visible
paper borders. A native scene registration from page 80 to page 95
has 338 held inliers, median error 1.04 pixels and p95 3.25 pixels.
The registered points describe visible scene details. The repeated
image stays in the same album group.
[Willett page 80](https://historycenter.mld.org/willett_album/items/col003_p80.html)

The offer is separate from the active count and the existing crop
masks. The original-image editor shows the measured spans along the
visible paper and prompts the reviewer to set the corners. Confirming
the count can withdraw the offer. Adding the corresponding print
enters the existing projective-outline review and requires separate
paper confirmation. Both modes remain held while the visible edges
cannot validate the proposed outline.

The publisher upload keeps three rows, and the half-size upload
keeps four. Each gains one deferred offer. The existing masks, mode
records and preview JPEGs match the previous revision. Desktop and
mobile require explicit confirmation of a separate print before
adding it. Manual addition, reprocessing and an explicit check of
unchanged corners keep both crop modes and the legacy trimmed
download held. The ordinary preview remains available. Source-only
testing finds the offer across all four turns of both raster sizes.
One of four publisher JPEG55 quarter-turn cases abstains after
recompression. All four tested JPEG75 views still show the offer.

The independent development replay screened 170 views. Forty reached
the normal segmentation replay and 35 invoked the mount check. The
two Willett 95 views above were the only new offers. The replay also
kept the Red Album title leaf and the separate mounted pairs on
Willett 38 and 46 out of this recovery path. Several views are
derivatives of one acquisition. These are development controls.
With cached claim masks and one OpenCV thread, the isolated check
took 93 ms median on the native page and 130 ms on the half-size
page. The eight paired crop reports match the baseline on
detection, measured geometry and availability in both modes. The
frozen detector still matches 18/18 prints, and the seeded crop
run passes 11/11 whole prints and 12/12 motifs.

The new count check uses `dark_album_review` validation revision 2.
The source and development
receipts are recorded locally under
`runs/crop-quality/dark-mounted-print-review/`.

```bash
python -m pytest -q backend/tests/test_pipeline/test_fragmented_prints.py -k red_mount
```

### Show the visible paper edges of an already counted mounted print

In the 6,016 × 3,812 Willett 95 source, the active upper-left
classical contour omits parts of the dam photograph. The four
holder centres enclose 99.19% of that contour. The contour covers
85.03% of the provisional holder quadrilateral. The ordinary upload
adds seven measured spans. Three lie wholly outside the old outline
by at least 177.7 px on top, 90.2 px on the right and 76.1 px below.
The top span alone is about 699 native pixels long.
[Native scan and catalogue](https://historycenter.mld.org/willett_album/items/col003_p95.html)

When the count search finds no independent offer, the algorithm can
attach measured spans to one weak classical claim. The existing
claim must be at least 97% inside the holder quadrilateral and
cover 60–90% of it. At least two measured spans must lie outside
the claim with clearance at both ends and at the midpoint. Other
photo claims and open count proposals still veto an overlap.
The editor draws the native measurements on the original page.
Both modes use the projective-outline review until the physical
corners can be verified. An unchanged geometry confirmation keeps
the open spans. A replacement outline gets its own geometry check.
The active count, source mask contour and current warp are kept on
upload. The mounted edges contribute only to the review overlay.

The native multipart replay gives three active rows and seven
segments on the clipped print. Desktop and mobile each display the
same seven segments. A normal reprocess preserves all three ordinary
JPEGs byte-for-byte. The explicit same-corner save on this source is
rejected by the protected-content check. The overlay and current
warp remain available for review, and both download modes and the
legacy trimmed download remain held. A separate API fixture covers
the accepted same-geometry save and checks that the spans persist
in the held mode.

The provisional geometry cannot establish the paper underneath the
holders. A separate experiment fit a quadrilateral through the four
rails. Against the native development draft, it recovered 771,210
visible source pixel centres and reduced omitted visible paper from
18.03% to 1.21%. A comparison with the second copy on Willett 80
found three distinct bridge landmarks missing in the old crop.
All three fall inside the experimental outline. The larger warp
also includes about 49,262 holder pixel centres. Its estimated
foreign bilinear sample weight is 1.26%. The preview guard rejects
the render because a new face-detector region crosses the upper-left
holder. Revision 3 shows the measured open spans on the original.

In 40 paired registered replays, this annotation appeared only on
native Willett 95. Both smaller Willett 95 count offers matched the
baseline, as did all active polygons and pending count records across
the replay. The partial check ran 12 times. Its median and p95 elapsed
times on one OpenCV thread were 73.9 and 290.8 ms under shared load.
Whole-print and motif scoring agree across all eight paired regression
reports. The frozen detector matches 18/18 prints and the seeded
geometry run passes 11/11 whole prints and 12/12 motifs.
`dark_album_review` advances to revision 3. Receipts remain local
under `runs/crop-quality/truncated-dark-mount-review/`.

### Compare only measured strips beside a mounted print

The larger trial warp for native Willett 95 regains the bridge and
also crosses the upper-left protected region and the physical holders.
Four closed red rings alone cannot establish which pixels they
cover. In four of twelve controls with a low-contrast printed subject
inside a ring, the ring remains optically consistent. The review
preview never uses its interior as evidence to delete image content.

The narrower comparison starts with the existing source footprint.
It selects a dominant open paper edge and resamples the original
pixels at neighbouring stations. Each addition needs an unambiguous
dark-to-bright transition with a valid backing sample. Short and
unsupported runs leave gaps on the transparent canvas. The preview
uses the existing homography, translated so that the supported
extension fits on the canvas. The saved crop retains its current warp.

Source-centre coverage checks require every old centre to remain.
An addition that intersects a protected region or another photo claim
is rejected. A closed red contour and its enclosed centre act as an
overlap veto. Added pixels must also avoid the one-pixel neighbourhood
of that contour. The premultiplied warp uses only the resulting
source coverage. A subsequent manual claim on the measured expansion
band removes the preview.

On the 6,016 × 3,812 source, the held PNG measures 2,721 × 1,866.
Against the development draft, the preview adds 279,887 visible source
centres, loses zero old centres and reduces the omitted visible
fraction from 18.02% to 11.91%. The addition introduces zero
out-of-draft source centres and zero output pixels with foreign
interpolation weight. The known holder contours contribute zero
source taps. One of three previously missed bridge landmarks is
visible in the comparison. The remaining gaps and the physical
corners still need review.

All four quarter-turn inputs with rotated source annotations produce
the same native centre counts. The isolated check took 300–432 ms
of CPU time with one OpenCV thread. The native upload preserved three
active rows, their warps and ordinary JPEG bytes. Desktop and mobile
display the JPEG and PNG together. The primary field's two-pixel
interior has full preview coverage. Nineteen output centres exactly
on the old left viewport seam have zero alpha within numerical
precision of the boundary. Reprocessing reproduces the PNG bytes.
A manual test claim placed solely within the addition retires the
preview, and another reprocess keeps that comparison withdrawn.

The PNG belongs to the review of **Ganzer Abzug**. Both export
statuses are held, and the motif mode retains its own boundary
review. The same source measurements appear on the original page.
All eight paired benchmarks match the previous measured results in
both crop modes. The frozen detector still matches 18/18 prints and
the seeded run passes 11/11 whole prints and 12/12 motifs.
The `dark_album_review` validation profile advances to revision 4.
Native samples and alpha metrics are local under
`runs/crop-quality/mounted-bridge-strip/`.

```bash
python -m pytest -q backend/tests/test_pipeline/test_fragmented_prints.py -k mounted_strip
```

### Preview a motif with one disputed paper edge

The `white_margin` and `camera_tilt` controls already had a supported
paper-stock fit for the one disputed edge. Their whole-print preview
was within 0.50 native pixels of the known corners. The ordinary motif
path deferred all inner geometry because of the paper ambiguity.
The isolated preview path checks four physical motif edges, refits
in the original source and requires a continuous pale paper rim.
It also checks clearance inside the paper and from other proposed
prints. It uses the existing protected-content checks.

The ordinary margin checks run before the optional preview. A second
check looks for compact deviations between similar local paper flanks
in the rectified margin. A positive check marks the source location
and withholds the tight view. It can indicate uncertain marks as well
as readable writing. In the paired control the fallback catches the
small graphite and coloured captions even after JPEG75 compression.
The clean mark controls keep the preview in all four rotations.
The 16 marked controls with four caption and encoding combinations
all withhold it. A separate neighboring sheet with a four-pixel
gutter withholds the preview, while the seven-pixel variant keeps it.
These results measure only the supplied development controls.

| Frozen motif preview | Footprint IoU | Max corner error | Aspect error | Landmark p95 |
| --- | ---: | ---: | ---: | ---: |
| `white_margin` | 0.99187 | 0.077 px | 0% | 0.052 px |
| `camera_tilt` | 0.99137 | 0.084 px | 0.658% | 0.069 px |

Both previews have zero cutoff beyond one native pixel and zero
foreign interpolation contributions. The whole-print pixels are
identical before and after in seven paired checks. The regular
geometry metric still excludes the optional motif PNG. Its preview
asset is available inline through `image_only_preview`. Approving the
task or resubmitting the tentative paper together with the motif
without an explicit paper confirmation is rejected. The three
trimmed variants remain held through upload and reprocessing.
For the marked comparison, saving only the paper corners and known
paper dimensions releases the manually confirmed whole print. The
source mark is carried into the new motif review and survives
reprocessing. The motif export stays held until a separate decision.
On the 820 × 670 isolated clean control, the median crop step rises
from 86 ms to 180 ms over ten alternating warm one-thread runs.

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

With complete spatial quadrants held out, the p95 residual of the same
American Colony landmarks is 2.890 native pixels for axis-aligned crop
and resize, 0.467 for an affine fit and 0.383 for a projective fit. Across
nine paired draft mode records, affine p95 is 0.153–0.910 pixels. The
coordinate matches remain compatible with digitally transformed
derivatives. Capture independence still needs the actual two rasters.

CU Boulder publishes whole-album records and item-level catalogue
links for four Ira Wolff albums. In the Civil War Carte-de-Visite
Album, 78 individual records have size measurements and a No Copyright
United States statement. The Lincoln card `0m746x06r0fj` retains its ARK
across two published file names and is catalogued at 2.25 × 3.25 in.
The item and position can be matched from metadata. Native raster
requests returned HTTP 403, and the separate acquisition of the print
still needs verification. [Collection](https://doi.org/10.25810/9e4g-k065) ·
[Pinned catalogue exports](https://github.com/cuboulder-MSD/digColl_iraWolffPhoto/tree/9659d61fc9d2de31473b1f650bd21d5bbf9b8b7c) ·
[Lincoln card record](https://ark.colorado.edu/ark:/47540/0m746x06r0fj)

The Dieulafoy catalogue mirror indexes 774 pages across eight INHA
albums. INHA publishes an Etalab Open Licence for its digitized
materials, including commercial reuse with attribution. The native
Perse 1 canvas 65 request and a separately published Perse 1 derivative
both returned HTTP 403. The exact page and independent single-print
pairing still need raster evidence. These catalogue checks added zero
verified calibration or release groups.
[INHA reuse terms](https://www.inha.fr/bibliotheque/collections/la-bibliotheque-numerique-de-linha/) ·
[Pinned catalogue index](https://github.com/moarthist/dieulafoy/tree/e8d67dc03a096d569feb380004fbb2101841e1f8) ·
[Published derivative](https://ac.nau.edu/omeka-s/s/pioneering-women-archaeologists/item/4181)

The Barcelona maritime museum documents separate captures of the
17 prints in Antonio Calvo album `102638F` after removal. The
archive also photographs the same pieces after fixing them to new
leaves. New mounts overlap the white print rims, so recoverability
must be assessed separately for the whole print and motif. The
catalogue identifies print `102638F-10` at 110 × 180 mm. The
catalogue JPEG returned HTTP 403 and the native album leaves and
their image reuse terms remain unverified. The publication licence
covers the article. [Conservation report](https://doi.org/10.55437/unicum.324) ·
[Print record](https://www.mmb.cat/catalegs/moll-de-carrega-del-port-de-valencia-i-vapor-santo-domingo/)

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

The Turku City Museum Ramsay collection `VA9901` catalogues seven
physical family albums and 725 individually described prints. Volume 1
lists `VA9901-1-a1.tif` through `VA9901-1-a26.tif` as page media. The
Public Domain Mark print `VA9901:1:48` lists `VA9901-1-48.tif` and the
same reproduction photographer as the volume. The exact page match
and two native rasters remain unverified after the API and preview
requests returned HTTP 403. Screen for duplicated images between the
volumes before treating them as independent groups.
[Ramsay collection](https://finna.fi/Record/aipa.node-8323?lng=en-gb) ·
[Volume 1](https://www.finna.fi/Record/tmk.161614005277700) ·
[Print record](https://www.finna.fi/Record/tmk.162305386114000)

The Boston Public Library finding aid for `MS q G.3.30` records that 85
original prints were physically removed, separately photographed in June
and July 2025, and returned to their original pockets. The public set
catalogues the parent album and individual print records, including item
78 `06_01_019931_0078` (William P. Lucas, archival component 74908)
and item 6 `06_01_019931_0006` (Robert Gould Shaw, component 74834).
The album's archive digital object is `do:180`. Its mounted views
contain four-slot leaves. The catalogue derivative of page 44 has yet
to be linked to the slot of either print. The native IIIF and public
DPLA metadata requests returned HTTP 403. Exact leaf-to-print mapping,
native raster hashes and visible stock boundaries still need verification.
Commons separately lists BPL-exported derivatives of pages 10, 32
and 68 at 3,575 × 2,640, 3,500 × 2,612 and 1,862 × 2,846 pixels.
Their slot correspondence to Lucas or Shaw has not been established.
The published image rights designation is No Copyright – United States.
[Finding aid](https://archives.bpl.org/repositories/2/resources/180) ·
[Album digital object](https://archives.bpl.org/repositories/2/digital_objects/7564) ·
[Image collection](https://www.digitalcommonwealth.org/collections/commonwealth:63964b37k) ·
[Item 6](https://ark.digitalcommonwealth.org/ark:/50959/8910t015r) ·
[Page 10](https://commons.wikimedia.org/w/index.php?title=File:Photograph_album_of_teachers_of_the_New_England_Freedman%27s_%26_Union_Commission_-_DPLA_-_b4b7e02983eb15ad31e81dcf51fb207b_(page_10).jpg&oldid=1161726133) ·
[Page 32](https://commons.wikimedia.org/wiki/File:Photograph_album_of_teachers_of_the_New_England_Freedman%27s_%26_Union_Commission_-_DPLA_-_b4b7e02983eb15ad31e81dcf51fb207b_(page_32).jpg) ·
[Page 68](https://commons.wikimedia.org/wiki/File:Photograph_album_of_teachers_of_the_New_England_Freedman%27s_%26_Union_Commission_-_DPLA_-_b4b7e02983eb15ad31e81dcf51fb207b_(page_68).jpg)

The 1908 Hungarian family album on Commons separately catalogues 17
mounted page JPEGs and 68 individual TIFFs. The uploader links print
001 to page 1. The public derivatives show a nine-minute interval in
their scan records. The catalogue lists page 01 at 10,970 × 14,816
with SHA-1 `ab172e358af0c7de96ae55c73e37e12cee3f55d3` and print 001
TIFF at 3,694 × 4,836 with SHA-1 `299834405988e550651762c4704cebf5eef3a802`.
MediaWiki image info, Commons File REST metadata and the public Your.Org
mirror requests returned 403 from this workspace. PICRYL indexes
derivatives via GetArchive. Native
raster identity is unverified. The native pixels are still needed to distinguish
separate exposure from a transformed copy and to verify the visible
paper and motif for both modes.
[Hungarian album](https://commons.wikimedia.org/wiki/Category:Photograph_album_of_an_unknown_Hungarian_family_(1908)) ·
[Page 01](https://commons.wikimedia.org/wiki/File:Hungarian_family_photo_album_1908,_page_01.jpg) ·
[Print 001](https://commons.wikimedia.org/wiki/File:Hungarian_family_photo_album_1908,_image_001.tif)

The Museums & Galleries NSW guidance illustrates an album page scanned
before a separate image capture. The illustrated album lacks an accession
identifier. Item `9873_a025_a025000097` is the preceding illustration
of two sides of a separate photograph of Dr Lawrence William Cock.
The mounted and individual album files and their reuse terms remain open.
[Scanning guidance](https://mgnsw.org.au/wp-content/uploads/2019/01/Digitising-your-collection-Part-4-Scanning-and-handling-tips.pdf#page=6)

AfricaMuseum records high-resolution scanning of 538 albums and identifies
an example physical album as `HP.1929.613.1`. The project reattached
loose photographs before scanning and exposed the mounted page images
through its IIIF intranet. The record has no identified independent
capture of an individual physical print for the illustrated album.
[Digitization report](https://www.africamuseum.be/sites/default/files/media/about-us/annual-reports/doc/annualreport2017.pdf#page=46)

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

The Starrett Scrapbook, accession `2002.05`, has a documented batch
of 94 mounted page scans and 511 independent scans of the prints
after removal. The finding aid maps page 1 to prints 1–3 and page 2 to
prints 4–6. The three page 1 prints measure 120.65 × 92.075 mm each.
The public gallery provides page images. The matched native print
scans and usage permission require archival access. The catalogue
dimensions establish physical paper proportions. Motif dimensions
and visible corner coverage still need native pixel verification.
[Digitization project](https://www.skyscraper.org/viva2/scrapbook/) ·
[Print and page crosswalk](https://old.skyscraper.org/viva2/collections/findingaids/esbb.html) ·
[Image access and rights](https://www.skyscraper.org/rights-reproductions/)

Grey Roots identifies the Aberdeen volume 1 (`WIF22`) and Woodford
volume 1 (`WIF56`) as separate Tweedsmuir books. The archive documents
page scans and additional high-resolution scanning of each photo.
Native paired rasters, exact print identifiers and evaluation rights
remain unverified. The workspace requests for the public PDFs
returned HTTP 403. Both collections remain source leads.
[Digitization project](https://greyroots.com/collections-and-research/archives/tweedsmuirs) ·
[Aberdeen](https://greyroots.com/sites/default/files/aberdeen_tweedsmuir_vol._1.pdf) ·
[Woodford](https://greyroots.com/sites/default/files/woodford_tweedsmuir_history_vol._1_part_1.pdf)

An exact-byte inventory recheck examined 13,070 readable local image
files and 22 image members in 11 archives against the seven tracked
reference packets. It found none of the 26 pinned page images or the
12 separate reference files. Direct attempts to restore all 18 pages
from the parked Getty, Rijksmuseum and Met calibration manifest
received HTTP 403 from the network tunnel before any image arrived.
No image was scored and the draft split roles are unchanged.

Te Papa catalogues physical albums `AL.000638` and `AL.000640`
with linked print objects and a public reuse declaration for the
marked media. In `AL.000640`, individual records `O.048980`,
`O.048985` and `O.048987` record motif dimensions of 292 × 190,
206 × 153 and 289 × 188 mm. The catalogue has to be checked on
the original pixels for the leaf-to-print links, physical stock
corners and independent captures. The official media and catalogue
gateway returned HTTP 403 in this intake. These albums remain
source leads. [Album AL.000638](https://collections.tepapa.govt.nz/object/1840035) ·
[Album AL.000640](https://collections.tepapa.govt.nz/object/1840095) ·
[Media terms](https://www.tepapa.govt.nz/learn/research/datasets/collections-api/api-terms-of-use)

The Pembina Manitou Archive records the Père J. Brachet album as
`PMA_2024_076806`. The digitization report documents an original
flatbed capture of every complete album page and separate 300 dpi
scans of more than 200 original photos. The later Word workbook
rearranges the photographs and captions. Individual scans were
adjusted to standard print sizes and tone ranges. Native page and
print files, exact pair identifiers and rights are still needed to
establish usable pairs. The archive record leads to a login.
[Archive record](https://pembinamanitouarchive.ca/documents/76806) ·
[Digitization report, page 6](https://www.mhs.mb.ca/heritage/downloads/expresssummer09.pdf)
