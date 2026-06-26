# Label Studio Annotation

Run Label Studio with local file serving enabled for the repository `data/` directory:

```powershell
docker run -it -p 8080:8080 `
  -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true `
  -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data `
  -v ${PWD}/data:/label-studio/data `
  heartexlabs/label-studio:latest
```

Create a project with this labeling interface:

```xml
<View>
  <Image name="image" value="$image"/>
  <PolygonLabels name="label" toName="image">
    <Label value="photo" background="#0f766e"/>
  </PolygonLabels>
</View>
```

## Recommended Workspace Flow

Create the training workspace and editable preannotations:

```powershell
python scripts/setup_segmentation_training.py bootstrap
```

The command copies the existing example album pages into `data/segmentation_training/images/` when that folder is empty or missing images, then writes `data/segmentation_training/label_studio_tasks.json`. You can add more album page images to `data/segmentation_training/images/` at any time and rerun the command.

Import `data/segmentation_training/label_studio_tasks.json` into Label Studio. Correct the polygons so each complete printed photo is one `photo` instance. Do not label internal regions inside a photo, handwriting, page borders, or album frames.

Export the reviewed project as JSON to `data/segmentation_training/exports/label_studio_export.json`, then run:

```powershell
python scripts/convert_label_studio_to_yolo.py --input data/segmentation_training/exports/label_studio_export.json --output data/segmentation_training/labels
python scripts/setup_segmentation_training.py train
```

The `train` command prepares the deterministic train/val/golden split, validates YOLO polygon labels, trains the one-class segmentation model, and exports `models/yolov8-seg-album.onnx`.

## Lower-Level Commands

Create import tasks from `data/raw_album_pages`:

```powershell
python scripts/export_label_studio_tasks.py --source-images data/raw_album_pages --output data/label_studio_tasks.json --preannotate
```

Import `data/label_studio_tasks.json` into the project. The optional `--preannotate` flag exports the classical segmentation fallback as editable polygon predictions, which should still be reviewed before export.

Export the reviewed project from Label Studio as JSON, then convert it to YOLO segmentation labels:

```powershell
python scripts/convert_label_studio_to_yolo.py --input data/label_studio_export.json --output data/label_exports/yolo
```

You can also place a direct Label Studio YOLO export in `data/label_exports/yolo/` if you prefer. Then create the train/val/golden split with `python scripts/prepare_yolo_dataset.py --source-images data/raw_album_pages --source-labels data/label_exports/yolo`. The prepared labels are placed beside split images:

- `data/yolo_dataset/labels/train/*.txt`
- `data/yolo_dataset/labels/val/*.txt`
- `data/golden_fixtures/labels/*.txt`

Golden fixture labels must never be copied into the training or validation folders. Before training, run `python scripts/validate_yolo_dataset.py --data data/data.yaml` to verify label completeness, polygon validity, and golden split isolation.


## In-App Review Export

If you review and correct photo corners in PhoSca instead of Label Studio, export those saved masks directly:

```powershell
python scripts/export_reviewed_yolo_labels.py --album-id <album-id> --output data/label_exports/yolo --manual-only --require-complete
```

Use `--manual-only` when the labels should include only masks explicitly saved by a reviewer. Drop that flag to export all current segmentation masks for bootstrapping.

## Orientation Training Crops

After reviewing extracted photos, copy a small curated set of correctly upright crops into `data/orientation_photos/`. These should be photo crops, not raw album pages. `scripts/train_orientation_model.py` creates rotated variants automatically and trains the 0/90/180/270 correction classifier exported to `models/photo-orientation.onnx`.
