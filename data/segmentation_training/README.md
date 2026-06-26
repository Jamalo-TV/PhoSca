# PhoSca Segmentation Training Workspace

This folder is the low-friction workspace for the dedicated YOLO segmentation model.

## Folders

- `images/`: put album page images here. The bootstrap command copies the existing examples from `data/raw_album_pages`.
- `labels/`: YOLO segmentation `.txt` labels go here, one file per image stem.
- `exports/`: put reviewed Label Studio JSON exports here, usually as `label_studio_export.json`.

YOLO polygon lines use one `photo` instance per line:

```text
0 x1 y1 x2 y2 x3 y3 x4 y4 ...
```

Coordinates are normalized to `0..1`.

## Fast Path

```powershell
python scripts/setup_segmentation_training.py bootstrap
```

Open Label Studio, import `data/segmentation_training/label_studio_tasks.json`, and review the preannotated `photo` polygons. Every complete printed photo should be one polygon; do not label internal sky/water/handwriting regions.

After exporting reviewed Label Studio JSON to `data/segmentation_training/exports/label_studio_export.json`:

```powershell
python scripts/convert_label_studio_to_yolo.py --input data/segmentation_training/exports/label_studio_export.json --output data/segmentation_training/labels
python scripts/setup_segmentation_training.py train
```

The train command prepares `data/yolo_dataset`, validates labels, fine-tunes `yolo11n-seg.pt`, and exports `models/yolov8-seg-album.onnx`.
