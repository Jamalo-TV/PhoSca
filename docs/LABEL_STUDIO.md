# Label Studio Annotation

Run Label Studio with:

```powershell
docker run -it -p 8080:8080 -v ${PWD}/data:/label-studio/data heartexlabs/label-studio:latest
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

Import images from `data/raw_album_pages`.

Export YOLO segmentation labels. Place labels beside the split images:

- `data/yolo_dataset/labels/train/*.txt`
- `data/yolo_dataset/labels/val/*.txt`
- `data/golden_fixtures/labels/*.txt`

Golden fixture labels must never be copied into the training or validation folders.

