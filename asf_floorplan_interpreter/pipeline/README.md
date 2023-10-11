## Reain a model to recognise doors and windows

## Roboflow data

There is [an existing annotation dataset of UK floorplans](https://universe.roboflow.com/prop/room-separation-instance/dataset/5) on Roboflow.
This consists of:

- Training = 1454 images
- val = 65 images
- test = 54

Polygon annotation with 6 classes:

1. Door
2. Double door
3. Folding door
4. Room
5. Sliding door
6. Window

To download this data, from this projects parent directory run:

```
aws s3 cp --recursive s3://asf-floorplan-interpreter/data/roboflow_data/ datasets/data/roboflow_data
```

Warning: ultralytics requires the parent data folder to be called "datasets".

## Training

Train: YOLOV8n (smallest) for instance segmentation (docs)

```
cd asf_floorplan_interpreter/pipeline/
python train_yolo.py --datastore=s3 --package-suffixes=.txt,.yaml,.jpg run
```
