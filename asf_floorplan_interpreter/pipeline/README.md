## Reain a model to recognise doors and windows

## :file_folder: Roboflow data

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

This dataset is stored on S3 [here](s3://asf-floorplan-interpreter/data/roboflow_data/).

Warning: ultralytics requires the parent data folder to be called "datasets".

## :muscle: Training

Train: YOLOV8n (smallest) for instance segmentation (docs) using metaflow:

```
cd asf_floorplan_interpreter/pipeline/
python train_yolo.py --package-suffixes=.txt,.yaml,.jpg --datastore=s3 run --config_file window_door_config.yaml
```

This will train the model and output the best model in [this S3 location](https://s3.console.aws.amazon.com/s3/buckets/asf-floorplan-interpreter?region=eu-west-2&prefix=window_door_config/&showversions=false).

The metaflow files are in this location: https://s3.console.aws.amazon.com/s3/buckets/open-jobs-lake?prefix=metaflow/FloorPlanYolo/&region=eu-west-1
