# Using Prodigy to tag entities

Activate the environment, if you need to run the Prodigy commands you will also need to do:

```
pip install prodigy -f https://[YOUR_LICENSE_KEY]@download.prodi.gy
```

Download the models to `outputs/models/rooms-model/best.pt` and `outputs/models/windows-doors-model/best.pt`.

```
aws s3 cp s3://asf-floorplan-interpreter/models/scoping/model-rooms/best.pt outputs/models/rooms-model/best.pt
aws s3 cp s3://asf-floorplan-interpreter/models/scoping/model-windows-doors/best.pt outputs/models/windows-doors-model/best.pt

```

## Data

Download and format the floorplan data into the jsonl file format needed for Prodigy annotation:

```
python asf_floorplan_interpreter/pipeline/annotation/create_images_data.py

```

This will create the file `asf_floorplan_interpreter/pipeline/annotation/floorplans.jsonl`.

## Windows, doors and stairs:

Since we have a model for identifying windows and doors, and its a long annotation task to tag every one, we can use the model to help us label.

Manually correct the original model for doors and windows, and label stairs from scratch:

```
prodigy classify-window-door window_doors_staircase_dataset asf_floorplan_interpreter/pipeline/annotation/floorplans.jsonl WINDOW,DOOR,STAIRCASE -F asf_floorplan_interpreter/pipeline/annotation/floorplan_recipe.py

```

```
prodigy db-out window_doors_staircase_dataset > asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/window_door_staircase.jsonl
```

Save it to S3:

```
aws s3 cp asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/window_door_staircase.jsonl s3://asf-floorplan-interpreter/data/annotation/prodigy_labelled/131123/window_door_staircase.jsonl

```

## Room model:

This will use the pretrained model to help label ROOM.

```
prodigy classify-rooms room_dataset asf_floorplan_interpreter/pipeline/annotation/floorplans.jsonl ROOM -F asf_floorplan_interpreter/pipeline/annotation/floorplan_recipe.py

```

```
prodigy db-out room_dataset > asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/room_dataset.jsonl
```

Save it to S3:

```
aws s3 cp asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/room_dataset.jsonl s3://asf-floorplan-interpreter/data/annotation/prodigy_labelled/131123/room_dataset.jsonl

```

## Room type model:

### Task type 1

Use the pretrained model to identify rooms, then annotation which room type they are

```
aws s3 sync s3://asf-floorplan-interpreter/models/room_config_yolov8m/ models/room_config_yolov8m/

prodigy room-type room_type_dataset asf_floorplan_interpreter/pipeline/annotation/floorplans.jsonl RESTROOM,BEDROOM,KITCHEN,LIVING,GARAGE,OTHER -F asf_floorplan_interpreter/pipeline/annotation/floorplan_recipe.py

```

```
prodigy db-out room_type_dataset > asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/room_type_dataset.jsonl
```

Save it to S3:

```
aws s3 cp asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/room_type_dataset.jsonl s3://asf-floorplan-interpreter/data/annotation/prodigy_labelled/131123/room_type_dataset.jsonl

```

### Task type 2

Using the room polygons labelled in the above Prodigy task, annotate which room type they are

```
# If you don't already have the labelled data:
aws s3 cp s3://asf-floorplan-interpreter/data/annotation/prodigy_labelled/131123/room_dataset.jsonl asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/room_dataset.jsonl

prodigy room-type-from-labels room_type_from_labels_dataset asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/room_dataset.jsonl RESTROOM,BEDROOM,KITCHEN,LIVING,GARAGE,OTHER -F asf_floorplan_interpreter/pipeline/annotation/floorplan_recipe.py

```

```
prodigy db-out room_type_from_labels_dataset > asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/room_type_from_labels_dataset.jsonl
```

Save it to S3:

```
aws s3 cp asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/room_type_from_labels_dataset.jsonl s3://asf-floorplan-interpreter/data/annotation/prodigy_labelled/131123/room_type_from_labels_dataset.jsonl

```

## Acceptance data:

Run through all the floorplans in Prodigy to accept whether they are suitable or not.

Unsuitable ones would be ones which are 3D.

This is helpful for two reasons:

1. To make sure we don't evaluate on images that are outliers
2. To create the metadata of image size, which Prodigy adds to a "image_manual" task quite conveniently (but doesn't seem to for a "choice" task)!

```
prodigy label-quality quality_dataset asf_floorplan_interpreter/pipeline/annotation/floorplans.jsonl GOOD -F asf_floorplan_interpreter/pipeline/annotation/floorplan_recipe.py

```

```
prodigy db-out quality_dataset > asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/quality_dataset.jsonl
```

20 out of the 497 floor plans in `floorplans.jsonl` were rejected for being 3D or not actually floor plans.

Save it to S3:

```
aws s3 cp asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/quality_dataset.jsonl s3://asf-floorplan-interpreter/data/annotation/prodigy_labelled/quality_dataset.jsonl

```

## Clean up

YOLO reads and saves each floorplan jpg, so you will end up with the directory being full of these. To clean them up run:

```
rm *.jpg
```

careful to be in the asf_floorplan_interpreter parent directory.
