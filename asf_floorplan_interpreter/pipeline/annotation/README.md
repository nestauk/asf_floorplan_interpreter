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

## Rooms:

Annotate bounding boxes for each:

```
prodigy image.manual room_type_dataset asf_floorplan_interpreter/pipeline/annotation/floorplans.jsonl --loader jsonl --label BEDROOM,KITCHEN,BATHROOM,LIVING,STAIRCASE,OTHER --no-fetch

```

```
prodigy db-out room_type_dataset > asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/room_type.jsonl

```

Use this to train a model.

## Windows and doors:

Since we have a model for this and its a long annotation task to tag every one, we can use the model to help us label.

Manually correct Adeola's original model for doors and windows?

```
prodigy classify-images window_doors_dataset asf_floorplan_interpreter/pipeline/annotation/floorplans.jsonl WINDOW,DOOR -F asf_floorplan_interpreter/pipeline/annotation/floorplan_recipe.py

```

```
prodigy db-out window_doors_dataset > asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/window_door.jsonl
```

## Everything model:

This will use the pretrained models to help label ROOM, WINDOW, DOOR.

```
prodigy classify-everything-images room_window_door_dataset asf_floorplan_interpreter/pipeline/annotation/floorplans.jsonl WINDOW,DOOR,ROOM,OTHER_ROOM,OTHER_DOOR -F asf_floorplan_interpreter/pipeline/annotation/floorplan_recipe.py

```

```
prodigy db-out room_window_door_dataset > asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/room_window_door.jsonl
```

## Room type model:

Use the pretrained model to identify rooms, then annotation which room type they are

```
prodigy room-type room_type_dataset_2 asf_floorplan_interpreter/pipeline/annotation/floorplans.jsonl BEDROOM,KITCHEN,BATHROOM,LIVING,STAIRCASE,OTHER -F asf_floorplan_interpreter/pipeline/annotation/floorplan_recipe.py

```

```
prodigy db-out room_type_dataset_2 > asf_floorplan_interpreter/pipeline/annotation/prodigy_labelled/room_type_dataset_2.jsonl
```

## Clean up

YOLO reads and saves each floorplan jpg, so you will end up with the directory being full of these. To clean them up run:

```
rm *.jpg
```

careful to be in the asf_floorplan_interpreter parent directory.
