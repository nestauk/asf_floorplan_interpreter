## Train a model to recognise doors and windows

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

## :file_folder: Our own labelled data

We created two datasets of labelled data.

1. Labelling rooms.
2. Labelleding doors, windows and staircases.

## :muscle: Training

Train: YOLOV8n (smallest) for instance segmentation (docs) using metaflow:

```
cd asf_floorplan_interpreter/pipeline/
python train_yolo.py --package-suffixes=.txt,.yaml,.jpg --datastore=s3 run --config_file configs/CONFIG_NAME.yaml
```

This will train the model and output the best model in [this S3 location](https://s3.console.aws.amazon.com/s3/buckets/asf-floorplan-interpreter?region=eu-west-2&prefix=window_door_config/&showversions=false).

The metaflow files are in this location: https://s3.console.aws.amazon.com/s3/buckets/open-jobs-lake?prefix=metaflow/FloorPlanYolo/&region=eu-west-1

### 📓 Configs

Note that for Batch to work, the configs need to be in the same folder as `train_yolo.py` - hence why we store them in the `configs/` folder rather than elsewhere in this repo.

To train each model a pair of configs are needed; the main config (e.g. `room_config.yaml`) and the config neccessary for training using YOLO (`yolo_room_config.yaml`). The latter is referenced in the former, so when you run `train_yolo.py` you only need the first config in the argument.

The main configs to choose from are:

1. `configs/roboflow_config.yaml`: To train a model to identify doors, windows and rooms using the Roboflow dataset.
2. `configs/room_config.yaml`: To train a model that will identify rooms using our labelled dataset.
3. `configs/window_door_config.yaml`: To train a model that will identify windows, doors and staircases using our labelled dataset.

There are also test versions of some of these which will run more quickly `configs/roboflow_test_config.yaml` and `configs/room_test_config.yaml`.

e.g.

```
python train_yolo.py --package-suffixes=.txt,.yaml,.jpg --datastore=s3 run --config_file configs/roboflow_test_config.yaml
```

The Yolo configs are in a specific format and include two variables which need setting according to your task. For example in the roboflow config:

```
path: "data/roboflow_data/"
train: images/train
val: images/val
test: images/test
save_dir: "roboflow_save_dir/"

nc: 6
names: ["DOOR", "DOUBLE DOOR", "FOLDING DOOR", "ROOM", "SLIDING DOOR", "WINDOW"]

```

we provide the paths to the training, test and validation sets (i.e. `data/roboflow_data/images/train` etc), as well as telling YOLO we want to train 6 class (`nc` = number of classes) and the names of these 6 classes (`names`). Yolo will then map each class to a number. The names of the classes should map to what's in the training datasets - although the classes you train can be a subset of all the classes labelled.
