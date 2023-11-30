# 🧮 Evaluation

Each model is evaluated on its validation dataset, and the results are stored in the different model output folders (e.g. `models/room_config_yolov8m/results.csv`).

We also run a full pipeline evaluation using our evaluation dataset - a manually coded sample of floorplans with how many floors, windows, rooms and each room type they have. This dataset is stored in the S3 location `data/annotation/evaluation/Econest_test_set_floorplans_211123.csv`.

## Model test metrics

As of 23rd November 2023 the models perform with the following metrics.

| Model                           | Precision (B) | Recall (B) | Precision (M) | Recall (m) |
| ------------------------------- | ------------- | ---------- | ------------- | ---------- |
| `window_door_config_yolov8m_wd` | 0.87          | 0.90       | 0.77          | 0.79       |
| `room_config_yolov8m`           | 0.87          | 0.87       | 0.88          | 0.88       |
| `staircase_config_yolov8m`      | 0.77          | 0.87       | 0.76          | 0.85       |
| `room_type_config_yolov8m`      | 0.65          | 0.84       | 0.65          | 0.84       |

Where the `B` is for the metrics of detection and `M` for segmentation.

Note: In the `results.csv` file created when a model is trained, these test metrics have been taken from the epoch with the maximum "metrics/mAP50-95(B)" value.

## Evalution results

To run the evaluation, which will compare the models' predictions with the ground truth run:

```
python asf_floorplan_interpreter/pipeline/evaluation/evaluate.py
```

The results are stored in `s3://asf-floorplan-interpreter/models/evaluation/`.

For the `20231123/` evaluation we calculate the root mean squared error (RMSE) between the model predicted results and the ground truth for each entity type (e.g. room, window, kitchen). We also calculate this for a rule-based estimate of these numbers (e.g. every house has one kitchen).

| Entity type      | Rule-based RMSE | Model prediction RMSE | Model better? | Number of floorplans in calculation |
| ---------------- | --------------- | --------------------- | ------------- | ----------------------------------- |
| DOOR             | 5.62            | 2.71                  | ✅            | 20                                  |
| WINDOW           | 4.83            | 2.17                  | ✅            | 20                                  |
| BEDROOM          | 0.54            | 1.24                  | ❌            | 76                                  |
| KITCHEN          | 0.11            | 0.88                  | ❌            | 77                                  |
| LIVING           | 0.95            | 1.01                  | Comparable    | 77                                  |
| RESTROOM         | 0.78            | 0.66                  | ✅            | 75                                  |
| ROOM             | 5.30            | 2.02                  | ✅            | 77                                  |
| ROOM_TYPE_SUMMED | -               | 2.38                  | ✅            | 77                                  |
| GARAGE           | -               | 0.52                  | NA            | 56                                  |
| OTHER            | -               | 1.76                  | NA            | 57                                  |

Notes:

- Our room model does a better job at counting rooms than the sum of all the room types (`ROOM_TYPE_SUMMED`).
- We found that we can get a better kitchen result if we count 1 kitchen at a minimum per floorplan. This blended rule + model approach gives a RMSE of 0.76. Thus, we hard code this into our prediction function by default.
