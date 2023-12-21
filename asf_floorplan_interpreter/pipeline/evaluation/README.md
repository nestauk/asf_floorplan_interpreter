# 🧮 Evaluation

Each model is evaluated on its validation dataset, and the results are stored in the different model output folders (e.g. `models/room_config_yolov8m/results.csv`).

We also run a full pipeline evaluation using our evaluation dataset - a manually coded sample of floorplans with how many floors, windows, rooms and each room type they have. This dataset is stored in the S3 location `data/annotation/evaluation/Econest_test_set_floorplans_211123.csv`.

## Model test metrics

<!-- As of 23rd November 2023 the models perform with the following metrics.

| Model                           | Precision (B) | Recall (B) | Precision (M) | Recall (m) |
| ------------------------------- | ------------- | ---------- | ------------- | ---------- |
| `window_door_config_yolov8m_wd` | 0.87          | 0.90       | 0.77          | 0.79       |
| `room_config_yolov8m`           | 0.87          | 0.87       | 0.88          | 0.88       |
| `staircase_config_yolov8m`      | 0.77          | 0.87       | 0.76          | 0.85       |
| `room_type_config_yolov8m`      | 0.65          | 0.84       | 0.65          | 0.84       | -->

As of 21st December 2023 the models perform with the following metrics.

| Model                               | Precision (B) | Recall (B) | Precision (M) | Recall (m) |
| ----------------------------------- | ------------- | ---------- | ------------- | ---------- |
| `window_door_config_yolov8m_wd`     | 0.87          | 0.90       | 0.77          | 0.79       |
| `window_door_types_roboflow_config` | 0.70          | 0.74       | 0.62          | 0.68       |
| `room_config_yolov8m`               | 0.87          | 0.87       | 0.88          | 0.88       |
| `staircase_config_yolov8m`          | 0.94          | 0.90       | 0.92          | 0.89       |
| `room_type_config_yolov8m`          | 0.57          | 0.79       | 0.57          | 0.73       |
| `room_type_bw_config_yolov8m`       | 0.71          | 0.77       | 0.72          | 0.78       |

Where the `B` is for the metrics of detection and `M` for segmentation.

Note: In the `results.csv` file created when a model is trained, these test metrics have been taken from the epoch with the maximum "metrics/mAP50-95(B)" value.

## Evalution results

To run the evaluation, which will compare the models' predictions with the ground truth run:

```
python asf_floorplan_interpreter/pipeline/evaluation/evaluate.py
```

The results are stored in `s3://asf-floorplan-interpreter/models/evaluation/`.

For the `20231221/` evaluation we calculate the root mean squared error (RMSE) between the model predicted results and the ground truth for each entity type (e.g. room, window, kitchen). We also calculate this for a rule-based estimate of these numbers (e.g. every house has one kitchen).

| Entity type      | Rule-based RMSE | Model prediction RMSE | Model better? | Number of floorplans in calculation | Average number per floorplan |
| ---------------- | --------------- | --------------------- | ------------- | ----------------------------------- | ---------------------------- |
| ALL_DOORS        | 7.27            | 2.5                   | ✅            | 28                                  | 12.9                         |
| DOOR             | -               | 1.18                  | ✅            | 42                                  | 10.0                         |
| DOUBLE_DOOR      | -               | 1.60                  | ✅            | 28                                  | 2.1                          |
| WINDOW           | 4.71            | 1.90                  | ✅            | 42                                  | 10.1                         |
| BEDROOM          | 0.61            | 1.08                  | ❌            | 82                                  | 2.7                          |
| KITCHEN          | 0.11            | 0.63                  | ❌            | 83                                  | 1.0                          |
| LIVING           | 0.98            | 1.06                  | Comparable    | 83                                  | 1.7                          |
| RESTROOM         | 0.71            | 0.94                  | ❌            | 81                                  | 1.6                          |
| ROOM             | 5.22            | 2.01                  | ✅            | 83                                  | 11.5                         |
| ROOM_TYPE_SUMMED | 5.22            | 2.73                  | ✅            | 83                                  | 11.5                         |
| GARAGE           | -               | 0.53                  | NA            | 61                                  | 0.3                          |
| OTHER            | -               | 2.05                  | NA            | 61                                  | 4.4                          |
| STAIRCASE        | 0.66            | 0.47                  | ✅            | 82                                  | 1.3                          |

Notes:

- Our room model does a better job at counting rooms than the sum of all the room types (`ROOM_TYPE_SUMMED`).
- We found that we can get a better kitchen result if we count 1 kitchen per floorplan.
- We found that the number of staircase predictions a model predicts should be divide by 2 and the ceiling rounded value returned, this will give an accurate number of staircases (avoiding deduplications).
