# 🧮 Evaluation

Each model is evaluated on its validation dataset, and metrics are stored in the different model output folders.

We also run a full pipeline evaluation using our evaluation dataset - a manually coded sample of floorplans with how many floors, windows, rooms and each room type they have. This dataset is stored in `s3://asf-floorplan-interpreter/data/annotation/evaluation/Econest_test_set_floorplans_211123.csv`.

## Model test metrics

As of 21st November 2023 the models perform with the following metrics.

| Model                           | Precision (B) | Recall (B) | Precision (M) | Recall (m) |
| ------------------------------- | ------------- | ---------- | ------------- | ---------- |
| `window_door_config_yolov8m_wd` | 0.87          | 0.90       | 0.77          | 0.79       |
| `room_config_yolov8m`           | 0.83          | 0.92       | 0.88          | 0.87       |
| `staircase_config_yolov8m`      | 0.81          | 0.82       | 0.78          | 0.80       |
| `room_type_config_yolov8m`      | 0.60          | 0.80       | 0.61          | 0.80       |

Where the `B` is for the metrics of detection and `M` for segmentation.

Note: In the `results.csv` file created when a model is trained, these test metrics have been taken from the epoch with the maximum "metrics/mAP50-95(B)" value.

## Evalution results

To run the evaluation, which will compare the models' predictions with the ground truth run:

```
python asf_floorplan_interpreter/pipeline/evaluation/evaluate.py

```

The results are stored in `s3://asf-floorplan-interpreter/models/evaluation/`.

For the `20231122/` evaluation we calculate the mean squared error between the model predicted results and the ground truth for each entity type (e.g. room, window, kitchen). We also calculate this for a rule-based estimate of these numbers (e.g. every house has one kitchen).

| Entity type      | Rule-based RMSE | Model prediction RMSE | Number of floorplans in calculation |
| ---------------- | --------------- | --------------------- | ----------------------------------- |
| DOOR             | 5.62            | 2.71                  | 20                                  |
| WINDOW           | 4.83            | 2.17                  | 20                                  |
| BEDROOM          | 0.54            | 1.05                  | 76                                  |
| KITCHEN          | 0.11            | 0.79                  | 77                                  |
| LIVING           | 0.95            | 1.01                  | 77                                  |
| RESTROOM         | 0.78            | 0.73                  | 75                                  |
| ROOM             | 5.30            | 2.24                  | 77                                  |
| ROOM_TYPE_SUMMED | -               | 2.67                  | 77                                  |
| GARAGE           | -               | 0.65                  | 56                                  |
| OTHER            | -               | 2.17                  | 57                                  |

- Our room model does a better job at counting rooms than the sum of all the room types (`ROOM_TYPE_SUMMED`).
- We found that we can get a better kitchen result if we count 1 kitchen at a minimum per floorplan. This blended rule + model approach gives a RMSE of 0.69. Thus, we hard code this into our prediction function.
