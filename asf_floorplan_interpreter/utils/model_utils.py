from asf_floorplan_interpreter import BUCKET_NAME, PROJECT_DIR
from sklearn.metrics import mean_squared_error
from ultralytics import YOLO

import os


def load_model(directory):
    """Function to load YOLO model trained on floor plans annotated with polygons identifying rooms"""
    return YOLO(directory)


def load_model_s3(model_name):
    """Function to load YOLO model from S3 trained on floor plans annotated with polygons identifying rooms"""
    os.system(
        f"aws s3 cp s3://asf-floorplan-interpreter/models/{model_name}/weights/best.pt outputs/models/{model_name}/weights/best.pt"
    )
    model = load_model(os.path.join("outputs/models", model_name, "weights/best.pt"))

    return model


def evaluate_model(true_labels, pred_labels):
    """Calculate mean squared error bewtween sets of labels"""
    return mean_squared_error(true_labels, pred_labels)


def yolo_2_segments(results):
    """
    Convert the YOLO model prediction output from bounding boxes to segmentation points format.
    Needed for labelling in Prodigy or for use in predict_floorplan.py

    The (x, y) coordinates of the bounding box represent the center of the box,
    while in the segmentation format, the coordinates represent the corners of the polygon.
    See https://github.com/ultralytics/ultralytics/issues/3592.
    """
    segments = []
    for (x, y, w, h), label, conf in zip(
        results[0].boxes.xywh, results[0].boxes.cls, results[0].boxes.conf.numpy()
    ):
        x_min = x.item() - (w.item() / 2)
        y_min = y.item() - (h.item() / 2)
        x_max = x.item() + (w.item() / 2)
        y_max = y.item() + (h.item() / 2)
        segment = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        segments.append(
            {
                "label": results[0].names[label.item()],
                "points": segment,
                "type": "polygon",
                "confidence": round(conf, 3),
            }
        )
    return segments
