from asf_floorplan_interpreter import BUCKET_NAME, PROJECT_DIR
from sklearn.metrics import mean_squared_error
from ultralytics import YOLO


def load_model(directory):
    """Function to load YOLO model trained on floor plans annotated with polygons identifying rooms"""
    return YOLO(directory)


def evaluate_model(true_labels, pred_labels):
    """Calculate mean squared error bewtween sets of labels"""
    return mean_squared_error(true_labels, pred_labels)
