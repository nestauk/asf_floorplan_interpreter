import os
from collections import Counter
from asf_floorplan_interpreter import PROJECT_DIR
from asf_floorplan_interpreter.utils.model_utils import load_model


def create_results_dict(img, model):
    """Predict a given image and output a dictionary of counts for each class

    Args:
        img (str): Image directory
        model (pytorch model): chosen model for predictions
    """

    results = model(img, save=True)
    class_names = results[0].names
    class_pred_count = dict(Counter(results[0].boxes.cls.tolist()))

    return {class_names[k]: v for k, v in class_pred_count.items()}


def predict_image(directory, img):
    "Collects model and applies to chosen images"
    model = load_model(
        (PROJECT_DIR / f"{directory}")
    )  # e.g. directory = "outputs/models/rooms-model/best.pt"

    return create_results_dict(img, model)
