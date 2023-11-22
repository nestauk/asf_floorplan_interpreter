import os
from collections import Counter
from asf_floorplan_interpreter import PROJECT_DIR
from asf_floorplan_interpreter.utils.model_utils import load_model


def create_results_dict(img, model, save=True):
    """Predict a given image and output a dictionary of counts for each class

    Args:
        img (str): Image directory
        model (pytorch model): chosen model for predictions
    """

    results = model(img, save=save, verbose=False)
    class_names = results[0].names
    class_pred_count = dict(Counter(results[0].boxes.cls.tolist()))

    return {class_names[k]: v for k, v in class_pred_count.items()}


def predict_image(local_directory, img):
    "Collects model from a local location and applies to chosen images"
    model = load_model(
        (PROJECT_DIR / f"{local_directory}")
    )  # e.g. directory = "outputs/models/rooms-model/best.pt"

    return create_results_dict(img, model)


def full_output(
    img, room_model, window_door_model, room_type_model, staircase_model=None
):
    results = {}
    results.update(create_results_dict(img, room_model, save=False))
    results.update(create_results_dict(img, window_door_model, save=False))
    results.update(create_results_dict(img, room_type_model, save=False))
    if staircase_model:
        results.update(create_results_dict(img, staircase_model, save=False))
    return results
