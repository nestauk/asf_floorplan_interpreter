import os
import yaml
from collections import Counter, defaultdict
from asf_floorplan_interpreter import PROJECT_DIR
from asf_floorplan_interpreter.utils.model_utils import load_model, yolo_2_segments
from asf_floorplan_interpreter.utils.visualise_image import (
    load_image,
    overlay_boundaries_plot,
)

from asf_floorplan_interpreter.utils.model_utils import load_model_s3, load_model


class FloorplanPredictor(object):
    """
    Predict segments in a floorplan using pretrained models.
    Arguments:
            labels (list): which labels to predict from ["ROOM", "WINDOW", "DOOR",
    "STAIRCASE", "KITCHEN", "LIVING", "RESTROOM", "BEDROOM", "GARAGE", "OTHER"]
    """

    def __init__(
        self,
        labels_to_predict=[
            "ROOM",
            "WINDOW",
            "DOOR",
            "STAIRCASE",
            "KITCHEN",
            "LIVING",
            "RESTROOM",
            "BEDROOM",
            "GARAGE",
            "OTHER",
        ],
        config_name="base",
    ):
        self.labels_to_predict = labels_to_predict

        # Set variables from the config file
        config_path = os.path.join(
            "asf_floorplan_interpreter/config/", config_name + ".yaml"
        )
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.room_model_name = config["room_model_name"]
        self.window_door_model_name = config["window_door_model_name"]
        self.staircase_model_name = config["staircase_model_name"]
        self.room_type_model_name = config["room_type_model_name"]

    def load(self, local=False):
        """
        Only load the models you need
        """
        if local:
            if "ROOM" in self.labels_to_predict:
                self.room_model = load_model(
                    f"outputs/models/{self.room_model_name}/weights/best.pt"
                )
            if "STAIRCASE" in self.labels_to_predict:
                self.staircase_model = load_model(
                    f"outputs/models/{self.staircase_model_name}/weights/best.pt"
                )
            if any([label in self.labels_to_predict for label in ["DOOR", "WINDOW"]]):
                self.window_door_model = load_model(
                    f"outputs/models/{self.window_door_model_name}/weights/best.pt"
                )
            if any(
                [
                    label in self.labels_to_predict
                    for label in [
                        "KITCHEN",
                        "LIVING",
                        "RESTROOM",
                        "BEDROOM",
                        "GARAGE",
                        "OTHER",
                    ]
                ]
            ):
                self.room_type_model = load_model(
                    f"outputs/models/{self.room_type_model_name}/weights/best.pt"
                )
        else:
            if "ROOM" in self.labels_to_predict:
                self.room_model = load_model_s3(self.room_model_name)
            if "STAIRCASE" in self.labels_to_predict:
                self.staircase_model = load_model_s3(self.staircase_model_name)
            if any([label in self.labels_to_predict for label in ["DOOR", "WINDOW"]]):
                self.window_door_model = load_model_s3(self.window_door_model_name)
            if any(
                [
                    label in self.labels_to_predict
                    for label in [
                        "KITCHEN",
                        "LIVING",
                        "RESTROOM",
                        "BEDROOM",
                        "GARAGE",
                        "OTHER",
                    ]
                ]
            ):
                self.room_type_model = load_model_s3(self.room_type_model_name)

    def predict_labels(self, image_url, correct_kitchen=True):
        labels = []
        if "ROOM" in self.labels_to_predict:
            results = self.room_model(image_url, save=False, verbose=False)
            labels += yolo_2_segments(results)
        if "STAIRCASE" in self.labels_to_predict:
            results = self.staircase_model(image_url, save=False, verbose=False)
            labels += yolo_2_segments(results)
        if any([label in self.labels_to_predict for label in ["DOOR", "WINDOW"]]):
            results = self.window_door_model(image_url, save=False, verbose=False)
            labels += yolo_2_segments(results)
        if any(
            [
                label in self.labels_to_predict
                for label in [
                    "KITCHEN",
                    "LIVING",
                    "RESTROOM",
                    "BEDROOM",
                    "GARAGE",
                    "OTHER",
                ]
            ]
        ):
            results = self.room_type_model(image_url, save=False, verbose=False)
            labels += yolo_2_segments(results)

        # Remove any labels you aren't asking for
        labels = [label for label in labels if label["label"] in self.labels_to_predict]

        # Get label counts
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label["label"]] += 1

        # We found the results are best if the output says at least one kitchen per floorplan
        if correct_kitchen:
            if "KITCHEN" in label_counts:
                label_counts["KITCHEN"] = max(label_counts["KITCHEN"], 1)

        return labels, dict(label_counts)

    def plot(self, image_url, labels, output_name, plot_label=True):
        visual_image = load_image(image_url)
        visual_image = overlay_boundaries_plot(
            visual_image, labels, show=False, plot_label=plot_label
        )

        visual_image.savefig(output_name)
