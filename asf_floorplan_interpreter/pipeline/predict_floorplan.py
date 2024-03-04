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

import numpy as np


class FloorplanPredictor(object):
    """
    Predict segments and the counts of each label type in a floorplan using pretrained models.
    Arguments:
        labels (list): which labels to predict from ["ROOM", "WINDOW", "DOOR",
    "STAIRCASE", "KITCHEN", "LIVING", "RESTROOM", "BEDROOM", "GARAGE", "OTHER"] this list will inform
        which models are needed.
        config_name (str): the name of the config file to use. This file will give the names of the
            models to use.

    Methods:
        load: Load the neccessary models
        predict_labels: Input an image URL or local pathway and output where the label segments are,
            as well as counts of each label.
        plot: Plot the label segments on the original floorplan image.

    """

    def __init__(
        self,
        labels_to_predict=[
            "ROOM",
            "WINDOW",
            "DOOR",
            "DOUBLE_DOOR",
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
            PROJECT_DIR, "asf_floorplan_interpreter/config/", config_name + ".yaml"
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
            local_dir = os.path.join(PROJECT_DIR, "outputs/models/")
            if "ROOM" in self.labels_to_predict:
                self.room_model = load_model(
                    os.path.join(local_dir, f"{self.room_model_name}/weights/best.pt")
                )
            if "STAIRCASE" in self.labels_to_predict:
                self.staircase_model = load_model(
                    os.path.join(
                        local_dir, f"{self.staircase_model_name}/weights/best.pt"
                    )
                )
            if any(
                [
                    label in self.labels_to_predict
                    for label in ["DOOR", "DOUBLE_DOOR", "WINDOW"]
                ]
            ):
                self.window_door_model = load_model(
                    os.path.join(
                        local_dir, f"{self.window_door_model_name}/weights/best.pt"
                    )
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
                    os.path.join(
                        local_dir, f"{self.room_type_model_name}/weights/best.pt"
                    )
                )
        else:
            if "ROOM" in self.labels_to_predict:
                self.room_model = load_model_s3(self.room_model_name)
            if "STAIRCASE" in self.labels_to_predict:
                self.staircase_model = load_model_s3(self.staircase_model_name)
            if any(
                [
                    label in self.labels_to_predict
                    for label in ["DOOR", "DOUBLE_DOOR", "WINDOW"]
                ]
            ):
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

    def predict_labels(
        self,
        image_url,
        correct_kitchen=True,
        correct_staircase=True,
        conf_threshold=0,
    ):
        """
        Predict label segments and a counts of labels using the loaded models for a floorplan.

        Arguments:
            image_url (str): A URL or a local directory of your floorplan image.
            correct_kitchen (bool): Whether to at a minimum predict 1 kitchen per floorplan (True) or not (False).
            conf_threshold (float): The prediction confidence threshold for outputted labels.

        Outputs:
            labels (list): A list of the predicted image segments for each label, this is in the form
                [{'label': 'DOOR', 'points': [[525.348, 319.445], ...], 'type': 'polygon', 'confidence': 0.9}, ...]
            label_counts (dict): A summary of the counts of all labels found for this image. For example,
                {'DOOR': 12, 'WINDOW': 10, 'LIVING': 2, 'KITCHEN': 2, 'BEDROOM': 3, 'RESTROOM': 1}
        """

        if os.path.exists(os.path.join(PROJECT_DIR, image_url)):
            image_url = os.path.join(PROJECT_DIR, image_url)
        else:
            print(
                f"{os.path.join(PROJECT_DIR, image_url)} doesn't exist locally, assuming it is a url"
            )

        labels = []
        if "ROOM" in self.labels_to_predict:
            results = self.room_model(image_url, save=False, verbose=False)
            labels += yolo_2_segments(results)
        if "STAIRCASE" in self.labels_to_predict:
            results = self.staircase_model(image_url, save=False, verbose=False)
            labels += yolo_2_segments(results)
        if any(
            [
                label in self.labels_to_predict
                for label in ["DOOR", "DOUBLE_DOOR", "WINDOW"]
            ]
        ):
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

        # Remove any labels you aren't asking for and only if the prediction probability is over a threshold
        labels = [
            label
            for label in labels
            if (
                (label["label"] in self.labels_to_predict)
                and (label["confidence"] >= conf_threshold)
            )
        ]

        # Get label counts
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label["label"]] += 1

        # We found the results are best if the output says at least one kitchen per floorplan
        if correct_kitchen:
            label_counts["KITCHEN"] = 1

        if correct_staircase:
            if "STAIRCASE" in label_counts:
                label_counts["STAIRCASE"] = np.ceil(label_counts["STAIRCASE"] / 2)

        if ("DOOR" in label_counts) or ("DOUBLE_DOOR" in label_counts):
            label_counts["ALL_DOORS"] = label_counts.get("DOOR", 0) + label_counts.get(
                "DOUBLE_DOOR", 0
            )

        return labels, dict(label_counts)

    def plot(self, image_url, labels, output_name, plot_label=True):
        """
        Plot your predicted labels on the floorplan

        Arguments:
            image_url (str): A URL or a local directory of your floorplan image.
            labels (list): The list of labels, in the format returned from running predict_labels.
            output_name (str): The directory for the outputted image.
            plot_label (bool): Whether to plot the text of each bounding box label or not.
        """
        if os.path.exists(os.path.join(PROJECT_DIR, image_url)):
            image_url = os.path.join(PROJECT_DIR, image_url)
        else:
            print(
                f"{os.path.join(PROJECT_DIR, image_url)} doesn't exist locally, assuming it is a url"
            )

        visual_image = load_image(image_url)
        visual_image = overlay_boundaries_plot(
            visual_image, labels, show=False, plot_label=plot_label
        )

        visual_image.savefig(
            os.path.join(PROJECT_DIR, output_name), bbox_inches="tight"
        )
