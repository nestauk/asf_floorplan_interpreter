from asf_floorplan_interpreter.getters.get_data import (
    save_to_s3,
    load_s3_data,
    get_s3_data_paths,
)
from asf_floorplan_interpreter import BUCKET_NAME, logger
import os

from asf_floorplan_interpreter.utils.model_utils import load_model_s3
from asf_floorplan_interpreter.utils.config_utils import read_base_config
from asf_floorplan_interpreter.pipeline.predict_floorplan import (
    FloorplanPredictor,
)
import torch
from sklearn.metrics import mean_squared_error

import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
from datetime import datetime


def load_process_eval_data(eval_data_file, all_training_floorplans):
    """
    Load the evaluation dataset, remove any rows which were used in training.
    Create dictionaries of the ground truth results and the rule-based predictions for each floorplan.
    """

    # The evaluation data
    eval_data = load_s3_data(BUCKET_NAME, eval_data_file)
    eval_data = eval_data[
        pd.notnull(eval_data["total_rooms"]) & eval_data["total_rooms"] != 0
    ]
    eval_floorplan_urls = eval_data["floorplan_url"].tolist()
    eval_floorplan_names = [f.split("/")[-1].split(".")[0] for f in eval_floorplan_urls]
    eval_data["floorplan_name"] = eval_floorplan_names

    # Only evaluate on data never used in training
    eval_data_fresh = eval_data[
        ~eval_data["floorplan_name"].isin(all_training_floorplans)
    ]
    logger.info(
        f"Evaluating on {len(eval_data_fresh)} floorplans from {len(eval_data)}"
    )

    # Combine both the other categories in our ground truth labels
    eval_data_fresh["other_both"] = eval_data_fresh.apply(
        lambda x: int(x["other_rooms"]) + int(x["halls_and_storage"])
        if (pd.notnull(x["other_rooms"]) & pd.notnull(x["halls_and_storage"]))
        else None,
        axis=1,
    )

    # Combine both the door categories in the rule-based predictions
    eval_data_fresh["rulebased_total_doors"] = eval_data_fresh.apply(
        lambda x: int(x["doors_internal"]) + int(x["doors_external"])
        if x["doors_internal"] != "#VALUE!"
        else None,
        axis=1,
    )

    eval_dict = (
        eval_data_fresh[
            [
                "floorplan_url",
                "# of doors",
                "# of windows",
                "bedrooms.1",
                "kitchen.1",
                "living_room.1",
                "bathrooms.1",
                "garage",
                "other_both",
                "total_rooms",
            ]
        ]
        .set_index("floorplan_url")
        .rename(
            columns={
                "# of doors": "DOOR",
                "# of windows": "WINDOW",
                "bedrooms.1": "BEDROOM",
                "kitchen.1": "KITCHEN",
                "living_room.1": "LIVING",
                "bathrooms.1": "RESTROOM",
                "garage": "GARAGE",
                "other_both": "OTHER",
                "total_rooms": "ROOM",
            }
        )
        .to_dict(orient="index")
    )

    rule_based_pred_dict = (
        eval_data_fresh[
            [
                "floorplan_url",
                "bedrooms",
                "bathrooms",
                "stairway",
                "living_room",
                "kitchen",
                "rooms",
                "windows",
                "rulebased_total_doors",
            ]
        ]
        .set_index("floorplan_url")
        .rename(
            columns={
                "bedrooms": "BEDROOM",
                "bathrooms": "RESTROOM",
                "stairway": "STAIRCASE",
                "living_room": "LIVING",
                "kitchen": "KITCHEN",
                "rooms": "ROOM",
                "windows": "WINDOW",
                "rulebased_total_doors": "DOOR",
            }
        )
        .to_dict(orient="index")
    )

    return eval_dict, rule_based_pred_dict


if __name__ == "__main__":
    # Set variables from the config file
    config = read_base_config()

    room_model_name = config["room_model_name"]
    window_door_model_name = config["window_door_model_name"]
    staircase_model_name = config["staircase_model_name"]
    room_type_model_name = config["room_type_model_name"]
    eval_data_file = config["eval_data_file"]

    floorplan_pred = FloorplanPredictor(
        labels_to_predict=[
            "ROOM",
            "WINDOW",
            "DOOR",
            "KITCHEN",
            "LIVING",
            "RESTROOM",
            "BEDROOM",
            "GARAGE",
            "OTHER",
        ]
    )
    floorplan_pred.load()

    today = datetime.now().strftime("%Y%m%d")

    output_folder = f"models/evaluation/{today}/"

    # Find all the floorplans used in training any of these models
    all_training_floorplans = []
    for model_name in [
        room_model_name,
        window_door_model_name,
        staircase_model_name,
        room_type_model_name,
    ]:
        path_files = get_s3_data_paths(BUCKET_NAME, os.path.join("models", model_name))
        config_name = [p for p in path_files if "config.yaml" in p][0]
        model_config = load_s3_data(BUCKET_NAME, config_name)
        # Find the training data
        training_data_path = os.path.join(model_config["path"], model_config["train"])
        training_image_paths = get_s3_data_paths(BUCKET_NAME, training_data_path)
        training_images = [t.split("/")[-1].split(".")[0] for t in training_image_paths]
        all_training_floorplans += training_images

    # Load and process the evaluation data - removing any training floorplans
    eval_dict, rule_based_pred_dict = load_process_eval_data(
        eval_data_file, all_training_floorplans
    )

    logger.info("Predict numbers of rooms etc for each of the evaluation floorplans")
    pred_dict = {}
    for floorplan_url in tqdm(eval_dict.keys()):
        _, results = floorplan_pred.predict_labels(floorplan_url, correct_kitchen=False)
        results["SUM_ROOM_TYPES"] = sum(
            [
                v
                for k, v in results.items()
                if k in ["KITCHEN", "LIVING", "RESTROOM", "BEDROOM", "GARAGE", "OTHER"]
            ]
        )
        pred_dict[floorplan_url] = results

    # Everytime you predict from an S3 image, the image is downloaded.. so clean these up
    os.system("rm *.jpg")
    os.system("rm *.png")

    per_label_results = defaultdict(list)
    floorplan_urls = []
    for floorplan_url, truth_labels in eval_dict.items():
        for label_key, truth_value in truth_labels.items():
            per_label_results[label_key].append(
                [
                    truth_value,
                    pred_dict[floorplan_url].get(label_key, 0),
                    rule_based_pred_dict[floorplan_url].get(label_key, None),
                ]
            )
            if label_key == "ROOM":
                # Also add the summed room types
                per_label_results["SUM_ROOM_TYPES"].append(
                    [
                        truth_value,
                        pred_dict[floorplan_url].get("SUM_ROOM_TYPES", 0),
                        rule_based_pred_dict[floorplan_url].get(label_key, None),
                    ]
                )
        floorplan_urls.append(floorplan_url)

    all_results = {
        "results": per_label_results,
        "floorplan_urls": floorplan_urls,
        "order": ["Truth", "Model prediction", "Rule-based prediction"],
    }

    save_to_s3(
        BUCKET_NAME, all_results, os.path.join(output_folder, "all_results.json")
    )

    mse_model_results = {}
    mse_rule_based_results = {}
    for label_key, results in all_results["results"].items():
        model_truth = []
        rulebased_truth = []
        model_pred = []
        rulebased_pred = []
        for t, m, e in results:
            if pd.notnull(t):
                if m != None:
                    model_truth.append(t)
                    model_pred.append(m)
                if e != None:
                    if e != "\\N":
                        rulebased_truth.append(t)
                        rulebased_pred.append(int(e))
        if model_pred:
            mse_model_results[label_key] = {
                "n": len(model_truth),
                "mse": mean_squared_error(model_truth, model_pred),
                "rmse": mean_squared_error(model_truth, model_pred, squared=False),
            }
            if label_key == "KITCHEN":
                corrected_model_pred = [
                    max(1, v) for v in model_pred
                ]  # If there are 0 kitchens, bump it to 1
                mse_model_results["KITCHEN_correct"] = {
                    "n": len(model_truth),
                    "mse": mean_squared_error(model_truth, corrected_model_pred),
                    "rmse": mean_squared_error(
                        model_truth, corrected_model_pred, squared=False
                    ),
                }
        if rulebased_pred:
            mse_rule_based_results[label_key] = {
                "n": len(rulebased_truth),
                "mse": mean_squared_error(rulebased_truth, rulebased_pred),
                "rmse": mean_squared_error(
                    rulebased_truth, rulebased_pred, squared=False
                ),
            }
    mse_results = {
        "rulebased_prediction_mse": mse_rule_based_results,
        "model_prediction_mse": mse_model_results,
    }

    save_to_s3(
        BUCKET_NAME, mse_results, os.path.join(output_folder, "mse_results.json")
    )
