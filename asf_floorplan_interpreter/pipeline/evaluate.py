from asf_floorplan_interpreter.getters.get_data import (
    save_to_s3,
    load_s3_data,
    get_s3_data_paths,
)
from asf_floorplan_interpreter import BUCKET_NAME, logger
import os

from asf_floorplan_interpreter.utils.model_utils import load_model
from asf_floorplan_interpreter.pipeline.predict_floorplan import (
    create_results_dict,
    full_output,
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
    Create dictionaries of the ground truth results and the econest predictions for each floorplan.
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

    # Combine both the door categories in EcoNests predictions
    eval_data_fresh["eco_total_doors"] = eval_data_fresh.apply(
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

    econest_pred_dict = (
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
                "eco_total_doors",
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
                "eco_total_doors": "DOOR",
            }
        )
        .to_dict(orient="index")
    )

    return eval_dict, econest_pred_dict


if __name__ == "__main__":
    eval_data_file = "data/annotation/evaluation/Econest_test_set_floorplans_211123.csv"
    room_model_name = "room_config_yolov8m"
    window_door_model_name = "window_door_config_yolov8m_wd"
    staircase_model_name = "staircase_config_yolov8m"
    room_type_model_name = "room_type_config_yolov8m"

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
    eval_dict, econest_pred_dict = load_process_eval_data(
        eval_data_file, all_training_floorplans
    )

    logger.info("Downloading all the models from S3 and local")
    for model_name in [
        room_model_name,
        window_door_model_name,
        staircase_model_name,
        room_type_model_name,
    ]:
        os.system(
            f"aws s3 cp s3://asf-floorplan-interpreter/models/{model_name}/weights/best.pt outputs/models/{model_name}/weights/best.pt"
        )

    logger.info("Loading all the models")
    room_model = load_model(
        os.path.join("outputs/models", room_model_name, "weights/best.pt")
    )
    window_door_model = load_model(
        os.path.join("outputs/models", window_door_model_name, "weights/best.pt")
    )
    staircase_model = load_model(
        os.path.join("outputs/models", staircase_model_name, "weights/best.pt")
    )
    room_type_model = load_model(
        os.path.join("outputs/models", room_type_model_name, "weights/best.pt")
    )

    logger.info("Predict numbers of rooms etc for each of the evaluation floorplans")
    pred_dict = {}
    for floorplan_url in tqdm(eval_dict.keys()):
        results = full_output(
            floorplan_url,
            room_model,
            window_door_model,
            room_type_model,
            staircase_model=None,
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
                    econest_pred_dict[floorplan_url].get(label_key, None),
                ]
            )
        floorplan_urls.append(floorplan_url)

    all_results = {
        "results": per_label_results,
        "floorplan_urls": floorplan_urls,
        "order": ["Truth", "Model prediction", "Econest prediction"],
    }

    save_to_s3(
        BUCKET_NAME, all_results, os.path.join(output_folder, "all_results.json")
    )

    mse_model_results = {}
    mse_econest_results = {}
    for label_key, results in all_results["results"].items():
        if label_key not in ["floorplan_urls", "order"]:
            model_truth = []
            econest_truth = []
            model_pred = []
            econest_pred = []
            for t, m, e in results:
                if pd.notnull(t):
                    if m != None:
                        model_truth.append(t)
                        model_pred.append(m)
                    if e != None:
                        if e != "\\N":
                            econest_truth.append(t)
                            econest_pred.append(int(e))
            if model_pred:
                mse_model_results[label_key] = {
                    "n": len(model_truth),
                    "mse": mean_squared_error(model_truth, model_pred),
                }
            if econest_pred:
                mse_econest_results[label_key] = {
                    "n": len(econest_truth),
                    "mse": mean_squared_error(econest_truth, econest_pred),
                }
    mse_results = {
        "econest_prediction_mse": mse_econest_results,
        "model_prediction_mse": mse_model_results,
    }

    save_to_s3(
        BUCKET_NAME, mse_results, os.path.join(output_folder, "mse_results.json")
    )
