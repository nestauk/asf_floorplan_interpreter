# Convert json Prodigy output to Yolo8 input

import os
import random
import requests
from io import BytesIO
import pandas as pd

import boto3

from asf_floorplan_interpreter.getters.get_data import (
    load_prodigy_jsonl_s3_data,
    save_to_s3,
    load_s3_data,
)
from asf_floorplan_interpreter import BUCKET_NAME, logger

from asf_floorplan_interpreter.utils.config_utils import read_base_config
from asf_floorplan_interpreter.utils.visualise_image import load_image


def scale_points_by_hw(lst, width, height):
    "helper function to scale the x,y coordinates by object height and width"
    return [[elem[0] / width, elem[1] / height] for elem in lst]


def convert_prod_to_yolo(prod_label, object_to_class_dict):
    """
    Takes a single prodigy label (dictionary, saved in json file format) and
    converts it to a yolo8 label (string of class and x,y coordinates)

    Args:
    prod_label: a prodigy labels
    object_to_class_dict: A dictionary for converting the names of classes in prodigy (eg, window, bathroom)
                        to a number for the yolo label - eg {'WINDOW': 5, 'DOOR': 0, 'OTHER_ROOM': 3, 'ROOM': 3, 'OTHER_DOOR': 0}
    """

    output_list = []
    w = prod_label["width"]
    h = prod_label["height"]

    if "spans" in prod_label:
        for i in range(0, len(prod_label["spans"])):
            # Get the class and output the corresponding number
            # Don't include this class in the output if it's not in object_to_class_dict
            if prod_label["spans"][i]["label"] in object_to_class_dict:
                class_no = object_to_class_dict[prod_label["spans"][i]["label"]]

                # Get the polygon points, scale by the width and height of the image
                points = prod_label["spans"][i]["points"]
                scaled_list = scale_points_by_hw(points, w, h)
                flat_list = [item for sublist in scaled_list for item in sublist]

                # Combine scaled points with class number
                total_shape = [class_no] + flat_list
                output_list.append(total_shape)

    # Flatten from list to string
    final_format = [" ".join(map(str, item)) for item in output_list]

    return final_format


def convert_prodigy_file(
    file_name, object_to_class_dict, unique_images, use_all=False, image_key="image"
):
    data = load_prodigy_jsonl_s3_data(BUCKET_NAME, file_name)

    yolo_labels = {}
    prod_time_test = {}
    for prod_label in data:
        if prod_label[image_key] in unique_images:
            if use_all or prod_label["answer"] == "accept":
                yolo_label = convert_prod_to_yolo(prod_label, object_to_class_dict)
                image_url = prod_label[image_key]
                # Use the latest label if an image has come up more than once
                if prod_label["_timestamp"] > prod_time_test.get(image_url, 0):
                    yolo_labels[
                        image_url
                    ] = yolo_label  # Will get replaced with the most recent
                    prod_time_test[image_url] = prod_label["_timestamp"]

    yolo_labels = [(k, v) for k, v in yolo_labels.items()]
    logger.info(
        f"Original data was {len(data)} annotations, filtered for accepted labels and deduplicated gives us {len(yolo_labels)} annotations"
    )
    return yolo_labels


def transform_room_type_json(original_json_path, hw_json_path):
    """Function that takes the room type json and converts it to a json in the dictionary format
    of all other room labelling tasks (with the the house, rather than room, as the unit of analysis).
    Merges in height and width data from sepearate json file.
    Args:
    original_json_path: s3 path to the latest room type model data
    hw_json_path: s3 path to the prodigy json that simply accepts/ rejects all images, and provides height/ width data
    ('data/annotation/prodigy_labelled/quality_dataset.jsonl' as of 13 November 2023)"""
    room_dict = load_prodigy_jsonl_s3_data(BUCKET_NAME, original_json_path)
    height_width_dict = load_prodigy_jsonl_s3_data(BUCKET_NAME, hw_json_path)
    transformed_data = {}

    # Iterate through the original data

    for item in room_dict:
        if item.get("answer") == "accept":
            image_id = item["image"]
            if image_id not in transformed_data:
                transformed_data[image_id] = {"image": image_id, "spans": []}
            # Extract the relevant information from the item
            accept_options = item.get("accept", [])
            options = item.get("options", [])
            if (
                accept_options
            ):  # I'm not sure why but sometimes this is [] which causes an error
                spans = {
                    "label": [
                        option["text"]
                        for option in options
                        if option["id"] in accept_options
                    ][0],
                    "points": item["spans"][0][
                        "points"
                    ],  # Assuming there's always one span
                    "type": item["spans"][0]["type"],
                    "input_hash": item["_input_hash"],
                    "task_id": item["_task_hash"],
                }
                # Append the span to the image's 'span' list
                transformed_data[image_id]["spans"].append(spans)

    # Convert the transformed data to a list of dictionaries
    transformed_data_list = list(transformed_data.values())

    # Merge in height and width values
    room_type_df = pd.DataFrame(transformed_data_list)
    height_width_df = pd.DataFrame(height_width_dict)
    merged_df = pd.merge(
        room_type_df,
        height_width_df[["image", "height", "width"]],
        on="image",
        how="left",
    )
    data_for_conversion = merged_df.to_dict(orient="records")

    return data_for_conversion


def find_identical_points(data):
    """Function to check if there are duplicate rooms in the room type model
    Not yet an issue, if it says there are, may need to consider updated transform function to include most recent label
    """
    identical_points = []

    for item in data:
        image_id = item["image"]
        spans = item.get("spans", [])

        for span in spans:
            points_list = span.get("points", [])
            span_id = span.get("label", [])

            for i in range(len(points_list)):
                for j in range(i + 1, len(points_list)):
                    if points_list[i] == points_list[j]:
                        identical_points.append(
                            {
                                "image": image_id,
                                "span_id": span_id,
                                "identical_points": points_list[i],
                            }
                        )

    return identical_points


def count_label_types(data):
    """Helper function to check how many of each type are in the labelled set"""
    label_counts = {}

    for item in data:
        spans = item.get("spans", [])

        for span in spans:
            label = span.get("label", [])

            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

    return label_counts


def check_room_output(transformed_data_list):
    """Runs the helper functions to check the conversion has worked"""
    logger.info(f"Total images represented: {len(transformed_data_list)}")
    logger.info(f"Total rooms by type: {count_label_types(transformed_data_list)}")
    duplicated = find_identical_points(transformed_data_list)
    if len(duplicated) > 0:
        logger.info("Caution, duplicate points found")
    else:
        logger.info("No duplicate labels found")


def convert_room_type_to_yolo(data, object_to_class_dict):
    yolo_labels = {}

    for prod_label in data:
        # Remove duplicate labels, keep the first one only (i.e. in the same floorplan a room has been labelled twice)
        task_ids = set()
        new_spans = []
        for span in prod_label["spans"]:
            if span["task_id"] not in task_ids:
                new_spans.append(span)
                task_ids.add(span["task_id"])

        prod_label["spans"] = new_spans

        yolo_label = convert_prod_to_yolo(prod_label, object_to_class_dict)
        image_url = prod_label["image"]

        # Use the latest label if an image has come up more than once
        yolo_labels[image_url] = yolo_label

    yolo_labels = [(k, v) for k, v in yolo_labels.items()]
    logger.info(
        f"Original data was {len(data)} annotations, filtered for accepted labels and deduplicated gives us {len(yolo_labels)} annotations"
    )
    return yolo_labels


def split_save_data(
    yolo_labels, eval_floorplan_urls, train_prop, test_prop, output_folder_name
):
    """
    Arguments:
        yolo_labels (list): the labelled floorplans in the format needed for Yolo
        eval_floorplan_urls (list): the urls for the floorplans in the final pipeline evaluation dataset (counts of rooms)
        train_prop (float): proportion of data in the training set
        test_prop (float): proportion of data in the test set
        output_folder_name (str): the folder name to output the test/train/val data splits into

    A note on test/evaluation:
        When we are testing this specific image segmentation model at how well it segments the image we use the test+validation datasets
        When we are evaluating the final room count outputs of the entire pipeline we use the evaluation dataset
        We don't want to evaluate the model with data which was used in the train or test datasets, so try to make it so most of (if not all) of
        the validation data is the evaluation data
    """

    logger.warning(
        f"You should start off running this script with the {output_folder_name} folder cleared out"
    )

    s3 = boto3.resource("s3")
    bucket = s3.Bucket(name=BUCKET_NAME)

    # Randomly split up
    random.seed(42)
    random.shuffle(yolo_labels)

    # Separate out the floorplans which are in the evaluation dataset
    yolo_labels_not_eval = [y for y in yolo_labels if y[0] not in eval_floorplan_urls]
    yolo_labels_in_eval = [y for y in yolo_labels if y[0] in eval_floorplan_urls]
    logger.info(
        f"{len(yolo_labels_in_eval)} labelled images were in the evaluation dataset"
    )

    # Order so that the labels in the evaluation dataset are at the end
    yolo_labels_ordered = yolo_labels_not_eval + yolo_labels_in_eval

    train_data_n = round(len(yolo_labels_ordered) * train_prop)  # e.g. 60
    test_data_n = round(len(yolo_labels_ordered) * test_prop)  # e.g. 20

    # Abandoned: There is more evaluation data than validation, so we are eating into the size
    # of our training set. Best to train on as much as we can, and cut out some of the evaluation data if needed.
    # Only include floorplans not in the evaluation data in the train + test
    # val_data_n = len(yolo_labels_ordered) - (train_data_n+test_data_n) # e.g. 20
    # eval_difference = len(yolo_labels_in_eval) - val_data_n # e.g 30 - 20 = 10
    # if eval_difference > 0:
    #     # If there are more eval floorplans than the size of the validation data
    #     # then we need to rebalance the train/test split
    #     logger.info(f"Rebalancing the train/test split to a 80/20 split since there are more evaluation floorplans than the desired validation size")
    #     train_data_n = round(len(yolo_labels_not_eval) * 0.8) # e.g. 56
    #     test_data_n = round(len(yolo_labels_not_eval) * 0.2) # e.g. 14

    logger.info(f"Saving {train_data_n} image in the training set")
    logger.info(f"Saving {test_data_n} image in the test set")
    logger.info(
        f"Saving {len(yolo_labels_ordered) - (train_data_n+test_data_n)} image in the val set"
    )

    # Save labels and download + save images too
    for i, floorplan_labels in enumerate(yolo_labels_ordered):
        image_url = floorplan_labels[0]
        yolo_label = "\n".join(floorplan_labels[1])
        image_name = image_url.split("/")[-1].split(".")[0]

        if i in range(0, train_data_n):
            data_type = "train"
        elif i in range(train_data_n, (train_data_n + test_data_n)):
            data_type = "test"
        else:
            data_type = "val"

        image_data = requests.get(image_url).content
        # Save the image
        bucket.upload_fileobj(
            BytesIO(image_data),
            os.path.join(output_folder_name, f"images/{data_type}/{image_name}.jpg"),
        )
        # Save the image in black and white

        # Save the image to an in-memory file
        pil_image = load_image(image_url)
        pil_image = pil_image.convert("L")
        in_mem_file = BytesIO()
        pil_image.save(in_mem_file, format="JPEG")
        in_mem_file.seek(0)

        # Upload image to s3
        bucket.upload_fileobj(
            in_mem_file,
            os.path.join(
                output_folder_name + "_bw", f"images/{data_type}/{image_name}.jpg"
            ),
        )

        # Save the labels
        save_to_s3(
            BUCKET_NAME,
            yolo_label,
            os.path.join(output_folder_name, f"labels/{data_type}/{image_name}.txt"),
            verbose=False,
        )
        save_to_s3(
            BUCKET_NAME,
            yolo_label,
            os.path.join(
                output_folder_name + "_bw", f"labels/{data_type}/{image_name}.txt"
            ),
            verbose=False,
        )


if __name__ == "__main__":
    config = read_base_config()
    prodigy_labelled_date = {
        "room_dataset": config["prodigy_labelled_date"]["room_dataset"],
        "window_door_staircase_dataset": config["prodigy_labelled_date"][
            "window_door_staircase_dataset"
        ],
        "room_type_from_labels_dataset": config["prodigy_labelled_date"][
            "room_type_from_labels_dataset"
        ],
        "staircase_dataset": config["prodigy_labelled_date"]["staircase_dataset"],
    }
    hw_json_path = config["hw_json_path"]

    # Our hold out evaluation data (manually labelled with final room counts)
    eval_data = load_s3_data(BUCKET_NAME, config["eval_data_file"])
    eval_floorplan_urls = eval_data[
        pd.notnull(eval_data["total_rooms"]) & eval_data["total_rooms"] != 0
    ]["floorplan_url"].tolist()

    train_prop = 0.6
    test_prop = 0.2
    # val_prop = 1 - (train_prop + test_prop) # Dont need to set this
    if train_prop + test_prop == 1:
        print("Warning - there will be no data in the validation set")

    # Get the list of unique floorplans
    unique_images = set(load_s3_data(BUCKET_NAME, config["unique_floorplan_file"]))

    # ================================================================
    print("Process the room dataset")

    prodigy_labelled_dir = (
        f"data/annotation/prodigy_labelled/{prodigy_labelled_date['room_dataset']}"
    )
    prod_file_name = os.path.join(prodigy_labelled_dir, "room_dataset.jsonl")
    yolo_data_folder_name = os.path.join(
        prodigy_labelled_dir, "yolo_formatted/room_yolo_formatted"
    )
    object_to_class_dict = {
        "ROOM": 0,
    }
    room_yolo_labels = convert_prodigy_file(
        prod_file_name, object_to_class_dict, unique_images
    )
    split_save_data(
        room_yolo_labels,
        eval_floorplan_urls,
        train_prop,
        test_prop,
        yolo_data_folder_name,
    )

    # ================================================================
    print("Process room type dataset")

    prodigy_labelled_dir = f"data/annotation/prodigy_labelled/{prodigy_labelled_date['room_type_from_labels_dataset']}"
    prod_file_name = os.path.join(
        prodigy_labelled_dir, "room_type_from_labels_dataset.jsonl"
    )
    yolo_data_folder_name = os.path.join(
        prodigy_labelled_dir, "yolo_formatted/room_type_yolo_formatted"
    )
    object_to_class_dict = {
        "RESTROOM": 0,
        "BEDROOM": 1,
        "KITCHEN": 2,
        "LIVING": 3,
        "GARAGE": 4,
        "OTHER": 5,
    }
    prodigy_convert = transform_room_type_json(prod_file_name, hw_json_path)
    check_room_output(prodigy_convert)
    room_type_yolo_labels = convert_room_type_to_yolo(
        prodigy_convert, object_to_class_dict
    )
    split_save_data(
        room_type_yolo_labels,
        eval_floorplan_urls,
        train_prop,
        test_prop,
        yolo_data_folder_name,
    )

    # ================================================================
    print("Process just staircase dataset")

    prodigy_labelled_dir = (
        f"data/annotation/prodigy_labelled/{prodigy_labelled_date['staircase_dataset']}"
    )
    prod_file_name = os.path.join(prodigy_labelled_dir, "staircase_dataset.jsonl")
    yolo_data_folder_name = os.path.join(
        prodigy_labelled_dir, "yolo_formatted/staircase_yolo_formatted"
    )
    object_to_class_dict = {"STAIRCASE": 0}
    staircase_yolo_labels = convert_prodigy_file(
        prod_file_name, object_to_class_dict, unique_images, image_key="path"
    )
    split_save_data(
        staircase_yolo_labels,
        eval_floorplan_urls,
        train_prop,
        test_prop,
        yolo_data_folder_name,
    )
