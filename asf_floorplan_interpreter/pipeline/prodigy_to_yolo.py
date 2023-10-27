# Convert json Prodigy output to Yolo8 input

import os
import random
import requests
from io import BytesIO

import boto3

from asf_floorplan_interpreter.getters.get_data import (
    load_prodigy_jsonl_s3_data,
    save_to_s3,
    load_s3_data,
)
from asf_floorplan_interpreter import BUCKET_NAME


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

    for i in range(0, len(prod_label["spans"])):
        # Get the class and output the corresponding number
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


def convert_prodigy_file(file_name, object_to_class_dict):
    data = load_prodigy_jsonl_s3_data(BUCKET_NAME, file_name)

    yolo_labels = {}
    prod_time_test = {}
    for prod_label in data:
        if prod_label["answer"] == "accept":
            yolo_label = convert_prod_to_yolo(prod_label, object_to_class_dict)
            image_url = prod_label["image"]
            # Use the latest label if an image has come up more than once
            if prod_label["_timestamp"] > prod_time_test.get(image_url, 0):
                yolo_labels[
                    image_url
                ] = yolo_label  # Will get replaced with the most recent
                prod_time_test[image_url] = prod_label["_timestamp"]

    yolo_labels = [(k, v) for k, v in yolo_labels.items()]
    print(
        f"Original data was {len(data)} annotations, filtered for accepted labels and deduplicated gives us {len(yolo_labels)} annotations"
    )
    return yolo_labels


def split_save_data(yolo_labels, train_prop, test_prop, output_folder_name):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(name=BUCKET_NAME)

    # Randomly split up
    random.seed(42)
    random.shuffle(yolo_labels)

    train_data_n = round(len(yolo_labels) * train_prop)
    test_data_n = round(len(yolo_labels) * test_prop)
    print(f"Saving {train_data_n} image in the training set")
    print(f"Saving {test_data_n} image in the test set")
    print(
        f"Saving {len(yolo_labels) - (train_data_n+test_data_n)} image in the val set"
    )

    # Save labels and download + save images too
    for i, floorplan_labels in enumerate(yolo_labels):
        image_url = floorplan_labels[0]
        yolo_label = "\n".join(floorplan_labels[1])
        image_name = image_url.split("/")[-1].split(".")[0]

        image_data = requests.get(image_url).content

        if i in range(0, train_data_n):
            data_type = "train"
        elif i in range(train_data_n, (train_data_n + test_data_n)):
            data_type = "test"
        else:
            data_type = "val"

        # Save the image
        bucket.upload_fileobj(
            BytesIO(image_data),
            os.path.join(output_folder_name, f"images/{data_type}/{image_name}.jpg"),
        )
        # Save the labels
        save_to_s3(
            BUCKET_NAME,
            yolo_label,
            os.path.join(output_folder_name, f"labels/{data_type}/{image_name}.txt"),
            verbose=False,
        )


if __name__ == "__main__":
    prodigy_labelled_date = "191023"

    train_prop = 0.6
    test_prop = 0.2
    # val_prop = 1 - (train_prop + test_prop) # Dont need to set this
    if train_prop + test_prop == 1:
        print("Warning - there will be no data in the validation set")

    print("Process the room dataset")

    prod_file_name = (
        f"data/annotation/prodigy_labelled/{prodigy_labelled_date}/room_dataset.jsonl"
    )
    yolo_data_folder_name = (
        f"data/annotation/prodigy_labelled/{prodigy_labelled_date}/room_yolo_formatted/"
    )
    object_to_class_dict = {
        "ROOM": 0,
    }
    room_yolo_labels = convert_prodigy_file(prod_file_name, object_to_class_dict)
    split_save_data(room_yolo_labels, train_prop, test_prop, yolo_data_folder_name)

    print("Process the window/door/staircase dataset")

    prod_file_name = f"data/annotation/prodigy_labelled/{prodigy_labelled_date}/window_door_staircase.jsonl"
    yolo_data_folder_name = f"data/annotation/prodigy_labelled/{prodigy_labelled_date}/window_door_staircase_yolo_formatted/"
    object_to_class_dict = {
        "WINDOW": 0,
        "DOOR": 1,
        "STAIRCASE": 2,
    }
    window_door_yolo_labels = convert_prodigy_file(prod_file_name, object_to_class_dict)
    split_save_data(
        window_door_yolo_labels, train_prop, test_prop, yolo_data_folder_name
    )

    # Original data was 122 annotations, filtered for accepted labels and deduplicated gives us 105 annotations
    # Saving 63 image in the training set
    # Saving 21 image in the test set
    # Saving 21 image in the val set
    # Process the window/door/staircase dataset
    # Original data was 119 annotations, filtered for accepted labels and deduplicated gives us 102 annotations
    # Saving 61 image in the training set
    # Saving 20 image in the test set
    # Saving 21 image in the val set
