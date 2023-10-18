# Convert json Prodigy output to Yolo8 input

from asf_floorplan_interpreter.getters.get_data import (
    load_prodigy_jsonl_s3_data,
    save_to_s3,
    load_s3_data,
)
from asf_floorplan_interpreter import BUCKET_NAME
import os


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


def convert_prodigy_file(file_name, object_to_class_dict, output_folder_name):
    data = load_prodigy_jsonl_s3_data(BUCKET_NAME, file_name)

    for prod_label in data:
        if prod_label["answer"] == "accept":
            yolo_label = convert_prod_to_yolo(prod_label, object_to_class_dict)
            image_name = prod_label["image"].split("/")[-1].split(".")[0]
            # Output to text file - one file per image
            save_to_s3(
                BUCKET_NAME,
                image_name,
                os.path.join(output_folder_name, f"{image_name}.txt"),
                verbose=False,
            )


if __name__ == "__main__":
    ## Process the room dataset

    file_name = "data/annotation/prodigy_labelled/181023/room_dataset.jsonl"
    output_folder_name = (
        "data/annotation/prodigy_labelled/181023/room_dataset_yolo_formatted/"
    )
    object_to_class_dict = {
        "ROOM": 0,
    }
    convert_prodigy_file(file_name, object_to_class_dict, output_folder_name)

    ## Process the window/door/staircase dataset

    file_name = "data/annotation/prodigy_labelled/181023/window_door_staircase.jsonl"
    output_folder_name = (
        "data/annotation/prodigy_labelled/181023/window_door_staircase_yolo_formatted/"
    )
    object_to_class_dict = {
        "WINDOW": 0,
        "DOOR": 1,
        "STAIRCASE": 2,
    }
    convert_prodigy_file(file_name, object_to_class_dict, output_folder_name)
