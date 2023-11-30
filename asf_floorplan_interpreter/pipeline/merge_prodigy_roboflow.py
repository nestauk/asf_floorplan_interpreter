"""
Combine the Roboflow and our labelled data using Prodigy into a single dataset
"""

from asf_floorplan_interpreter.pipeline.prodigy_to_yolo import (
    convert_prodigy_file,
    split_save_data,
)
from asf_floorplan_interpreter.getters.get_data import (
    save_to_s3,
    load_s3_data,
    get_s3_data_paths,
)
from asf_floorplan_interpreter import BUCKET_NAME
from asf_floorplan_interpreter.utils.config_utils import read_base_config

from collections import Counter
import os
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":
    config = read_base_config()

    prodigy_labelled_dir = f"data/annotation/prodigy_labelled/{config['prodigy_labelled_date']['window_door_staircase_dataset']}"
    roboflow_dir = "data/roboflow_data"
    output_dir = os.path.join(
        prodigy_labelled_dir, "yolo_formatted/window_door_prodigy_plus_roboflow"
    )

    eval_data = load_s3_data(BUCKET_NAME, config["eval_data_file"])

    object_to_class_dict = {
        "WINDOW": 0,
        "DOOR": 1,
    }

    print("Sync all the existing Roboflow images to this new location")

    os.system(
        f"aws s3 sync s3://asf-floorplan-interpreter/{roboflow_dir}/images/ s3://asf-floorplan-interpreter/{output_dir}/images --no-progress"
    )

    print("Edit the roboflow labels to just give us the labels we want, and save them")

    roboflow_dict = {
        0: "door",
        1: "double door",
        2: "folding door",
        3: "room",
        4: "sliding door",
        5: "window",
    }
    new_roboflow_dict = {"window": 0, "door": 1}

    label_files = get_s3_data_paths(BUCKET_NAME, os.path.join(roboflow_dir, "labels"))
    for label_file in tqdm(label_files):
        if ".txt" in label_file:
            labels = load_s3_data(BUCKET_NAME, label_file)
            image_name = label_file.split("/")[-1]
            data_type = label_file.split("/")[-2]  # e.g. "train"
            new_labels = []
            for label in labels:
                label_type = roboflow_dict[int(label.split()[0])]  # e.g. "window"
                label_coords = " ".join(label.split()[1:])
                if label_type in new_roboflow_dict:
                    new_label = new_roboflow_dict[label_type]
                    new_labels.append(" ".join([str(new_label), label_coords]))
            if new_labels:
                yolo_label = "\n".join(new_labels)
                save_to_s3(
                    BUCKET_NAME,
                    yolo_label,
                    os.path.join(output_dir, f"labels/{data_type}/{image_name}"),
                    verbose=False,
                )

    print("Format and add our prodigy labels to the datasets")
    # Could use same proportions of train/test as in roboflow, but since we have such little data 92% train might be too much!
    # roboflow_split = Counter([file.split('data/roboflow_data/labels/')[1].split('/')[0] for file in label_files])
    # train_prop = roboflow_split['train']/(roboflow_split['train'] + roboflow_split['test'] + roboflow_split['val']) # 0.92
    # test_prop = roboflow_split['test']/(roboflow_split['train'] + roboflow_split['test'] + roboflow_split['val']) # 0.03

    eval_floorplan_urls = eval_data[
        pd.notnull(eval_data["total_rooms"]) & eval_data["total_rooms"] != 0
    ]["floorplan_url"].tolist()

    train_prop = 0.6
    test_prop = 0.2
    # val_prop = 1 - (train_prop + test_prop) # Dont need to set this
    if train_prop + test_prop == 1:
        print("Warning - there will be no data in the validation set")

    prod_file_name = os.path.join(prodigy_labelled_dir, "window_door_staircase.jsonl")

    window_door_yolo_labels = convert_prodigy_file(prod_file_name, object_to_class_dict)
    split_save_data(
        window_door_yolo_labels, eval_floorplan_urls, train_prop, test_prop, output_dir
    )
