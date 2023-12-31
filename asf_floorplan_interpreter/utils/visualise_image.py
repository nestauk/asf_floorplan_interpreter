import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PIL import Image, ImageDraw, ImageFont
from asf_floorplan_interpreter.getters.get_data import (
    load_prodigy_jsonl_s3_data,
    get_s3_data_paths,
    load_s3_data,
    get_s3_resource,
)
import urllib.request
from asf_floorplan_interpreter import BUCKET_NAME
import io
import os

label_colors = {
    "ROOM": "pink",
    "WINDOW": "blue",
    "DOOR": "red",
    "STAIRCASE": "yellow",
    "KITCHEN": "orange",
    "LIVING": "magenta",
    "RESTROOM": "green",
    "BEDROOM": "purple",
    "GARAGE": "brown",
    "OTHER": "cyan",
}


def get_corresponding_label(img_path, label_list):
    """
    Function that takes a single image path, and returns its corresponding label
    Arguments:
    img_path: A single element of the list generated by running get_s3_data_paths()
    label_list: The entire list generated by running load_prodigy_jsonl_s3_data()
    """
    # Get image name
    img_name = img_path.split("/")[-1].split(".")[0]
    # Get a list of urls from each label
    image_urls = [label_list[i]["image"] for i in range(0, len(label_list))]
    # Extract the image name from each url
    label_names = [i.split("/")[-1].split(".")[0] for i in image_urls]
    # Get position of the label corresponding to the image given
    position = label_names.index(img_name)
    return label_list[position]


def load_image(img):
    """
    Can input an image from a url or a local directory, and load it
    """
    try:
        urllib.request.urlretrieve(img, "image_temporary.png")
        visual_image = Image.open("image_temporary.png")
        os.system("rm image_temporary.png")
    except:
        visual_image = Image.open(img)
    return visual_image


def load_s3_image_and_label(image_position, img_list, label_list):
    """
    Arguments:
    image position [int]: position of the image you want in the image list
    img_list: The list generated by running get_s3_data_paths
    label_list: The list generated by running load_prodigy_jsonl_data
    """

    img_path = img_list[image_position]
    object_label = get_corresponding_label(img_path, label_list)

    img = load_s3_data(BUCKET_NAME, img_path)
    readable_image = io.BytesIO(img)
    visual_image = Image.open(readable_image)
    labels = object_label["spans"]

    return visual_image, labels


def overlay_boundaries_plot(
    visual_image, labels, label_colors=label_colors, show=True, plot_label=True
):
    # Create a drawing context
    draw = ImageDraw.Draw(visual_image)

    # Overlay bounding boxes and labels on the image
    labels_found = set()
    for label in labels:
        if label["type"] == "polygon":
            labels_found.add(label["label"])
            label["color"] = label_colors.get(label["label"], "black")
            points = label["points"]
            for i in range(len(points)):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % len(points)]
                draw.line([(x1, y1), (x2, y2)], fill=label["color"], width=10)
            if plot_label:
                label_text = label["label"]
                x, y = points[0]  # Take the first point for labeling
                font_size = 36
                font = ImageFont.truetype(
                    font="/System/Library/Fonts/Supplemental/Arial.ttf", size=font_size
                )
                draw.text(
                    (x, y), label_text, fill="black", font=font
                )  # Adjust font and fill color as needed

    # Display the image with the overlaid bounding boxes and labels
    handles = [
        Rectangle((0, 0), 1, 1, color=v)
        for k, v in label_colors.items()
        if k in labels_found
    ]
    labels = [k for k in label_colors.keys() if k in labels_found]

    plt.imshow(np.array(visual_image))
    plt.axis("off")
    if not plot_label:
        plt.legend(handles, labels, loc=(1.04, 0.5), fontsize="small")
    if show:
        plt.show()
    else:
        return plt


if __name__ == "__main__":
    """Example prints three annotated rooms, from the 301023 batch"""
    label_list = load_prodigy_jsonl_s3_data(
        BUCKET_NAME, "data/annotation/prodigy_labelled/301023/room_dataset.jsonl"
    )
    img_list = get_s3_data_paths(
        BUCKET_NAME,
        "data/annotation/prodigy_labelled/301023/yolo_formatted/room_yolo_formatted/images/train/",
    )
    for i in range(0, 3):
        visual_image, labels = load_s3_image_and_label(
            image_position, img_list, label_list
        )
        overlay_boundaries_plot(visual_image, labels)

    visual_image, labels = load_s3_image_and_label(i, img_list, label_list)
