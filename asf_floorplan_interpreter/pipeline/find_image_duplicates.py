"""
Read in the raw floorplans and return a subset of deduplicated floorplan ids.
Sometimes the url can be different, but the image is the same.

This needs to be run as a one off:

python asf_floorplan_interpreter/pipeline/find_image_duplicates.py
"""

import json

import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from asf_floorplan_interpreter import BUCKET_NAME
from asf_floorplan_interpreter.getters.get_data import save_to_s3
from asf_floorplan_interpreter.utils.visualise_image import load_image

if __name__ == "__main__":
    floorplans_data = pd.read_csv(
        f"s3://{BUCKET_NAME}/data/floorplans/Econest-Floorplans for Nesta data science project-2023.09-floorplans.csv"
    )

    image_hashes = []
    floorplan_urls = []
    for _, floorplan_row in tqdm(floorplans_data.iterrows()):
        image_url = floorplan_row["floorplan_url"]
        visual_image = load_image(image_url)
        image_hashes.append(hash(str(list(visual_image.getdata()))))
        floorplan_urls.append(image_url)

    _, uix = np.unique(image_hashes, return_index=True)
    unique_images = [floorplan_urls[i] for i in uix]

save_to_s3(
    BUCKET_NAME, unique_images, "data/floorplans/unique_floorplan_urls_2023_09.json"
)
print(f"{len(unique_images)} of the {len(floorplan_urls)} urls are unique images")
# 495 of the 497 urls are unique images
