"""
Create images.jsonl for labelling floorplans using Prodigy

This should look like
{"id":1, "image": "https://storage.googleapis.com/...195041.jpg"}
{"id":2, "image": "https://storage.googleapis.com/...441966.jpg"}
"""

import json

import pandas as pd

from asf_floorplan_interpreter import BUCKET_NAME

floorplans_data = pd.read_csv(
    f"s3://{BUCKET_NAME}/data/floorplans/Econest-Floorplans for Nesta data science project-2023.09-floorplans.csv"
)

output_string = ""
for _, floorplan_row in floorplans_data.iterrows():
    line = {"id": floorplan_row["property_id"], "image": floorplan_row["floorplan_url"]}
    output_string += json.dumps(line, ensure_ascii=False)
    output_string += "\n"

with open(
    "asf_floorplan_interpreter/pipeline/annotation/floorplans.jsonl", "w"
) as outfile:
    outfile.write(output_string)
