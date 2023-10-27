from asf_floorplan_interpreter import BUCKET_NAME, PROJECT_DIR, logger
from nesta_ds_utils.loading_saving.S3 import download_file

# from asf_floorplan_interpreter.utils.s3 import download_directory_from_s3

import json

import boto3
from fnmatch import fnmatch

# def get_data_for_model(s3_directory):
#     """Download data for training model (locally)"""

#     download_directory_from_s3(
#         BUCKET_NAME, s3_directory, (PROJECT_DIR / "inputs/data/roboflow_data/")
#     )

def get_config():
    """Download model config file"""
    download_file("data/config.yaml", BUCKET_NAME, (PROJECT_DIR / "inputs/config.yaml"))


def load_files_for_yolo():
    """Loads relevant files for training the YOLO model"""
    get_data_for_model("data/roboflow_data/")
    get_config()

def get_s3_resource():
    s3 = boto3.resource("s3")
    return s3


def load_prodigy_jsonl_s3_data(bucket_name, file_name):
    """
    Load prodigy jsonl formatted data from S3 location.

    bucket_name: The S3 bucket name
    file_name: S3 key to load
    """
    s3 = get_s3_resource()
    obj = s3.Object(bucket_name, file_name)
    if fnmatch(file_name, "*.jsonl"):
        file = obj.get()["Body"].read().decode()
        return [json.loads(str(item)) for item in file.strip().split("\n")]


def save_to_s3(bucket_name, output_var, output_file_dir, verbose=True):
    s3 = get_s3_resource()

    obj = s3.Object(bucket_name, output_file_dir)

    if fnmatch(output_file_dir, "*.csv"):
        output_var.to_csv("s3://" + bucket_name + "/" + output_file_dir, index=False)
    elif fnmatch(output_file_dir, "*.pkl") or fnmatch(output_file_dir, "*.pickle"):
        obj.put(Body=pickle.dumps(output_var))
    elif fnmatch(output_file_dir, "*.gz"):
        obj.put(Body=gzip.compress(json.dumps(output_var).encode()))
    elif fnmatch(output_file_dir, "*.txt"):
        obj.put(Body=output_var)
    else:
        obj.put(Body=json.dumps(output_var, cls=CustomJsonEncoder))

    if verbose:
        logger.info(f"Saved to s3://{bucket_name} + {output_file_dir} ...")


def load_s3_data(bucket_name, file_name):
    """
    Load data from S3 location.

    bucket_name: The S3 bucket name
    file_name: S3 key to load
    """
    s3 = get_s3_resource()

    obj = s3.Object(bucket_name, file_name)
    if fnmatch(file_name, "*.jsonl.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return [json.loads(line) for line in file]
    if fnmatch(file_name, "*.yml") or fnmatch(file_name, "*.yaml"):
        file = obj.get()["Body"].read().decode()
        return yaml.safe_load(file)
    elif fnmatch(file_name, "*.jsonl"):
        file = obj.get()["Body"].read().decode()
        return [json.loads(line) for line in file]
    elif fnmatch(file_name, "*.json.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return json.load(file)
    elif fnmatch(file_name, "*.json"):
        file = obj.get()["Body"].read().decode()
        return json.loads(file)
    elif fnmatch(file_name, "*.csv"):
        return pd.read_csv("s3://" + bucket_name + "/" + file_name)
    elif fnmatch(file_name, "*.pkl") or fnmatch(file_name, "*.pickle"):
        file = obj.get()["Body"].read().decode()
        return pickle.loads(file)
    else:
        logger.error(
            'Function not supported for file type other than "*.csv", "*.jsonl.gz", "*.jsonl", or "*.json"'
        )
