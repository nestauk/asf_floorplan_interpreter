from asf_floorplan_interpreter import BUCKET_NAME, PROJECT_DIR, logger
from nesta_ds_utils.loading_saving.S3 import download_file

import json

import boto3
from fnmatch import fnmatch
import yaml


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
    elif fnmatch(file_name, "*.txt"):
        return obj.get()["Body"].read().decode().split("\n")
    elif fnmatch(file_name, "*.jpg"):
        return obj.get()["Body"].read()
    else:
        logger.error(
            'Function not supported for file type other than "*.csv", "*.jsonl.gz", "*.jsonl", "*.jpg" or "*.json"'
        )


def get_s3_data_paths(bucket_name, root):
    """
    Get all paths to particular file types in a S3 root location

    bucket_name: The S3 bucket name
    root: The root folder to look for files in
    """
    s3 = get_s3_resource()

    bucket = s3.Bucket(bucket_name)

    s3_keys = []
    for files in bucket.objects.filter(Prefix=root):
        key = files.key
        s3_keys.append(key)

    return s3_keys
