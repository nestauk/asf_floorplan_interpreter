import yaml
import os


def read_base_config():
    config_path = os.path.join("asf_floorplan_interpreter/config/base.yaml")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
