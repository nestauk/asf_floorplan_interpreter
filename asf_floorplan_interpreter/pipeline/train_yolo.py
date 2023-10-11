#### ONLY TO BE TRAINED ON GPU
### USE IF A RERUN OF TRAINING - CURRENT MODEL WEIGHTS IN S3 BUCKET
## ADD FUNCTION TO SAVE WEIGHTS ON S3

# code to install requirements on batch machine
import os

try:
    os.system("apt-get update && apt-get install -y libgl1-mesa-glx 1> /dev/null")
    os.system(
        f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/flow_reqs.txt 1> /dev/null"
    )
except:
    pass

from metaflow import FlowSpec, project, step, Parameter, batch, conda

# def train_yolo_model(epoch_number):
#     """Train YOLO model locally"""

#     model.train(
#         data=(PROJECT_DIR / "asf_floorplan_interpreter/config/config.yaml"), epochs=epoch_number, imgsz=640
#     )
#     # HOW TO SAVE TRAINED MODEL?? IS THERE AN ARG TO DIRECT OUTPUTS? MAYBE PUT OUTPUS IN AN S3 OUTPUT FOLDER
#     # AND DIRECT THE WEIGHTS FOR MODELS TO ANOTHER


# def mini_model(config_file, epochs):
#     model = YOLO("yolov8n-seg.pt")
#     results = model.train(
#             batch=8,
#             device="cpu",
#             data=config_file,
#             epochs=epochs,
#             imgsz=120,
#         )
#     return results


# if __name__ == '__main__':
#     model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

#     epochs = 2

#     results = model.train(
#         data=(PROJECT_DIR / "asf_floorplan_interpreter/config/config.yaml"), epochs=epochs, imgsz=640, batch=16, device="cpu",
#         patience=3,
#         save_dir="liztest/here/",
#         project="lizproject"
#     )

# def pip(libraries):
#     # From https://github.com/Netflix/metaflow/issues/24
#     def decorator(function):
#         @functools.wraps(function)
#         def wrapper(*args, **kwargs):
#             import subprocess
#             import sys

#             for library, version in libraries.items():
#                 print('Pip Install:', library, version)
#                 subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', library + '==' + version])
#             return function(*args, **kwargs)

#         return wrapper

#     return decorator


class FloorPlanYolo(FlowSpec):
    config_file = Parameter(
        "config_file",
        help="The config file path for this model",
        default="window_door_test_config.yaml",
    )

    @step
    def start(self):
        """Start flow"""

        self.next(self.load)

    @step
    def load(self):
        import yaml

        os.system(
            "aws s3 cp --recursive s3://asf-floorplan-interpreter/data/roboflow_data/ datasets/data/roboflow_data"
        )

        with open(self.config_file, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.epochs = self.config["epochs"]
        self.yolo_config_file = self.config["yolo_config_file"]
        self.yolo_pretrained_model_name = self.config["yolo_pretrained_model_name"]
        self.batch = self.config["batch"]
        self.patience = self.config["patience"]
        self.imgsz = self.config["imgsz"]
        self.project_name = self.config["project_name"]

        self.next(self.train)

    @batch(gpu=1, memory=60000, cpu=6, queue="job-queue-GPU-nesta-metaflow")
    @step
    def train(self):
        from ultralytics import YOLO
        import torch

        print(torch.cuda.is_available())
        torch.cuda.set_device(0)

        self.model = YOLO(self.yolo_pretrained_model_name)

        results = self.model.train(
            data=self.yolo_config_file,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=0,
            patience=self.patience,
            project=self.project_name,
        )

        self.next(self.end)

    @step
    def end(self):
        """End flow"""

        os.system("rm -rf datasets/data/roboflow_data")

        pass


if __name__ == "__main__":
    FloorPlanYolo()
