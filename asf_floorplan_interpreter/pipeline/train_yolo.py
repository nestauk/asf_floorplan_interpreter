#### ONLY TO BE TRAINED ON GPU
### USE IF A RERUN OF TRAINING - CURRENT MODEL WEIGHTS IN S3 BUCKET
## ADD FUNCTION TO SAVE WEIGHTS ON S3

import os

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


@project(name="floorplan_yolo")
# @conda_base(libraries={'ultralytics': '8.0.195'})
class FloorPlanYolo(FlowSpec):
    config_file = Parameter(
        "config_file",
        help="The config file path for this model",
        default="asf_floorplan_interpreter/config/window_door_test_config.yaml",
    )

    @step
    def start(self):
        """Start flow"""

        self.next(self.load)

    @conda(libraries={"pyyaml": "5.3.1"})
    @step
    def load(self):
        import yaml

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
    # @conda(libraries={"ultralytics": "8.0.195"}) # For some reason this doesnt work
    # @pip(libraries={'ultralytics': '8.0.195'})
    @step
    def train(self):
        # from ultralytics import YOLO

        # self.model = YOLO(self.yolo_pretrained_model_name)

        # results = self.model.train(
        #     data=self.yolo_config_file,
        #     epochs=self.epochs,
        #     imgsz=self.imgsz,
        #     batch=self.batch,
        #     device="gpu",
        #     patience=self.patience,
        #     project=self.project_name
        # )
        results = 2

        self.next(self.save)

    @step
    def save(self):
        self.r = 2 + 2

        self.next(self.end)

    @step
    def end(self):
        """End flow"""
        pass


if __name__ == "__main__":
    FloorPlanYolo()
