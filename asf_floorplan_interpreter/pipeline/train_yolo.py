import os

try:
    os.system("apt-get update && apt-get install -y libgl1-mesa-glx 1> /dev/null")
    os.system(
        f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/flow_reqs.txt 1> /dev/null"
    )
except:
    pass

from metaflow import FlowSpec, project, step, Parameter, batch, conda


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

    @batch(
        gpu=1,
        memory=60000,
        cpu=6,
        queue="job-queue-GPU-nesta-metaflow",
        shared_memory=500,
    )
    @step
    def train(self):
        from ultralytics import YOLO
        import torch

        print(torch.cuda.is_available())
        torch.cuda.set_device(0)
        os.system(
            "aws s3 sync s3://asf-floorplan-interpreter/data/roboflow_data/ datasets/data/roboflow_data"
        )
        model = YOLO(self.yolo_pretrained_model_name)

        results = model.train(
            data=self.yolo_config_file,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=0,
            patience=self.patience,
            project=self.project_name,
        )

        model.export()
        os.system(
            f"aws s3 cp {self.project_name}/train/weights/best.pt s3://asf-floorplan-interpreter/{self.config_file.split('.yaml')[0]}/"
        )
        os.system(
            f"aws s3 cp {self.yolo_config_file} s3://asf-floorplan-interpreter/{self.config_file.split('.yaml')[0]}/"
        )
        self.next(self.end)

    @step
    def end(self):
        """End flow"""

        pass


if __name__ == "__main__":
    FloorPlanYolo()
