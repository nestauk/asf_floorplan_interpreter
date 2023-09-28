#### ONLY TO BE TRAINED ON GPU
### USE IF A RERUN OF TRAINING - CURRENT MODEL WEIGHTS IN S3 BUCKET
## ADD FUNCTION TO SAVE WEIGHTS ON S3
from asf_floorplan_interpreter import PROJECT_DIR
from asf_floorplan_interpreter.getters.get_data import load_files_for_yolo
import os
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)


def train_yolo_model(epoch_number):
    """"""
    load_files_for_yolo()

    model.train(
        data=(PROJECT_DIR / "inputs/config.yaml"), epochs=epoch_number, imgsz=640
    )  # change file path
    # HOW TO SAVE TRAINED MODEL?? IS THERE AN ARG TO DIRECT OUTPUTS? MAYBE PUT OUTPUS IN AN S3 OUTPUT FOLDER
    # AND DIRECT THE WEIGHTS FOR MODELS TO ANOTHER
