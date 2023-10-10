import prodigy
from prodigy.components.loaders import JSONL
from ultralytics import YOLO


def load_model(directory):
    """Function to load YOLO model trained on floor plans annotated with polygons identifying rooms"""
    return YOLO(directory)


def yolo_2_segments(results):
    segments = []
    for (x, y, w, h), label in zip(results[0].boxes.xywh, results[0].boxes.cls):
        x_min = x.item() - (w.item() / 2)
        y_min = y.item() - (h.item() / 2)
        x_max = x.item() + (w.item() / 2)
        y_max = y.item() + (h.item() / 2)
        segment = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        segments.append(
            {
                "label": results[0].names[label.item()],
                "points": segment,
                "type": "polygon",
            }
        )
    return segments


@prodigy.recipe("classify-images")
def classify_images(dataset, source, label):
    model = load_model("outputs/models/windows-doors-model/best.pt")

    def predict(stream, model=model):
        for eg in stream:
            results = model(eg["image"])
            span_predictions = yolo_2_segments(results)
            eg["spans"] = span_predictions
            yield eg

    stream = JSONL(source)

    stream = predict(stream, model=model)

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "image_manual",
        "config": {
            "labels": label.split(","),
        },
    }


@prodigy.recipe("classify-everything-images")
def classify_everything_images(dataset, source, label):
    window_door_model = load_model("outputs/models/windows-doors-model/best.pt")
    room_model = load_model("outputs/models/rooms-model/best.pt")

    def predict(stream, window_door_model=window_door_model, room_model=room_model):
        for eg in stream:
            window_door_results = window_door_model(eg["image"])
            span_predictions = yolo_2_segments(window_door_results)

            room_results = room_model(eg["image"])
            span_predictions += yolo_2_segments(room_results)

            eg["spans"] = span_predictions
            yield eg

    stream = JSONL(source)

    stream = predict(stream, window_door_model=window_door_model, room_model=room_model)

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "image_manual",
        "config": {
            "labels": label.split(","),
        },
    }


OPTIONS = [
    {"id": 0, "text": "Ray Ban Wayfarer (Original)"},
    {"id": 1, "text": "Ray Ban Wayfarer (New)"},
    {"id": 2, "text": "Ray Ban Clubmaster (Classic)"},
    {"id": 3, "text": "Ray Ban Aviator (Classic)"},
    {"id": -1, "text": "Other model"},
]


# This isnt working as I'd like yet
@prodigy.recipe("room-type")
def classify_room_type(dataset, source, label):
    room_model = load_model("outputs/models/rooms-model/best.pt")

    # OPTIONS =

    def predict(stream, room_model=room_model):
        for eg in stream:
            room_results = room_model(eg["image"])
            span_predictions = yolo_2_segments(room_results)
            for span_prediction in span_predictions:
                eg["spans"] = [span_prediction]
                eg["options"] = OPTIONS
                yield eg

    stream = JSONL(source)

    stream = predict(stream, room_model=room_model)

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "choice",
        "config": {
            "choice_style": "single",  # or "multiple"
            # Automatically accept and submit the answer if an option is
            # selected (only available for single-choice tasks)
            "choice_auto_accept": True,
        },
    }
