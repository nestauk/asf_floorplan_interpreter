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


@prodigy.recipe("classify-window-door")
def classify_window_door(dataset, source, label):
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
            "port": 8501,
            "custom_theme": {
                "labels": {"WINDOW": "blue", "DOOR": "red", "STAIRCASE": "yellow"}
            },
            "buttons": ["accept", "ignore", "undo"],
            "keymap_by_label": {"WINDOW": "w", "DOOR": "d", "STAIRCASE": "s"},
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


@prodigy.recipe("classify-rooms")
def classify_rooms(dataset, source, label):
    room_model = load_model("outputs/models/rooms-model/best.pt")

    def predict(stream, room_model=room_model):
        for eg in stream:
            room_results = room_model(eg["image"])
            span_predictions = yolo_2_segments(room_results)

            eg["spans"] = span_predictions
            yield eg

    stream = JSONL(source)

    stream = predict(stream, room_model=room_model)

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "image_manual",
        "config": {
            "labels": label.split(","),
        },
    }


@prodigy.recipe("room-type-from-labels")
def classify_room_type_from_labels(dataset, source, label):
    """
    Assumes the input stream already has the room polygons labelled
    """

    OPTIONS_keymap = {
        "RESTROOM": "r",
        "BEDROOM": "b",
        "KITCHEN": "k",
        "LIVING": "l",
        "GARAGE": "g",
        "OTHER": "o",
    }

    OPTIONS = [
        {"id": OPTIONS_keymap.get(label_name, i), "text": label_name}
        for i, label_name in enumerate(label.split(","))
    ]

    # keymap_by_label is in the form {"0": "RESTROOM", "1": "BEDROOM",..}
    OPTIONS_fromnum = {
        str(i): OPTIONS_keymap.get(label_name, i)
        for i, label_name in enumerate(label.split(","))
    }

    # OPTIONS += [{"id": -1, "text": "Other"}]

    room_meta_keys = [
        "_input_hash",
        "_task_hash",
        "_view_id",
        "width",
        "height",
        "answer",
        "_timestamp",
        "_annotator_id",
        "_session_id",
    ]

    def predict(stream):
        for eg in stream:
            span_dict = {"id": eg["id"], "image": eg["image"]}
            for span in eg["spans"]:
                # One span at a time
                span_dict["spans"] = [span]
                # Keep the room labelled metadata (not sure how useful, but just in case!)
                span_dict["room_meta"] = {k: eg.get(k) for k in room_meta_keys}
                span_dict["options"] = OPTIONS
                yield span_dict

    stream = JSONL(source)

    stream = predict(stream)

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "choice",
        "config": {
            "choice_style": "single",  # or "multiple"
            # Automatically accept and submit the answer if an option is
            # selected (only available for single-choice tasks)
            "choice_auto_accept": True,
            "port": 8501,
            "buttons": ["accept", "ignore", "undo"],
            "keymap_by_label": OPTIONS_fromnum,
            "keymap": {"accept": ["enter"]},
        },
    }


@prodigy.recipe("room-type")
def classify_room_type(dataset, source, label):
    room_model = load_model("models/room_config_yolov8m/weights/best.pt")

    OPTIONS_keymap = {
        "RESTROOM": "r",
        "BEDROOM": "b",
        "KITCHEN": "k",
        "LIVING": "l",
        "GARAGE": "g",
        "OTHER": "o",
    }

    OPTIONS = [
        {"id": OPTIONS_keymap.get(label_name, i), "text": label_name}
        for i, label_name in enumerate(label.split(","))
    ]

    # keymap_by_label is in the form {"0": "RESTROOM", "1": "BEDROOM",..}
    OPTIONS_fromnum = {
        str(i): OPTIONS_keymap.get(label_name, i)
        for i, label_name in enumerate(label.split(","))
    }

    # OPTIONS += [{"id": -1, "text": "Other"}]

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
            "port": 8082,
            "buttons": ["accept", "ignore", "undo"],
            "keymap_by_label": OPTIONS_fromnum,
            "keymap": {"accept": ["enter"]},
        },
    }


@prodigy.recipe("label-quality")
def label_quality(dataset, source, label):
    def yield_stream(stream):
        for eg in stream:
            yield eg

    stream = JSONL(source)

    stream = yield_stream(stream)

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "image_manual",
        "config": {
            "labels": label.split(","),
            "port": 8501,
        },
    }
