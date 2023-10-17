import json

# Convert json Prodigy output to Yolo8 input


def scale_points_by_hw(lst, width, height):
    "helper function to scale the x,y coordinates by object height and width"
    return [[elem[0] / width, elem[1] / height] for elem in lst]


def convert_prod_to_yolo(input_file_path, object_to_class_dict, output_file_path):
    """
    Takes a single prodigy label (dictionary, saved in json file format) and
    converts it to a yolo8 label (string of class and x,y coordinates, saved in txt format)

    Args:
    input_file_path: the pathname of the prodigy label
    object_to_class_dict: A dictionary for converting the names of classes in prodigy (eg, window, bathroom)
                        to a number for the yolo label - eg {'WINDOW': 5, 'DOOR': 0, 'OTHER_ROOM': 3, 'ROOM': 3, 'OTHER_DOOR': 0}
    output_file_path: the pathname for the yolo label"""

    # Open the JSON file for reading
    with open(input_file_path, "r") as json_file:
        # Parse the JSON data
        prod_label = json.load(json_file)

        output_list = []
        w = prod_label["width"]
        h = prod_label["height"]

        for i in range(0, len(prod_label["spans"])):
            # Get the class and output the corresponding number
            class_no = object_to_class_dict[prod_label["spans"][i]["label"]]

            # Get the polygon points, scale by the width and height of the image
            points = prod_label["spans"][i]["points"]
            scaled_list = scale_points_by_hw(points, w, h)
            flat_list = [item for sublist in scaled_list for item in sublist]

            # Combine scaled points with class number
            total_shape = [class_no] + flat_list
            output_list.append(total_shape)

        # Flatten from list to string
        final_format = [" ".join(map(str, item)) for item in output_list]

    # Output to text file
    with open(output_file_path, "w") as file:
        for item in final_format:
            file.write(item + "\n")

    return final_format
