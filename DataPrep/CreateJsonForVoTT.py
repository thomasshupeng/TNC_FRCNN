#
# Tool set to convert annotation to VoTT json file
#
# Shu Peng
# for full license information.
# ==============================================================================

import os
import sys
import json
import numpy as np

# BU (Beijing University) data set
data_set_name = 'BU'
train_img_width = 682
train_img_height = 512


def get_files_in_directory(directory, postfix=""):
    file_names = [s for s in os.listdir(directory) if not os.path.isdir(os.path.join(directory, s))]
    if not postfix or postfix == "":
        return file_names
    else:
        return [s for s in file_names if s.lower().endswith(postfix)]


def _load_annotation(img_path, class_dict):
    bboxes_path = img_path[:-4] + ".bboxes.tsv"
    labels_path = img_path[:-4] + ".bboxes.labels.tsv"
    # if no ground truth annotations are available, return None
    if not os.path.exists(bboxes_path) or not os.path.exists(labels_path):
        return None
    bboxes = np.loadtxt(bboxes_path, np.float32)

    # in case there's only one annotation and numpy read the array as single array,
    # we need to make sure the input is treated as a multi dimensional array instead of a list/ 1D array
    if len(bboxes.shape) == 1:
        bboxes = np.array([bboxes])

    with open(labels_path, 'r') as f:
        lines = f.readlines()
    labels = [s.replace("\r\n", "").replace("\n", "") for s in lines]

    for l in labels:
        if l not in class_dict:
            n = len(class_dict)
            class_dict[l] = n

    label_idxs = np.asarray([class_dict[l] for l in labels])
    label_idxs.shape = label_idxs.shape + (1,)
    annotations = np.hstack((bboxes, label_idxs))
    return annotations


def create_vott_json_annotation(parent_path, class_name, overwrite=False):
    vott_json_file = os.path.join(parent_path, class_name + '.json')
    if os.path.exists(vott_json_file) and not overwrite:
        print(
            "Warning: VoTT json file {} exists. Skip converting annotation for {}.".format(vott_json_file, class_name))
        return

    target_folder = os.path.join(parent_path, class_name)
    class_dict = {'__background__': 0}
    img_files = get_files_in_directory(target_folder, postfix='.jpg')
    img_files.sort()
    id_count = 0
    frm_count = 0
    # VoTT tool cannot handle the space in tag name, replace it with '_'.
    tag_name = class_name.replace(' ', '_', 10)
    frame = {}
    for img in img_files:
        annotation = _load_annotation(os.path.join(target_folder, img), class_dict)
        boxes = []
        if annotation is not None:
            for i in range(annotation.shape[0]):
                box = {
                    'x1': annotation[i][0],
                    'y1': annotation[i][1],
                    'x2': annotation[i][2],
                    'y2': annotation[i][3],
                    'name': int(i+1),
                    'width': train_img_width,
                    'height': train_img_height,
                    'type': 'Rectangle',
                    'id': id_count,
                    'tags': [tag_name]
                }
                boxes.append(box)
                id_count += 1
        frame[frm_count] = boxes
        frm_count += 1
    j_data = {"frames": frame,
              "framerate": "1",
              "inputTags": tag_name,
              "suggestiontype": "track",
              "scd": False,
              "visitedFrames": [i for i in range(frm_count)],
              "tag_colors": ["#10cffa"]
              }
    with open(vott_json_file, 'w') as f:
        json.dump(j_data, f)


def print_usage():
    print("Usage:")
    print("\tpython CreateJsonForVoTT.py [-f] [dir]")
    print("\t\t-f\tForce to overwrite existing JSON annotation file.")
    print("\t\t<dir>\t Folder contains image its BB.tsv, BB.label.tsv file.")


if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.abspath(__file__))
    data_set_path = os.path.join(abs_path, "..\\DataSets", data_set_name)
    data_set_path = os.path.realpath(data_set_path)
    if os.path.exists(data_set_path):
        print(data_set_path)
    else:
        print("DataSets folder not found.")

    args = sys.argv[1:]

    # overwrite_existing flag
    overwrite_existing = False
    if '-f' in args:
        overwrite_existing = True
        args.remove('-f')
    elif '-F' in args:
        overwrite_existing = True
        args.remove('-F')

    if len(args) == 1:
        class_folder = args[0]
        target_path = os.path.join(data_set_path, class_folder)
        if not os.path.exists(target_path):
            print("Error: directory not found -", target_path)
            print_usage()
            print("Done.")
    elif len(args) == 0:
        dir_names = [s for s in os.listdir(data_set_path) if
                     (os.path.isdir(os.path.join(data_set_path, s)) and (s[-7:] != '_output'))]
        for class_folder in dir_names:
            print("Creating VoTT json files for images under ", os.path.join(data_set_path, class_folder))
            create_vott_json_annotation(data_set_path, class_folder, overwrite_existing)
        print("Done.")
    else:
        print_usage()
