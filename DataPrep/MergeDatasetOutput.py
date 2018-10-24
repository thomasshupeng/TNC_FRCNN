# ==============================================================================
# Tool to copy VoTT output back to its original folder
# MergeDatasetOutput.py <DataSet_Name>
# Shu Peng
# ==============================================================================

import os
import sys
import shutil


def copy_folder(src, dst):
    if not os.path.exists(src):
        print("Error: source path not found!")
        print("source path: ", src)
        return
    if not os.path.isdir(src) or not os.path.isdir(dst):
        print("Error: source/destination is not a folder")
        return
    src_list = [os.path.join(src, i) for i in os.listdir(src)]
    src_list = [i for i in src_list if os.path.isfile(i)]
    for f in src_list:
        file_name = os.path.basename(f)
        dst_name = os.path.join(dst, file_name)
        print("{} => {}".format(f, dst_name))
        shutil.copyfile(f, dst_name)
    # Iterating through the sub folder
    sub_folder = [os.path.join(src, i) for i in os.listdir(src)]
    sub_folder = [i for i in sub_folder if os.path.isdir(i)]
    for s in sub_folder:
        print("Copy sub folder ", s)
        copy_folder(s, dst)
    return


if __name__ == '__main__':
    data_sets_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\\DataSets")

    args = sys.argv
    if len(args) != 2:
        print("Please provide DataSet name under {}.".format(data_sets_root))
        print("    python MergeDatasetOutput.py <DataSet_Name>")
        exit()
    data_set_name = args[1]
    data_set_path = os.path.join(data_sets_root, data_set_name)
    if not os.path.exists(data_set_path):
        print("Error: path not found -", data_set_path)
        exit()

    postfix ='_output'
    postfix_len = len(postfix)
    output_folders = [os.path.join(data_set_path, i) for i in os.listdir(data_set_path)]
    output_folders = [os.path.realpath(i) for i in output_folders if os.path.isdir(i) and i[-postfix_len:] == postfix ]

    for f in output_folders:
        copy_folder(f, f[:-postfix_len])

