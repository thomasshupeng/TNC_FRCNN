#
# Tool set for CNTK Faster-RCNN training
#
# Shu Peng
#
# ==============================================================================
'''
This script is used to generate the training data for CNTK Faster-RCNN
1. class_map.txt
2. train_img_file.txt
3. train_roi_file.txt
4. test_img_file.txt
5. test_roi_file.txt
'''

import os

# BU (Beijing University) data set
data_set_name = 'BU'


def getROIString(img_file, class_map):
    label_file = img_file[:-3] + 'bboxes.labels.tsv'

    class_string_list = []
    class_id_list = []
    with open(label_file, 'r') as f:
        class_string_list = f.read().split('\n')
    for s in class_string_list:
        if s:
            class_id_list.append(class_map[s.lower()])

    coordinate_file = img_file[:-3] + 'bboxes.tsv'
    co_list = []
    with open(coordinate_file, 'r') as f:
        co_list = f.read().split('\n')

    roi_string = ''
    for i in range(len(class_id_list)):
        roi_string = roi_string + co_list[i] + "\t" + str(class_id_list[i]) + "\t"

    return roi_string


if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.abspath(__file__))
    data_set_path = os.path.join(abs_path, "..\\DataSets", data_set_name)

    class_map = {'__background__': 0}
    class_id = 1
    train_img_list = []
    test_img_list = []
    for root, dirs, files in os.walk(data_set_path):
        if root[-7:] == '_output':
            base_name = os.path.basename(root)
            class_name = base_name.replace('_output', '').replace(' ', '_', 10)
            if class_name not in class_map:
                class_map[class_name] = class_id
                class_id += 1
            positive_folder = os.path.join(root, 'positive')
            for f in os.listdir(positive_folder):
                if f[-4:] == '.JPG':
                    train_img_list.append(base_name + '\\positive\\' + f)

            test_folder = os.path.join(root, 'testImages')
            for f in os.listdir(test_folder):
                if f[-4:] == '.JPG':
                    test_img_list.append(base_name + '\\testImages\\' + f)

    # Creating class_map.txt
    class_map_file = os.path.join(data_set_path, 'class_map.txt')
    with open(class_map_file, 'w') as f:
        class_id_map = {}
        for k in class_map:
            class_id_map[class_map[k]] = k
        for k in class_id_map:
            f.write("{!s}\t{:d}\n".format(class_id_map[k], k))
        f.flush()
        f.close()

    low_case_class_map = {}
    for k in class_map:
        low_case_class_map[k.lower()] = class_map[k]
    # Creating train_img_file.txt and train_roi_file.txt
    train_img_file = os.path.join(data_set_path, 'train_img_file.txt')
    train_roi_file = os.path.join(data_set_path, 'train_roi_file.txt')
    print("Creating {}".format(train_img_file))
    print("Creating {}".format(train_roi_file))
    with open(train_img_file, 'w') as tf, open(train_roi_file, 'w') as rf:
        count = 0
        for img in train_img_list:
            tf.write('{:d}\t{!s}\t0\n'.format(count, img))
            region_data_string = getROIString(os.path.join(data_set_path, img), low_case_class_map)
            rf.write("{:d} |roiAndLabel {!s}\n".format(count, region_data_string))
            count += 1
        tf.flush()
        tf.close()
        rf.flush()
        rf.close()

    # Creating test_img_file.txt and test_roi_file.txt
    test_img_file = os.path.join(data_set_path, 'test_img_file.txt')
    test_roi_file = os.path.join(data_set_path, 'test_roi_file.txt')
    print("Creating {}".format(test_img_file))
    print("Creating {}".format(test_roi_file))
    with open(test_img_file, 'w') as tf, open(test_roi_file, 'w') as rf:
        count = 0
        for img in test_img_list:
            tf.write('{:d}\t{!s}\t0\n'.format(count, img))
            region_data_string = getROIString(os.path.join(data_set_path, img), low_case_class_map)
            rf.write("{:d} |roiAndLabel {!s}\n".format(count, region_data_string))
            count += 1
        tf.flush()
        tf.close()
        rf.flush()
        rf.close()
