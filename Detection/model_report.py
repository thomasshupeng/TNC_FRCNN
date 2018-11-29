import os
import sys
import numpy as np
import cntk
from sklearn.metrics import classification_report
from FasterRCNN.FasterRCNN_train import prepare
from FasterRCNN.FasterRCNN_eval import FasterRCNN_Evaluator
from utils.config_helpers import merge_configs
from utils.od_utils import filter_results


def get_configuration():
    # load configs for detector, base network and data set
    from FasterRCNN.FasterRCNN_config import cfg as detector_cfg
    # for VGG16 base model use:         from utils.configs.VGG16_config import cfg as network_cfg
    # for AlexNet base model use:       from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    # for Pascal VOC 2007 data set use: from utils.configs.Pascal_config import cfg as dataset_cfg
    # for the Grocery data set use:     from utils.configs.Grocery_config import cfg as dataset_cfg
    # from utils.configs.Grocery_config import cfg as dataset_cfg
    from utils.configs.BU_config import cfg as dataset_cfg
    return merge_configs([detector_cfg, network_cfg, dataset_cfg])


def read_test_images(map_file):
    img_files = []
    if os.path.exists(map_file):
        with open(map_file, 'r') as mf:
            for line in mf.readlines():
                ln, path, zero = line.split('\t', 10)
                img_files.append(path)
    return img_files


def read_test_classes(roi_file):
    img_classes = []
    if os.path.exists(roi_file):
        with open(roi_file, 'r') as rf:
            for line in rf.readlines():
                content = line.split('\t', 100)
                img_classes.append(int(content[4]))
    return img_classes


def read_zh_labels(en_zh_file):
    labels = []
    if os.path.exists(en_zh_file):
        with open(en_zh_file, 'r', encoding='utf-8') as ef:
            for line in ef.readlines():
                line = line.replace('\n', '')
                en_label, zh_label = line.split(',')
                labels.append(zh_label)
    return labels


if __name__ == '__main__':
    cfg = get_configuration()
    prepare(cfg, False)
    cntk.logging.set_trace_level(2)
    cntk.all_devices()
    # cntk.device.try_set_default_device(cntk.device.cpu())

    # reading test image and ground true
    classes = cfg['DATA'].CLASSES

    test_img_files = read_test_images(cfg['DATA'].TEST_MAP_FILE)
    test_img_root = os.path.realpath(os.path.join(cfg['DATA'].MAP_FILE_PATH))
    test_img_files = [os.path.join(test_img_root, i) for i in test_img_files]

    test_img_classes = read_test_classes(cfg['DATA'].TEST_ROI_FILE)

    test_size = len(test_img_files)
    if len(test_img_classes) != test_size:
        print("ERROR: size of images and ground true labels doesn't match")
        exit(-1)

    # load target_names
    zh_lables = read_zh_labels(os.path.join(cfg['DATA'].MAP_FILE_PATH, 'en_zh.txt'))
    label_dict = {}
    if len(classes) == len(zh_lables):
        for i in range(len(classes)):
            label_dict[classes[i]] = zh_lables[i]

    # loading model
    model = None
    model_path = cfg['MODEL_PATH']
    if os.path.exists(model_path):
        print("Loading existing model from %s" % model_path)
        model = cntk.load_model(model_path)
    evaluator = FasterRCNN_Evaluator(model, cfg)

    prediction_results = []
    progress_counter = 0
    for img_path in test_img_files:
        regressed_rois, cls_probs = evaluator.process_image(img_path)
        bboxes, labels, scores = filter_results(regressed_rois, cls_probs, cfg)
        fg_boxes = np.where(labels > 0)
        if len(fg_boxes[0]) > 0:
            prediction_results.append(labels[fg_boxes[0][0]])
        else:
            prediction_results.append(0)
        progress_counter += 1
        temp = "{:.2f}% - [{:d}/{:d}] images evaluated. {!s}".format(float((progress_counter / test_size) * 100),
                                                                     progress_counter, test_size, img_path)
        print(temp, end='\r')
        sys.stdout.flush()
    print("")

    if len(prediction_results) == test_size:
        classification_report(test_img_classes, prediction_results, classes)
    else:
        print("ERROR: number of prediction results is different from the total number of test images.")
