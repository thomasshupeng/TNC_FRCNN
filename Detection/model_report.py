import os
import sys
import numpy as np
import pandas as pd

import cntk
# from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

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
    zh_labels = []
    if os.path.exists(en_zh_file):
        with open(en_zh_file, 'r', encoding='utf-8') as ef:
            for line in ef.readlines():
                line = line.replace('\n', '')
                en_label, zh_label = line.split(',')
                zh_labels.append(zh_label)
    return zh_labels


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


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
    zh_labels = read_zh_labels(os.path.join(cfg['DATA'].MAP_FILE_PATH, 'en_zh.txt'))
    label_dict = {}
    if len(classes) == len(zh_labels):
        for i in range(len(classes)):
            label_dict[classes[i]] = zh_labels[i]

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
        if len(zh_labels) > 0:
            class_labels = zh_labels
        else:
            class_labels = classes

        '''

        # classification report
        # report = classification_report(test_img_classes, prediction_results, labels=list(range(1, len(classes))),
        #                               target_names=class_labels[1:]).encode('utf-8')
        # print(report)

        from sklearn.metrics import accuracy_score
        print("Accuracy score:")
        accuracy_score(test_img_classes, prediction_results)
        print("==============================================")
        print("")

        #print("Balanced accuracy score:")
        #from sklearn.metrics import balanced_accuracy_score
        #balanced_accuracy_score(test_img_classes, prediction_results)
        #print("==============================================")
        #print("")

        from sklearn.metrics import precision_score, recall_score
        print("Precision/recall score - Macro")
        precision_score(test_img_classes, prediction_results, labels=classes, average='macro')
        recall_score(test_img_classes, prediction_results, labels=classes, average='macro')
        print("==============================================")
        print("")
        print("Precision/recall score - Micro")
        precision_score(test_img_classes, prediction_results, labels=classes, average='micro')
        recall_score(test_img_classes, prediction_results, labels=classes, average='micro')
        print("==============================================")
        print("")
        print("Precision/recall score - Weighted")
        precision_score(test_img_classes, prediction_results, labels=classes, average='weighted')
        recall_score(test_img_classes, prediction_results, labels=classes, average='weighted')
        print("==============================================")
        print("")
        '''

        # confusion matrix
        cm = confusion_matrix(test_img_classes, prediction_results)
        np.set_printoptions(precision=2)
        df = pd.DataFrame(cm, index=classes, columns=classes)
        df.to_csv(os.path.join(os.getcwd(), "Confusion_matrix_en.csv"), encoding='utf-8')
        df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        df.to_csv(os.path.join(os.getcwd(), "Confusion_matrix_zh.csv"), encoding='utf-8')

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=classes,
                              title='Confusion matrix, without normalization')
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=classes, normalize=True,
                              title='Normalized confusion matrix')
        plt.show()

    else:
        print("ERROR: number of prediction results is different from the total number of test images.")
