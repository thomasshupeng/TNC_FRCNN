# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
#
# Modified by: Shu Peng to use Faster-RCNN
#

import os, sys
import numpy as np
import utils.od_utils as od
from utils.config_helpers import merge_configs


def get_configuration(detector_name):
    # load configs for detector, base network and data set
    from FasterRCNN.FasterRCNN_config import cfg as detector_cfg
    # AlexNet base model
    from utils.configs.AlexNet_config import cfg as network_cfg
    # BU (Beijing University) data set
    from utils.configs.BU_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg, {'DETECTOR': detector_name}])


if __name__ == '__main__':
    # Currently hard-coded detectors: 'FasterRCNN'
    args = sys.argv
    if len(args) != 2:
        print("Please provide a FULL path to the image (.jpg) file you want to detect. Usage:")
        print("    python DetectionDemo.py <full_path_to_image.jpg>")
        exit()
    img_path = args[1]
    if not os.path.exists(img_path):
        print("Error: file not found -", img_path)
        exit()

    #  Hard-coded to use Faster-RCNN
    detector_name = 'FasterRCNN'
    cfg = get_configuration(detector_name)

    # Force to load trained model
    cfg['CNTK'].MAKE_MODE = True

    eval_model = od.train_object_detector(cfg)

    # detect objects in single image
    regressed_rois, cls_probs = od.evaluate_single_image(eval_model, img_path, cfg)
    bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs, cfg)

    # write detection results to output
    fg_boxes = np.where(labels > 0)
    print("#bboxes: before nms: {}, after nms: {}, foreground: {}".format(len(regressed_rois), len(bboxes),
                                                                          len(fg_boxes[0])))
    for i in fg_boxes[0]: print("{:<12} (label: {:<2}), score: {:.3f}, box: {}".format(
        cfg["DATA"].CLASSES[labels[i]], labels[i], scores[i], [int(v) for v in bboxes[i]]))

    # visualize detections on image
    od.visualize_results(img_path, bboxes, labels, scores, cfg)

    # measure inference time
    #od.measure_inference_time(eval_model, img_path, cfg, num_repetitions=100)
