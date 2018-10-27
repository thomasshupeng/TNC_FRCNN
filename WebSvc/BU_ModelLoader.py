import os

# load configs for detector, base network and data set
from FasterRCNN.FasterRCNN_config import cfg as detector_cfg
# AlexNet base model
from utils.configs.AlexNet_config import cfg as network_cfg
# BU (Beijing University) data set
from utils.configs.BU_config import cfg as dataset_cfg
from utils.config_helpers import merge_configs

import numpy as np
import utils.od_utils as od
from FasterRCNN.FasterRCNN_train import prepare
from cntk import load_model
import cntk.device

_image_height = 512
_image_width = 682

cntk.device.try_set_default_device(cntk.device.cpu())

class FRCNN_Model:
    def __init__(self, model_type):
        # super().__init__()

        self.cfg = merge_configs([detector_cfg, network_cfg, dataset_cfg, {'DETECTOR': 'FasterRCNN'}])

        self._base_model_name = "faster_rcnn_eval_AlexNet_e2e_native.model"
        self.name = "BU_" + self._base_model_name
        self.model_path = os.path.join(os.path.join(os.path.dirname(__file__), 'models'), model_type)
        self.model_file = os.path.join(self.model_path, self.name)
        print("Model file:{}".format(self.model_file))
        self.cfg['DATA'].MAP_FILE_PATH = self.model_path
        print("Class map file path {}".format(self.cfg['DATA'].MAP_FILE_PATH))
        self.eval_model = None
        return

    def load(self):
        prepare(self.cfg, use_arg_parser=False)
        print("Loading existing model from %s" % self.model_file)
        self.eval_model = load_model(self.model_file)
        return

    def predict(self, image_path):
        predictions = self._eval_single_image(image_path)
        return predictions

    # Evaluates a single image using the provided model
    def _eval_single_image(self, img_path):
        from FasterRCNN.FasterRCNN_eval import FasterRCNN_Evaluator
        evaluator = FasterRCNN_Evaluator(self.eval_model, self.cfg)
        regressed_rois, cls_probs = evaluator.process_image(img_path)
        bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs, self.cfg)

        # write detection results to output
        fg_boxes = np.where(labels > 0)
        print("#bboxes: before nms: {}, after nms: {}, foreground: {}".format(len(regressed_rois), len(bboxes),
                                                                              len(fg_boxes[0])))
        fg = fg_boxes[0]
        predictions = []
        if len(fg) == 0:
            print("Nothing found in current image.")
            predictions.append({"TagId": 0, "Tag": 'Empty', "Probability": 1.0, "BBox": [0, 0, 0, 0]})
        else:
            for i in fg:
                print("{:<12} (label: {:<2}), score: {:.3f}, box: {}".format(
                    self.cfg["DATA"].CLASSES[labels[i]], labels[i], scores[i], [int(v) for v in bboxes[i]]))
                predictions.append({"TagId": np.asscalar(labels[i]),
                                    "Tag": self.cfg["DATA"].CLASSES[labels[i]],
                                    "Probability": np.asscalar(scores[i]),
                                    "BBox": [int(v) for v in bboxes[i]]})
        return predictions
