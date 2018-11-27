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
        self.cfg = merge_configs([detector_cfg, network_cfg, dataset_cfg, {'DETECTOR': 'FasterRCNN'}])
        self.name = "TNC_faster_rcnn_eval_AlexNet_e2e_native.model"
        self.model_type = ''
        self.model_path = ''
        self.model_file = ''
        self.en_zh_file = ''
        self.en_zh_dict = {}
        self.eval_model = None
        self.evaluator = None
        self.set_model_type(model_type)
        return

    def set_model_type(self, model_type):
        self.model_type = model_type
        self.model_path = os.path.join(os.path.join(os.path.dirname(__file__), 'models'), model_type)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model_file = os.path.join(self.model_path, self.name)
        self.cfg['DATA'].MAP_FILE_PATH = self.model_path
        self.cfg['DATA'].CLASS_MAP_FILE = os.path.join(self.model_path,
                                                       os.path.basename(self.cfg['DATA'].CLASS_MAP_FILE))
        self.en_zh_file = os.path.join(self.model_path, 'en_zh.txt')
        return

    def get_model_type(self):
        return self.model_type

    def get_cntk_version(self):
        return cntk.__version__

    def get_model_file(self):
        return self.model_file

    def get_class_map_file(self):
        return self.cfg['DATA'].CLASS_MAP_FILE

    def get_en_zh_map_file(self):
        return self.en_zh_file

    def load(self):
        prepare(self.cfg, use_arg_parser=False)
        print("Loading existing model from %s" % self.model_file)
        self.eval_model = load_model(self.model_file)
        from FasterRCNN.FasterRCNN_eval import FasterRCNN_Evaluator
        self.evaluator = FasterRCNN_Evaluator(self.eval_model, self.cfg)

        # Loading en_zh dictionary
        if os.path.exists(self.en_zh_file):
            with open(self.en_zh_file, 'r', encoding='utf-8', ) as f:
                for line in f.readlines():
                    en_name, zh_name = line.split(',', 2)
                    self.en_zh_dict[en_name.lower()] = zh_name.replace('\n', '')
        return

    def predict(self, image_buf, lang='en'):
        predictions = self._eval_single_image(image_buf)

        # Translated to zh
        if lang == 'zh':
            for p in predictions:
                tag = p['Tag'].lower()
                if tag in self.en_zh_dict:
                    p['Tag'] = self.en_zh_dict[tag]
        return predictions

    # Evaluates a single image using the provided model
    def _eval_single_image(self, img_buf):
        regressed_rois, cls_probs = self.evaluator.process_image_mem(img_buf)
        bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs, self.cfg)

        # write detection results to output
        fg_boxes = np.where(labels > 0)
        print("#bboxes: before nms: {}, after nms: {}, foreground: {}".format(len(regressed_rois), len(bboxes),
                                                                              len(fg_boxes[0])))
        fg = fg_boxes[0]
        predictions = []
        if len(fg) == 0:
            print("Nothing found in current image.")
            predictions.append({"TagId": 0, "Tag": 'Empty', "Probability": 1.0,
                                "Region": {"Left": 0.0, "Top": 0.0, "Width": 0.0, "Height": 0.0}})
        else:
            for i in fg:
                print("{:<12} (label: {:<2}), score: {:.3f}, box: {}".format(
                    self.cfg["DATA"].CLASSES[labels[i]], labels[i], scores[i], [int(v) for v in bboxes[i]]))
                left, top, right, bottom = [int(v) for v in bboxes[i]]
                predictions.append({"TagId": np.asscalar(labels[i]),
                                    "Tag": self.cfg["DATA"].CLASSES[labels[i]],
                                    "Probability": np.asscalar(scores[i]),
                                    "Region": {"Left": float(left / _image_width),
                                               "Top": float(top / _image_height),
                                               "Width": float((right - left) / _image_width),
                                               "Height": float((bottom - top) / _image_height)}})
        return predictions
