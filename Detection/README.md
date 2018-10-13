# CNTK Examples: TNC_FRCNN/Detection

## Overview

This folder contains a set of scripts to build object detectors for TNC wildlife protection project.

They are customized scripts based on the examples given in CNTK/Examples/Image/Detection.

## Running the example

### Setup Python environment

1. Python 3.5 under Windows and Python 3.5/3.6 under Linux

2. CNTK 2.6 Python environment
CPU-Only:
```
pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.6-cp35-cp35m-win_amd64.whl
```
or, GPU:
```
pip install https://cntk.ai/PythonWheel/GPU/cntk_gpu-2.6-cp35-cp35m-win_amd64.whl
```
3. Install the following additional packages:

```
pip install opencv-python easydict pyyaml future
```

### Getting the trained model

Contact shpeng to copy \\shpeng440\TNC_RawData\BU\FasterRCNN_Model\faster_rcnn_eval_AlexNet_e2e.model to Detection/FasterRCNN/Output

### Detect single image

`python DetectImage.py <path_to_imge.jpg>`
