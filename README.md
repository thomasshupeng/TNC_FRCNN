# TNC_FRCNN
Faster-RCNN for The Nature Conservancy Wildlife protection project

## Setup
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
## Quick start
1. If you just want to play with single image prediction
    Download trained (with 10 classes) model from \\shpeng440\TNC_RawData\BU\FasterRCNN_Model
    Copy the model (faster_rcnn_eval_AlexNet_e2e.model) to TNC_FRCNN\Detection\FasterRCNN\Output folder

2. If you want to train your own model
    Go to PretrainedModels folder to download AlexNet_ImageNet_Caffe.model
    Install VoTT - https://github.com/Microsoft/VoTT#installation
    Go to DataPrep
        2.1 create json for VoTT
        2.2 review in VoTT
        2.3 export to Faster_RCNN
        2.4 create training/test data
    Go to \Detection\FasterRCNN to train your model
        Rename Output\faster_rcnn_eval_AlexNet_e2e.model or move to other folder
        Delete Output\faster_rcnn_eval_AlexNet_e2e.model
        run 'python run_faster_rcnn.py'

3. Detect single image
    Go to \Detection
    ```
    python DetectImg.py <image_full_path.jpg>
    ```
## Sub-folder content

### Detection
Faster-RCNN training, evaluation and demo.

### DataPrep
Scripts for data preparation.

### Pretrained Model
The base model for F-RCNN training

### DataSets
Contains groups of test/training set. default - BU (data from Beijing University)

shpeng@microsoft.com
