# TNC_FRCNN
Faster-RCNN for The Nature Conservancy Wildlife protection project

## Quick start
1. Install python, cntk and related packages
2. If you just want to play with single image prediction
    Download trained (with 10 classes) model from \\shpeng440\TNC_RawData\BU\FasterRCNN_Model
    copy the model (faster_rcnn_eval_AlexNet_e2e.model) to TNC_FRCNN\Detection\FasterRCNN\Output folder
2'. If you want to train your own model
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
3. Detection
    Go to \Detection
    'python DetectImg.py <image_full_path.jpg>

## Detection
Faster-RCNN training, evaluation and demo

## DataPrep
Scripts for 

## Pretrained Model
The base model for F-RCNN training

## DataSets
Contains groups of test/training set. default - BU (data from Beijing University)

