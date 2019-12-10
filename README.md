# Learning Rich Features at High-Speed for Single-Shot Object Detection

By Tiancai Wang†, Rao Muhammad Anwer†, Hisham Cholakkal, Fahad Shahbaz Khan, Yanwei Pang, Ling Shao

† denotes equal contribution

### Introduction
Single-stage object detection methods have received significant attention recently due to their characteristic real-time capabilities and  high detection accuracies. Generally, most existing single-stage detectors follow two common practices: they employ a network backbone that is pre-trained on ImageNet for the classification task and use a top-down feature pyramid representation for handling scale variations. Contrary to common pre-training strategy, recent works have demonstrated the benefits of training from scratch to reduce the task gap between classification and localization, especially at high overlap thresholds. However, detection models trained from scratch require significantly longer training time compared to their typical fine-tuning based counterparts. We introduce a single-stage detection framework that combines the advantages of both fine-tuning pre-trained models and training from scratch. Our framework constitutes a standard network that uses a pre-trained backbone and a parallel light-weight auxiliary network trained from scratch. Further, we argue that the commonly used top-down pyramid representation only focuses on passing high-level semantics from the top layers to bottom layers.  We introduce a bi-directional network that efficiently circulates both low-/mid-level and high-level semantic information in the detection framework. 
Experiments are performed on MS COCO and UAVDT datasets. Compared to the baseline, our detector achieives an absolute gain of 7.4% and 4.2% in average precision(AP) on MS COCO and UAVDT datasets, respectively us-ing VGG backbone. For a 300×300 input on the MS COCOtest set,  our detector with ResNet backbone surpasses existing single-stage detection methods for single-scale inference achieving 34.3 AP, while operating at an inference time of 19 milliseconds on a single Titan X GPU. 

## Installation
- Clone this repository. This repository is mainly based on [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), [RFBNet](https://github.com/ruinmessi/RFBNet).

```Shell
    LRFNet_ROOT=/path/to/clone/LRFNet
    git clone https://github.com/vaesl/LRFNet $LRFNet_ROOT
```
- The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.5/6 and [PyTorch]((http://pytorch.org/)) v0.3.1. 
NVIDIA GPUs are needed for testing. After install Anaconda, create a new conda environment, activate the environment and install pytorch0.3.1.

```Shell
    conda create -n LRFNet python=3.5
    source activate LRFNet
    conda install pytorch=0.3.1 torchvision -c pytorch
```

- Install opencv. 
```Shell
    conda install opencv
```

- Compile both [COCOAPI](https://github.com/cocodataset/cocoapi) and NMS:
```Shell
    cd $LRFNet_ROOT/
    ./make.sh
```

## Download
To evaluate the performance reported in the paper, COCO dataset as well as our trained models need to be downloaded.

### COCO Dataset
- Download the images and annotation files from coco website [coco website](http://cocodataset.org/#download). 
- Place the data (or create symlinks) to make the data folder like:

  ~~~
  ${$LRF_ROOT}
  |-- data
  `-- |-- coco
      `-- |-- annotations
          |   |-- instances_train2014.json
          |   |-- instances_val2014.json
          |   |-- image_info_test-dev2015.json
          `-- images
          |   |-- train2014
          |   |-- val2014
          |   |-- test2015
          `-- cache
  ~~~
  
  ### Trained Models
Please access to [Google Driver](https://drive.google.com/open?id=1_k6EOL1aIVtjI-3m5PEYC2Ri3ahUEi8T) 
or [BaiduYun Driver](https://pan.baidu.com/s/12yRVALNTc0ySdkHI7xzHYQ) to obtain our trained models for COCO dataset and put the models into corresponding directory(e.g. '~/weights/COCO/LRF_COCO_300/'). Note that the access code for the BaiduYun Driver is pzdx. 

## Evaluation
To check the performance reported in the paper:

```Shell
python test_LRF.py -d COCO -s 300 --trained_model /path/to/model/weights
```

where '-d' denotes datasets, COCO and '-s' represents image size, 300 or 512.

## Citation
Please cite our paper in your publications if it helps your research:

    @article{Wang2019LRF,
        title = {Learning Rich Features at High-Speed for Single-Shot Object Detection},
        author = {Tiancai Wang, Rao Muhammad Anwer, Hisham Cholakkal, Fahad Shahbaz Khan, Yanwei Pang, Ling Shao},
        booktitle = {ICCV},
        year = {2019}
    }
