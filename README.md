# RetinaNet
### pytorch 1.3 and python 3.6 is supported
A PyTorch implementation of RetinaNet, with support for training, inference and evaluation.

## Introduction
The method of RetinaNet was used to perform defect detection on NEU surface defect database, and We adopted data enhancement methods such as random clipping, flipping and color enhancement. Finally achieved a satisfactory result.


## Installation
##### Clone and install requirements
    $ git clone https://github.com/Gmy12138/RetinaNet.git
    $ cd RetinaNet/
    $ pip install -r requirements.txt

##### Download pretrained weights
    $ cd scripts/
    $ run get_state_dict.py

##### Download NEU-DET dataset
    $ Download address    http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
    $ cd data/
    $ Put the dataset in the data folder
    
## Test
Evaluates the model on NEU-DET test.


| Model      |Image Size|   Data Enhancement      | mAP (min. 50 IoU) |
|:----------:|:--------:|:-----------------------:|:-----------------:|
| RetinaNet  |300       |      YES                | 61.3              |



## Inference
Uses pretrained weights to make predictions on images. The ResNet-50 measurement marked shows the inference time of this implementation on GPU 2080ti.


| Model        |Backbone      |  Image Size     | GPU      | FPS      | parameters (10<sup>6</sup>)|FLOPs (10<sup>9</sup>)|
|:------------:|:------------:|:---------------:|:--------:|:--------:|:--------------------------:|:--------------------:|
| RetinaNet    | ResNet-50   |     300         | 2080ti   |   27      |            35.84           |     37.43            | 






