# Class Activation Map (CAM) generation and evaluation for Weakly Supervised Semantic SegmentationThis repository contains allows you to train a classification network which is based on ResNet-50.
Class Activation Map (CAM) can be generated from the trained model.

The mean Intersection-over-union (mIoU) is used to evaluate the accuracy of CAM.

Remember to change all files' paths to your own paths.

## Prerequisite
* Python 3.7, PyTorch 1.1.0, and more in requirements.txt
* PASCAL VOC 2012 devkit

## Usage

#### Install python dependencies
```
pip install -r requirements.txt
```
#### Download PASCAL VOC 2012 devkit
* Follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

All 4 executing files exist in the script directory.
#### train the ResNet50 classifier
```
python train_cls.py
```
* This file allows you to train the ResNet50 classifier.

#### generate CAM from the trained ResNet50 classifier  
```
python make_cam.py
```
* This classifier can be mannualy trained in the previous step, or use pretrained weights, by setting the args.train_cam_pass variable as True or False.
* An available pretrained weights is online: https://drive.google.com/file/d/1h8_LKaE70OZVKFaeR9JjzkG66mYFGcka/view?usp=sharing
* The make_cam.py will output the CAM in .npy file and basic pseudo-masks from the CAM in '.png'' file
* The generated basic pseudo-masks can be used to train a segmentation network.

#### evaluate the generated CAM 
```
python eval_cam_npy.py
```
* This file allows you to evaluate the generated CAM by mIoU metric.
* Different threshold can be assigned to the ResNet50 classifier.
* if you use the pretrained weights provided in the previous step, you will get mIoU scores around 46.8 in train set.

#### evaluate the generated basic pseudo-masks 
```
python eval_cam_png.py
```
* This file allows you to evaluate the generated basic pseudo-masks by mIoU metric.

#### (Optional) Colorizing the basic pseudo-masks
```
python colorize.py
```
* The pixel values of generated basic pseudo-masks in previous steps equal to the class number.
* This file allows you to colorize the pseudo-masks.
* 
## Acknowledgements

This repository is based on IRNet: https://github.com/jiwoon-ahn/irn. Thanks for their impressive work.
