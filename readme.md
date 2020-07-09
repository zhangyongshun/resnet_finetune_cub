# Fine Tune ResNet on CUB-200-2011 Dataset

# Introduction

This repo contains codes for fine tuning ResNet on CUB_200_2011 datasets.

Because ResNet_SE and ResNet_ED's  model files do not belong to me, so I remove them in the projects. 

The ResNet models provided by torchvision are available.

## Datasets

#### 1.CUB200-2011

&nbsp;&nbsp;CUB-200-2011 dataset has 11,788 images of 200 bird species. The project page
is as follows.

&nbsp;&nbsp;Detailed information as follows:

- Images are contained in the directory data/cub200/raw/images/,
  with 200 subdirectories.
- Format of images.txt: <image_id> <image_name>
- Format of train_test_split.txt: <image_id> <is_training_image>
- Format of classes.txt: <class_id> <class_name>
- Format of iamge_class_labels.txt: <image_id> <class_id>

## How to use

```
git clone https://github.com/zhangyongshun/resnet_finetune_cub.git
cd resnet_finetune_cub
#You need to modify the paths of model and data in utils/Config.py
python train.py --net_choice ResNet --model_choice 50 #ResNet50, use default setting to get the Acc reported in readme
```

## Results

There are some results as follows:  

![result](https://github.com/zhangyongshun/resnet_finetune_cub/raw/master/imgs/results.png)
