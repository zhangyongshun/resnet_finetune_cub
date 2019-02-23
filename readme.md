# Fine Tune ResNet on CUB-200-2011 and Stanford Cars Datasets

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

#### 2.Stanford Cars

Stanford Cars datasets has 16185 images of 196 car species. The project page is as follows.

[Stanford Car](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)

Detailed information as follows:

- Directory car_ims contains total images (both training and testing images,  whose number is 16185)
- File car_nori.list contains information as follows: 


## How to use

```
git clone https://github.com/zhangyongshun/resnet_finetune_cub.git
cd base_model_finetune
python train.py [params]
```

## Results

There are some results as follows:  

![result](https://github.com/zhangyongshun/resnet_finetune_cub/raw/master/imgs/results.png)

![acc](https://github.com/zhangyongshun/resnet_finetune_cub/raw/master/imgs/Acc.png)
