train.py和utils.py用于图像分类任务，即imagenet数据集训练。
train1.py和utils1.py用于场景分割任务，即driveseg数据集训练。

需要下载预训练权重ViT-B_16.h5：
链接: https://pan.baidu.com/s/1ro-6bebc8zroYfupn-7jVQ  密码: s9d9

实验环境：
Python = 3.7.6
TensorFlow = 2.7.0
CUDA = 11.2
cuDNN= 8.9.7
GPU = Tesla V100-SXM2-32GB×4

数据集下载：
imagenet网址：https://image-net.org/index.php
driveseg网址：https://ieee-dataport.org/open-access/mit-driveseg-manual-dataset