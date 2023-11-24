环境配置：

①Python3.6/3.7/3.8

②Pytorch1.7.1(注意：必须是1.6.0或以上，因为使用官方提供的混合精度训练1.6.0后才支持)

文件结构：

├── backbone: 特征提取网络

├── configs: Faster R-CNN网络（包括Fcos等基准模型）

├── tools: 训练验证相关模块

├── faster_rcnn_r50_fpn.py: 以resnet50+CAGFPN做为backbone进行训练

├── train_multi_GPU.py: 针对使用多GPU的用户使用

预训练权重下载地址：

ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth ，注意，下载的预训练权重记得要重命名，比如在train_resnet50_fpn.py中读取的是fasterrcnn_resnet50_fpn_coco.pth文件， 不是fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

数据集下载（默认使用的是COCO格式的数据集）

COCO官网地址：https://cocodataset.org/

这里以下载coco2017数据集为例，主要下载三个文件：

2017 Train images [118K/18GB]：训练过程中使用到的所有图像文件

2017 Val images [5K/1GB]：验证过程中使用到的所有图像文件

2017 Train/Val annotations [241MB]：对应训练集和验证集的标注json文件

训练方法：
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py work_dirs/..../epoch_12.pth


