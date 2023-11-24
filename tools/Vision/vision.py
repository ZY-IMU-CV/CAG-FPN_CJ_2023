import warnings
import  argparse
import mmcv
from mmdet.models.necks.fpn import FPN
from mmdet.models.necks.dacam_module import _ChannelAttentionModule
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import AblationCAM, EigenCAM
from tools.Vision.ablation_layer import AblationLayerFasterRCNN
from tools.Vision.model_targets import FasterRCNNBoxScoreTarget
from tools.Vision.reshape_transforms import fasterrcnn_reshape_transform
from tools.Vision.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image

def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices


def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image


coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


import requests
import torch
import time
import cv2
import os
import torchvision
from tools.Vision.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision import transforms
from tools.cam.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from tools.cam.resnet_fpn_model import resnet50_fpn_backbone
from tools.cam.draw_box_utils import draw_objs

from PIL import Image
# im    return modelage_path = "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png"
img_path = "../image/000000216497.jpg"
# for path in img_path:
#     img_path = cv2.imread(path)
# image = np.array(Image.open(requests.get(image_path, stream=True).raw))
# image_float_np = np.float32(image) / 255
# define the torchvision image transforms


img = cv2.imread(img_path, 1)[:, :, ::-1]   # 1是读取rgb
                                                 #imread返回从指定路径加载的图像
img = cv2.imread(img_path, 1) #imread()读取的是BGR格式
image_float_np = np.float32(img) / 255

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

input_tensor = transform(img)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = input_tensor.to(device)
# Add a batch dimension:
input_tensor = input_tensor.unsqueeze(0)
# config = '../../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# cfg = mmcv.Config.fromfile(config)
# model = FPN(in_channels=[3],out_channels=512,num_outs=5)
# model = _ChannelAttentionModule()



# load train weights
# model_urls = {
#     'fasterrcnn_resnet50_fpn_coco':
#         'https://github.com/Ccchang-jie/work_dirs/blob/master/latest.pth',
#
# }
# assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
# model.load_state_dict(torch.load(weights_path, map_location='cpu')["model"])

# model.to(device)



model = fasterrcnn_resnet50_fpn(pretrained=True)
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model = torchvision.models.resnet50(pretrained=True)
# model.load_state_dict(model.state_dict(),model_urls)

model.eval().to(device)

# Run the model and display the detections
boxes, classes, labels, indices = predict(input_tensor, model, device, 0.9)
image = draw_boxes(boxes, labels, classes, img)

target_layers = [model.backbone]

cam = EigenCAM(model,
               target_layers,
               use_cuda=torch.cuda.is_available(),
               reshape_transform=fasterrcnn_reshape_transform
               )

targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]

grayscale_cam = cam(input_tensor=input_tensor,  targets=targets)
grayscale_cam = grayscale_cam[0, :]
# cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
# And lets draw the boxes again:
# image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
# Image.fromarray(image_with_bounding_boxes)
# plt.imshow(image_with_bounding_boxes)
# plt.show()

def renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1]
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    images = []
    for x1, y1, x2, y2 in boxes:
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        images.append(img)

    renormalized_cam = np.max(np.float32(images), axis=0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, eigencam_image_renormalized)
    return image_with_bounding_boxes

plt.imshow(renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam))
# cv2.imwrite('home/zystub/mmdetection/checkpoint' , renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam))
plt.show()



# Image.fromarray(image_with_bounding_boxes)
# Show the image:
# Image.fromarray(image)
# plt.imshow(image)
# plt.show()