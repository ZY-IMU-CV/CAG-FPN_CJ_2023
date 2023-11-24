import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision import models
from torchvision import transforms
from tools.utils import GradCAM, show_cam_on_image, center_crop_img
import glob as gb


def main():
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    model = models.resnet50(pretrained=True)
    target_layers = [model.layer4[-1]]

    # model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # target_layers = [model.backbone]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image


    img_path = "image/000000000139.jpg"
    # for path in img_path:
    #     img_path = cv2.imread(path)

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)


    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    # target_category = 281  # tabby, tabby cat
    target_category = 548# pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    #     img_savepath = "home/zystub/mmdetection/checkpoint"
    #     savepath = os.path.join(img_savepath)
    #     cv2.imwrite(savepath, visualization)
    plt.show()


if __name__ == '__main__':
    main()
