import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from tools.utils import GradCAM, show_cam_on_image, center_crop_img


def main():
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    model = models.resnet50(pretrained=True)
    target_layers = [model.layer4[-1]]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image

    def read_path(file_pathname):
        # 遍历该目录下的所有图片文件
        for filename in os.listdir(file_pathname):
            print(filename)
            img = cv2.imread(file_pathname + '/' + filename)
            ####change to gray
            # （下面第一行是将RGB转成单通道灰度图，第二步是将单通道灰度图转成3通道灰度图）
            assert os.path.exists(img), "file: '{}' dose not exist.".format(img)
            img = Image.open(img).convert('RGB')
            img = np.array(img, dtype=np.uint8)
            img_tensor = data_transform(img)
            input_tensor = torch.unsqueeze(img_tensor, dim=0)
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
            # target_category = 281  # tabby, tabby cat
            target_category = 552  # pug, pug-dog

            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                              grayscale_cam,
                                              use_rgb=True)
            plt.imshow(visualization)
            #####save figure
            cv2.imwrite('home/zystub/mmdetection/checkpoint' + "/" + filename, visualization)

            # 注意*处如果包含家目录（home）不能写成~符号代替

    # 必须要写成"/home"的格式，否则会报错说找不到对应的目录
    # 读取的目录
    read_path("home/zystub/coco/val2017")

    # img = center_crop_img(img, 224)

    # [C, H, W]

    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]




    plt.show()


if __name__ == '__main__':
    main()
