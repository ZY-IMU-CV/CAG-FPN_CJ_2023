import torchvision
import argparse
import mmcv
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models, transforms
from mmdet.models.necks.fpn import FPN
from mmdet.models.necks.dacam_module import _ChannelAttentionModule

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        # FeatureExtractor(model.layer4, ["2"])
        self.model = model  # model.layer4
        self.target_layers = target_layers  # ["2"]
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)  # torch.Size([1, 2048, 7, 7])

    def __call__(self, x):  # torch.Size([1, 1024, 14, 14])
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            # '0'、 '1'、 '2'
            x = module(x)
            if name in self.target_layers:  # ["2"]
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x  # 单个元素的列表torch.Size([1, 2048, 7, 7]) torch.Size([1, 2048, 7, 7])


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        # ModelOutputs(model, model.layer4, ["2"])
        self.model = model  # model
        self.feature_module = feature_module  # model.layer4
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)
        # FeatureExtractor(model.layer4, ["2"])

    def get_gradients(self):
        return self.feature_extractor.gradients  # 只有一个元素列表类型 torch.Size([1, 2048, 7, 7])

    def __call__(self, x):
        # target_activations = []  # 这行代码没有意义
        for name, module in self.model._modules.items():  # 遍历有序字典
            # 'conv1' 'bn1' 'relu' 'maxpool' 'layer1'
            # 'layer2' 'layer3' 'layer4'  'avgpool' 'fc'
            if module == self.feature_module:  # model.layer4
                target_activations, x = self.feature_extractor(x)
                # torch.Size([1, 1024, 14, 14]) -> torch.Size([1, 2048, 7, 7])
            elif "avgpool" in name.lower():  # 'avgpool'
                x = module(x)  # torch.Size([1, 2048, 7, 7]) -> torch.Size([1, 2048, 1, 1])
                x = x.view(x.size(0), -1)  # torch.Size([1, 2048])
            else:
                x = module(x)

        return target_activations, x  # 列表torch.Size([1, 2048, 7, 7]), torch.Size([1, 1000])


def preprocess_image(img):
    '''将numpy的(H, W, RGB)格式多维数组转为张量后再进行指定标准化,最后再增加一个batchsize维度后返回'''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def show_cam_on_image(img, mask):
    '''将mask图片转化为热力图,叠加到img上,再返回np.uint8格式的图片.'''
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        # GradCam(model=model, feature_module=model.layer4, \
        #                target_layer_names=["2"], use_cuda=args.use_cuda)
        self.model = model  # model
        self.feature_module = feature_module  # model.layer4
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)
        # ModelOutputs(model, model.layer4, ["2"])

    def forward(self, input_img):  # 似乎这个方法没有使用到,注释掉之后没有影响,没有被执行到
        print("林麻子".center(50, '-'))  # 这行打印语句用来证明,该方法并没有被调用执行.
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()  # torch.Size([1, 3, 224, 224])

        features, output = self.extractor(input_img)  # 保存中间特征图的列表, 以及网络最后输出的分类结果
        # 列表[torch.Size([1, 2048, 7, 7])], 张量:torch.Size([1, 1000])
        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())  # 多维数组展平后最大值的索引
            # <class 'numpy.int64'>  243

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)  # 独热编码,shape:(1, 1000)
        one_hot[0, target_category] = 1  # 独热编码  shape (1, 1000) # one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(False)  # torch.Size([1, 1000]) # requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        loss = torch.sum(
            one_hot * output)  # tensor(9.3856, grad_fn=<SumBackward0>) one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()  # 将模型的所有参数的梯度清零.
        self.model.zero_grad()  # 将模型的所有参数的梯度清零.
        loss.backward()  # one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[0].cpu().data.numpy()  # shape:(1, 2048, 7, 7)  # 顾名思义,梯度值
        # 注: self.extractor.get_gradients()[-1]返回保存着梯度的列表,[-1]表示最后一项,即最靠近输入的一组特征层上的梯度
        target = features[-1]  # torch.Size([1, 2048, 7, 7])  列表中的最后一项,也是唯一的一项,特征图
        target = target.cpu().data.numpy()[0, :]  # shape: (2048, 7, 7)

        weights = np.mean(grads_val, axis=(2, 3))[0, :]  # shape: (2048,)  计算每个特征图上梯度的均值,以此作为权重
        cam = np.zeros(target.shape[1:], dtype=np.float32)  # 获得零矩阵 shape: (7, 7)

        for i, w in enumerate(weights):  # 迭代遍历该权重
            cam += w * target[i, :, :]  # 使用该权重,对特征图进行线性组合

        cam = np.maximum(cam, 0)  # shape: (7, 7) # 相当于ReLU函数
        # print(type(input_img.shape[3:1:-1]),'cxq林麻子cxq',input_img.shape[3:1:-1])
        # print(type(input_img.shape[2:]),'cxq林麻子cxq',input_img.shape[2:])
        cam = cv2.resize(cam, input_img.shape[3:1:-1])  # shape: (224, 224) # 这里要留意传入的形状是(w,h) 所以这里切片的顺序是反过来的
        cam = cam - np.min(cam)  # shape: (224, 224)  # 以下两部是做归一化
        cam = cam / np.max(cam)  # shape: (224, 224)  # 归一化,取值返回是[0,1]
        return cam  # shape: (224, 224) 取值返回是[0,1]


class GuidedBackpropReLU(Function):
    '''特殊的ReLU,区别在于反向传播时候只考虑大于零的输入和大于零的梯度'''

    '''
    @staticmethod
    def forward(ctx, input_img):  # torch.Size([1, 64, 112, 112])
        positive_mask = (input_img > 0).type_as(input_img)  # torch.Size([1, 64, 112, 112])
        # output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        output = input_img * positive_mask  # 这行代码和上一行的功能相同
        ctx.save_for_backward(input_img, output)
        return output  # torch.Size([1, 64, 112, 112])
    '''

    # 上部分定义的函数功能和以下定义的函数一致
    @staticmethod
    def forward(ctx, input_img):  # torch.Size([1, 64, 112, 112])
        output = torch.clamp(input_img, min=0.0)
        # print('函数中的输入张量requires_grad',input_img.requires_grad)
        ctx.save_for_backward(input_img, output)
        return output  # torch.Size([1, 64, 112, 112])

    @staticmethod
    def backward(ctx, grad_output):  # torch.Size([1, 2048, 7, 7])
        input_img, output = ctx.saved_tensors  # torch.Size([1, 2048, 7, 7]) torch.Size([1, 2048, 7, 7])
        # grad_input = None  # 这行代码没作用
        positive_mask_1 = (input_img > 0).type_as(grad_output)  # torch.Size([1, 2048, 7, 7])  输入的特征大于零
        positive_mask_2 = (grad_output > 0).type_as(grad_output)  # torch.Size([1, 2048, 7, 7])  梯度大于零
        # grad_input = torch.addcmul(
        #                             torch.zeros(input_img.size()).type_as(input_img),
        #                             torch.addcmul(
        #                                             torch.zeros(input_img.size()).type_as(input_img),
        #                                             grad_output,
        #                                             positive_mask_1
        #                             ),
        #                             positive_mask_2
        # )
        grad_input = grad_output * positive_mask_1 * positive_mask_2  # 这行代码的作用和上一行代码相同
        return grad_input


class GuidedBackpropReLUModel:
    '''相对于某个类别(默认是最大置信度对应的类别)的置信度得分,计算输入图片上的梯度,并返回'''

    def __init__(self, model, use_cuda):
        # GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            '''递归地将模块内的relu模块替换掉用户自己定义的GuidedBackpropReLU模块 '''
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':  # module对象所属的类,该类的名称
                    # print('成功替换...')  # 验证确实得到了替换
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    # def forward(self, input_img):
    #     return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        '''相对于某个类别(默认是最大置信度对应的类别)的置信度得分,计算输入图片上的梯度,并返回'''
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)  # torch.Size([1, 3, 224, 224])
        output = self.model(input_img)  # torch.Size([1, 1000])
        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())  # 243

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)  # (1, 1000)
        one_hot[0, target_category] = 1  # one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(False)  # torch.Size([1, 1000])
        # one_hot = torch.from_numpy(one_hot).requires_grad_(True)  # 这个张量不需要计算梯度
        if self.cuda:
            one_hot = one_hot.cuda()

        loss = torch.sum(one_hot * output)
        loss.backward()  # one_hot.backward(retain_graph=True)

        img_grad = input_img.grad.cpu().data.numpy()  # shape (1, 3, 224, 224)
        img_grad = img_grad[0, :, :, :]  # shape (3, 224, 224)

        return img_grad  # shape (3, 224, 224)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='image/000000210299.jpg',
                        # default='./examples/1.jpg', # './examples/both.png'
                        help='Input image path')  # default='./examples/both.png',
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    '''先作标准化处理,然后做变换y=0.1*x+0.5,限定[0,1]区间后映射到[0,255]区间'''
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    # 默认情况下: args.image_path = './examples/both.png',
    # 默认情况下: args.use_cuda = False,
    # model = models.resnet50(pretrained=True)
    config = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    cfg = mmcv.Config.fromfile(config)  # 调用mmcv库解析
    model =FPN(
        in_channels=[3],out_channels=512,num_outs=5)
    # model=_ChannelAttentionModule()
    # grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       # target_layer_names=["2"], use_cuda=args.use_cuda)
    grad_cam = GradCam(model=model, feature_module=torchvision.models.detection.fasterrcnn_resnet50_fpn, \
                       target_layer_names=["4"], use_cuda=args.use_cuda)
    model.load_state_dict(model.state_dict(),'../work_dirs/daC5/epoch_12.pth')
    img = cv2.imread(args.image_path, 1)  # 读取图片文件 (H, W, BGR)
    # If set, always convert image to the 3 channel BGR color image.
    img = np.float32(img) / 255  # 转为float32类型,范围是[0,1]
    # Opencv loads as BGR:
    img = img[:, :, ::-1]  # BGR格式转换为RGB格式 shape: (224, 224, 3) 即(H, W, RGB)
    input_img = preprocess_image(img)  # torch.Size([1, 3, 224, 224])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None
    grayscale_cam = grad_cam(input_img, target_category = None)  # shape: (224, 224)

    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    # shape: (224, 224) # 这里要留意传入的形状是(w,h)  其实以上这行代码不需要执行,暂且先留着

    cam = show_cam_on_image(img, grayscale_cam)  # shape: (224, 224, 3)
    cv2.imwrite("cam.jpg", cam)  # 保存图片

    # -----------------------------------------------------------------------------------

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # input_img.grad.zero_()  # AttributeError: 'NoneType' object has no attribute 'zero_'
    gb = gb_model(input_img, target_category=None)  # shape: (3, 224, 224) 相对于输入图像的梯度
    gb = gb.transpose((1, 2, 0))  # 调整通道在维度中的位置顺序 shape:(224, 224, 3)  相对于输入图像的梯度

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])  # shape:(224, 224, 3) # 由多个单通道的数组创建一个多通道的数组
    cam_gb = deprocess_image(cam_mask * gb)  # shape: (224, 224, 3)
    cv2.imwrite('cam_gb.jpg', cam_gb)  # 保存图片

    gb = deprocess_image(gb)  # shape: (224, 224, 3)
    cv2.imwrite('gb.jpg', gb)  # 保存图片

    # -----------------------------------------------------------------------------------

    # cv2.imwrite("cam.jpg", cam)  # 保存图片
    # cv2.imwrite('gb.jpg', gb)  # 保存图片
    # cv2.imwrite('cam_gb.jpg', cam_gb)  # 保存图片

# 运行程序: python gradcam.py --image-path 1.jpg
# 运行程序: python gradcam.py --image-path ./examples/both.png



