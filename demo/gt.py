from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import os,cv2
import mmcv
config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'#_1 256
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '../work_dirs/FPN/latest.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
# img = '../data/coco/val2017/000000000139.jpg'
img = 'demo.jpg'
# mmcv.imwrite(img,"../result/GT/")
result = inference_detector(model, img)
# show the results
show_result_pyplot(model, img, result)

