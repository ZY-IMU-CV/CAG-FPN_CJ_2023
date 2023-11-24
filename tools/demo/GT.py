# 本程序用于生成带gt的图片
import json
# from tkinter.filedialog import Open
import os
import cv2
import matplotlib.pyplot as plt

# 包含所有groud truth .json文件
file_name = '/home/zystub/data/coco/annotations/instances_val2017.json'
# with语句自动调用f.close()语句，保证无论是否出错都能正常关闭文件
with open(file_name, 'r') as f:
    all_info = json.load(f)  # 从.json文件中读取数据,包括每幅影像的 "id"：bbox(二维数组),字典数据类型
    print(len(all_info))  # 共有多少条记录/键值对
    img_id = list(all_info.keys())  # 返回字典所有的keys,并转化为list


def ch_boxes(labels):  # 多个bbox循环变换
    bboxes = []
    for label in labels:
        x_min = label[0]
        y_min = label[1]
        x_max = label[0] + label[2]
        y_max = label[1] + label[3]
        label = [x_min, y_min, x_max, y_max]  # [xmin,ymin,width,height]
        bboxes.append(label)
    return bboxes


def drawLabel(label_img_path, bboxes, img):  # 定义绘制函数,写入的图片路径，读取的图片，图片上的bboxes
    for item in bboxes:
        a = (item[0], item[1])  # a = (x_min, y_min)#左上角坐标(x1,y1)
        b = (item[2], item[3])  # b = (x_max, y_max)#右下角坐标（x2,y2）
        cv2.rectangle(img, a, b, (0, 255, 0), 2)  # 绘制框
    cv2.imwrite(label_img_path, img)  # 写入label_img
    # cv2.namedWindow(label_img_path, flags=cv2.WINDOW_AUTOSIZE)
    # cv2.imshow(label_img_path, img)  #画图显示
    # cv2.waitKey(0)
    return


data_path = '/home/zystub/mmdetection/result/GT/'
images_path = os.path.join(data_path, 'images/')  # 记得加反斜杠
label_path = os.path.join(data_path, 'label_img/')  # 新文件夹位置

for filename in img_id:
    pic_path = os.path.join(images_path, filename + '.png')  # 后缀记得加.!
    label_img_path = os.path.join(label_path, filename + '.png')  # label_img写入的路径
    bboxesraw = all_info[filename]  # 得到该张img对应的1-n个bbox
    bboxes = ch_boxes(bboxesraw)
    img = cv2.imread(pic_path)  # opencv 读取图片
    if len(bboxes) == 0:
        print('该图片无标签数据', filename + '.png')

    drawLabel(label_img_path, bboxes, img)
    print('正在处理%s.png' % filename)

print('共生成 %s 个labeled_img文件' % len(img_id))
