import cv2
import numpy as np
import torch
from directkeys import key_press
from grabscreen import grab_screen

from models.experimental import attempt_load
from utils.general import non_max_suppression, xyxy2xywh

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']


# 1-8 自动爬楼

weights = 'best.pt'  # yolo5 模型存放的位置
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
half = device.type != 'cpu'
conf_thres = 0.3  # NMS的置信度过滤
iou_thres = 0.2  # NMS的IOU阈值
classes = None
agnostic_nms = False  # 不同类别的NMS时也参数过滤
model = attempt_load(weights, device)

# img0 = cv2.imread("shiwu.jpg")
img0 = grab_screen((0, 0, 1280, 800))

img = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).to(device).unsqueeze(dim=0)
img = img.half() if half else img.float()  # uint8 to fp16/32~
img /= 255.0  # 0 - 255 to 0.0 - 1.0

pred = model(img, augment=False)[0]
# Apply NMS
det = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
det = det[0]

if det is not None and len(det):
    # # Rescale boxes from img_size to im0 size
    # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
    #
    # # Print results
    # for c in det[:, -1].unique():
    #     n = (det[:, -1] == c).sum()  # detections per class
    img_object = []
    cls_object = []
    for idx, (*xyxy, conf, cls) in enumerate(reversed(det)):
        # print(idx,conf,cls)
        # if True:  # Write to file
        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        #     with open('giraffe_label.txt', 'a') as f:
        #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
        cls = int(cls)
        img_object.append(xywh)  # [[位置]]
        cls_object.append(names[cls])  # [分类]
        if names[cls] == 'giraffe':
            hero_index = idx
        # 游戏
    thx = 30  # 捡东西时，x方向的阈值
    thy = 30  # 捡东西时，y方向的阈值
    attx = 150  # 攻击时，x方向的阈值
    atty = 50  # 攻击时，y方向的阈值
    skillDis = 800
    skillDis = 400

    if 'giraffe' in cls_object:
        hero_xywh = img_object[hero_index]
    # else:
    #     continue
    # 对屏幕中的monster 进行平均  最终一个
    if 'orange' in cls_object:
        # 打怪
        for idx, (c, box) in enumerate(zip(cls_object, img_object)):
            print(c, box)
            dis = ((hero_xywh[0] - box[0]) ** 2 + (hero_xywh[1] - box[1]) ** 2) ** 0.5
            # if dis < min_distance:
            #     monster_box = box
            #     monster_index = idx
            #     min_distance = dis
    elif 'material' in cls_object:
        if 'option' in cls_object:
            # 聚物捡东西  f2  xxx
        else:
            # 普通捡东西
    elif 'boss' in cls_object:
        # 打boss
        key_press('g')
    elif 'sinan' in cls_object:
        # 下一局 出现思南选择界面  选择1-8阶随机一个思南  鼠标移动点击
        #   中场清理背包
        # if 1-8 in cls_object:
        #     nextGame
        # else:
        #     shutDown and 通知爬楼完成,需要补货，清理背包  返回主界面