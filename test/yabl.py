import random
import cv2
import numpy as np
import torch
import time
import directkeys
from directkeys import (key_press, ReleaseKey, PressKey)
from direction_move import move
from grabscreen import grab_screen
from models.experimental import attempt_load
from utils.general import non_max_suppression, xyxy2xywh
from getkeys import key_check

direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}  # 上下左右的键码
names = ['monster', 'hero', 'boss', 'option']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
weights = 'best.pt'  # yolo5 模型存放的位置
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
half = device.type != 'cpu'
model = attempt_load(weights, device)
if half:
    model.half()  # to FP16
conf_thres = 0.3  # NMS的置信度过滤
iou_thres = 0.2  # NMS的IOU阈值
classes = None
agnostic_nms = False  # 不同类别的NMS时也参数过滤
view_img = True
frame = 0
fs = 20
action_cache = None  # 动作标记
paused = False


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


while True:

    if not paused:
        frame += 1
        if frame % fs == 0:
            img0 = grab_screen((0, 0, 1200, 900))

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

                img_object = []
                cls_object = []
                for idx, (*xyxy, conf, cls) in enumerate(reversed(det)):

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    cls = int(cls)
                    img_object.append(xywh)  # [[位置]]
                    cls_object.append(names[cls])  # [分类]
                    if names[cls] == 'hero':
                        hero_index = idx
                    if view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)
                    # 游戏
                thx = 30  # 捡东西时，x方向的阈值
                thy = 30  # 捡东西时，y方向的阈值
                attx = 150  # 攻击时，x方向的阈值
                atty = 50  # 攻击时，y方向的阈值
                skillDis = 800
                skillDis = 400

                if 'hero' in cls_object:
                    hero_xywh = img_object[hero_index]
                else:
                    continue
                # 对屏幕中的monster 进行平均  最终一个
                if 'monster' in cls_object:
                    min_distance = float("inf")
                    # 遍历屏幕上的所有怪，找到距离最小的那个怪
                    for idx, (c, box) in enumerate(zip(cls_object, img_object)):
                        print(c)
                        if c == 'hero': continue
                        dis = ((hero_xywh[0] - box[0]) ** 2 + (hero_xywh[1] - box[1]) ** 2) ** 0.5
                        print(dis)
                        if dis < min_distance:
                            monster_box = box
                            monster_index = idx
                            min_distance = dis
                    print(hero_xywh, monster_box)
                    print(abs(hero_xywh[0] - monster_box[0]), hero_xywh[1] - monster_box[1])
                    if abs(hero_xywh[0] - monster_box[0]) < attx and abs(hero_xywh[1] - monster_box[1]) < atty:
                        print('准备攻击 , a')
                        # pydirectinput.press('a')
                        # directkeys.key_press('A')
                        directkeys.key_press("A")
                        if not action_cache:
                            pass
                        elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                            action_cache = None
                        elif action_cache:
                            ReleaseKey(direct_dic[action_cache])
                            action_cache = None
                    elif abs(hero_xywh[0] - monster_box[0]) < attx and abs(hero_xywh[1] - monster_box[1]) > atty:
                        if hero_xywh[1] - monster_box[1] < 0:
                            action_cache = move('DOWN', material=False, action_cache=None)
                        else:
                            action_cache = move('UP', material=False, action_cache=None)
                    elif abs(hero_xywh[0] - monster_box[0]) > attx and abs(hero_xywh[1] - monster_box[1]) < atty:
                        if hero_xywh[0] - monster_box[0] < 0:
                            action_cache = move('RIGHT', material=False, action_cache=None)
                        else:
                            action_cache = move('LEFT', material=False, action_cache=None)
                    elif abs(hero_xywh[0] - monster_box[0]) > attx and abs(hero_xywh[1] - monster_box[1]) > atty:
                        if hero_xywh[0] - monster_box[0] < 0 and hero_xywh[1] - monster_box[1] < 0:
                            action_cache = move('RIGHT_UP', material=False, action_cache=None)
                        elif hero_xywh[0] - monster_box[0] < 0 and hero_xywh[1] - monster_box[1] > 0:
                            action_cache = move('RIGHT_DOWN', material=False, action_cache=None)
                        elif hero_xywh[0] - monster_box[0] > 0 and hero_xywh[1] - monster_box[1] < 0:
                            action_cache = move('LEFT_UP', material=False, action_cache=None)
                        elif hero_xywh[0] - monster_box[0] > 0 and hero_xywh[1] - monster_box[1] > 0:
                            action_cache = move('LEFT_DOWN', material=False, action_cache=None)
                elif 'boss' in cls_object:
                    print('攻击boss')
                    # action_cache =  key_press('A')
                else:
                    print('没有识别到任何有效目标,去下一个门')
            img0 = cv2.resize(img0, (600, 375))
            if view_img:
                cv2.imshow('window', img0)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    raise StopIteration
    keys = key_check()
    if 'P' in keys:
        if not action_cache:
            pass
        elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
            action_cache = None
        else:
            ReleaseKey(direct_dic[action_cache])
            action_cache = None
        if paused:
            paused = False
            time.sleep(1)
        else:
            paused = True
            time.sleep(1)