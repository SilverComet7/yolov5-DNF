import random
import time

import cv2
import numpy as np
import torch

import directkeys
from direction_move import move
from directkeys import ReleaseKey
from getkeys import key_check
from grabscreen import grab_screen
from models.experimental import attempt_load
from utils.general import (
    non_max_suppression, scale_coords,
    xyxy2xywh)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


# 设置所有用到的参数
weights = 'best.pt'  # yolo5 模型存放的位置
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = attempt_load(weights, device)  # load FP32 model
window_size = (0, 0, 1280, 800)  # 截屏的位置
img_size = 640  # 输入到yolo5中的模型尺寸
paused = False
half = device.type != 'cpu'
view_img = True  # 是否观看目标检测结果
save_txt = False
conf_thres = 0.3  # NMS的置信度过滤
iou_thres = 0.2  # NMS的IOU阈值
classes = None
agnostic_nms = False  # 不同类别的NMS时也参数过滤
skill_char = "XYHGXFAXDSWXETX"  # 技能按键，使用均匀分布随机抽取
direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}  # 上下左右的键码
names = ['hero', 'small_map', "monster", 'money', 'material', 'door', 'BOSS', 'box', 'options']  # 所有类别名
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
if half:
    model.half()  # to FP16
action_cache = None  # 动作标记
press_delay = 0.1  # 按压时间
release_delay = 0.1  # 释放时间
frame = 0  # 帧
door1_time_start = -20
next_door_time = -20
fs = 1  # 每四帧处理一次


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


# 捕捉画面+目标检测+玩游戏
while True:
    if not paused:
        t_start = time.time()
        img0 = grab_screen(window_size)
        frame += 1
        if frame % fs == 0:

            img = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)

            # Padded resize
            # img = letterbox(img0, new_shape=img_size)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device).unsqueeze(dim=0)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            pred = model(img, augment=False)[0]

            # Apply NMS
            det = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            det = det[0]  # 所有的检测到的目标wqewq

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                img_object = []
                cls_object = []
                # Write results
                hero_conf = 0
                hero_index = 0
                # 遍历该对象属性 位置  类别
                for idx, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    # 转换xywh形式，方便计算距离
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    cls = int(cls)
                    img_object.append(xywh)
                    cls_object.append(names[cls])

                    if names[cls] == "hero" and conf > hero_conf:
                        hero_conf = conf
                        hero_index = idx

                    if view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)

                # 游戏
                thx = 30  # 捡东西时，x方向的阈值
                thy = 30  # 捡东西时，y方向的阈值
                attx = 150  # 攻击时，x方向的阈值
                atty = 50  # 攻击时，y方向的阈值

                # if current_door(img0) == 1 and time.time() - door1_time_start > 10:
                #     door1_time_start = time.time()
                #     # move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                #     #      release_delay=release_delay)
                #     # ReleaseKey(direct_dic["RIGHT"])
                #     # directkeys.key_press("SPACE")
                #     directkeys.key_press("CTRL")
                #     time.sleep(1)
                #     directkeys.key_press("ALT")
                #     time.sleep(0.5)
                #     action_cache = None
                # 扫描英雄
                if "hero" in cls_object:
                    # hero_xywh = img_object[cls_object.index("hero")]
                    hero_xywh = img_object[hero_index]
                    cv2.circle(img0, (int(hero_xywh[0]), int(hero_xywh[1])), 1, (0, 0, 255), 10)
                    # print(hero_index)
                    # print(cls_object.index("hero"))
                else:
                    continue
                # 打怪
                if "monster" in cls_object or "BOSS" in cls_object:
                    min_distance = float("inf")
                    for idx, (c, box) in enumerate(zip(cls_object, img_object)):
                        if c == 'monster' or c == "BOSS":
                            dis = ((hero_xywh[0] - box[0]) ** 2 + (hero_xywh[1] - box[1]) ** 2) ** 0.5
                            if dis < min_distance:
                                monster_box = box
                                monster_index = idx
                                min_distance = dis
                    # 处于攻击距离
                    if abs(hero_xywh[0] - monster_box[0]) < attx and abs(hero_xywh[1] - monster_box[1]) < atty:
                        directkeys.key_press("A")
                        print("释放技能攻击")
                        if not action_cache:
                            pass
                        elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                            action_cache = None
                        elif action_cache:
                            ReleaseKey(direct_dic[action_cache])
                            action_cache = None
                        # break
                    # 怪物在英雄右上  ， 左上     左下   右下
                    elif monster_box[1] - hero_xywh[1] < 0 and monster_box[0] - hero_xywh[0] > 0:
                        # y方向 小于攻击距离
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        #
                        elif hero_xywh[1] - monster_box[1] < monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - monster_box[1] >= monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                        # break
                    elif monster_box[1] - hero_xywh[1] < 0 and monster_box[0] - hero_xywh[0] < 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - monster_box[1] < hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="LEFT_UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - monster_box[1] >= hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif monster_box[1] - hero_xywh[1] > 0 and monster_box[0] - hero_xywh[0] < 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif monster_box[1] - hero_xywh[1] < hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="LEFT_DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif monster_box[1] - hero_xywh[1] >= hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif monster_box[1] - hero_xywh[1] > 0 and monster_box[0] - hero_xywh[0] > 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif monster_box[1] - hero_xywh[1] < monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif monster_box[1] - hero_xywh[1] >= monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="DOWN", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break

                # # 移动到下一个地图
                # if "door" in cls_object and "monster" not in cls_object and "BOSS" not in cls_object and "material" not in cls_object and "money" not in cls_object:
                #     for idx, (c, box) in enumerate(zip(cls_object, img_object)):
                #         if c == 'door':
                #             door_box = box
                #             door_index = idx
                #     # 门的位置小于抓取的一半，在左侧
                #     if door_box[0] < img0.shape[0] // 2:
                #         action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                #                             release_delay=release_delay)
                #         # break
                #     elif door_box[1] - hero_xywh[1] < 0 and door_box[0] - hero_xywh[0] > 0:
                #         if abs(door_box[1] - hero_xywh[1]) < thy and abs(door_box[0] - hero_xywh[0]) < thx:
                #             action_cache = None
                #             print("进入下一地图")
                #             # break
                #         elif abs(door_box[1] - hero_xywh[1]) < thy:
                #             action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif hero_xywh[1] - door_box[1] < door_box[0] - hero_xywh[0]:
                #             action_cache = move(direct="RIGHT_UP", action_cache=action_cache, press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif hero_xywh[1] - door_box[1] >= door_box[0] - hero_xywh[0]:
                #             action_cache = move(direct="UP", action_cache=action_cache, press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #     elif door_box[1] - hero_xywh[1] < 0 and door_box[0] - hero_xywh[0] < 0:
                #         if abs(door_box[1] - hero_xywh[1]) < thy and abs(door_box[0] - hero_xywh[0]) < thx:
                #             action_cache = None
                #             print("进入下一地图")
                #             # break
                #         elif abs(door_box[1] - hero_xywh[1]) < thy:
                #             action_cache = move(direct="LEFT", action_cache=action_cache, press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif hero_xywh[1] - door_box[1] < hero_xywh[0] - door_box[0]:
                #             action_cache = move(direct="LEFT_UP", action_cache=action_cache, press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif hero_xywh[1] - door_box[1] >= hero_xywh[0] - door_box[0]:
                #             action_cache = move(direct="UP", action_cache=action_cache, press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #     elif door_box[1] - hero_xywh[1] > 0 and door_box[0] - hero_xywh[0] < 0:
                #         if abs(door_box[1] - hero_xywh[1]) < thy and abs(door_box[0] - hero_xywh[0]) < thx:
                #             action_cache = None
                #             print("进入下一地图")
                #             # break
                #         elif abs(door_box[1] - hero_xywh[1]) < thy:
                #             action_cache = move(direct="LEFT", action_cache=action_cache, press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif door_box[1] - hero_xywh[1] < hero_xywh[0] - door_box[0]:
                #             action_cache = move(direct="LEFT_DOWN", action_cache=action_cache, press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif door_box[1] - hero_xywh[1] >= hero_xywh[0] - door_box[0]:
                #             action_cache = move(direct="DOWN", action_cache=action_cache, press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #     elif door_box[1] - hero_xywh[1] > 0 and door_box[0] - hero_xywh[0] > 0:
                #         if abs(door_box[1] - hero_xywh[1]) < thy and abs(door_box[0] - hero_xywh[0]) < thx:
                #             action_cache = None
                #             print("进入下一地图")
                #             # break
                #         elif abs(door_box[1] - hero_xywh[1]) < thy:
                #             action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif door_box[1] - hero_xywh[1] < door_box[0] - hero_xywh[0]:
                #             action_cache = move(direct="RIGHT_DOWN", action_cache=action_cache, press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif door_box[1] - hero_xywh[1] >= door_box[0] - hero_xywh[0]:
                #             action_cache = move(direct="DOWN", action_cache=action_cache, press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                # if "money" not in cls_object and "material" not in cls_object and "monster" not in cls_object \
                #         and "BOSS" not in cls_object and "door" not in cls_object and 'box' not in cls_object \
                #         and 'options' not in cls_object:
                #     # if next_door(img0) == 0 and abs(time.time()) - next_door_time > 10:
                #     #     next_door_time = time.time()
                #     #     action_cache = move(direct="LEFT", action_cache=action_cache, press_delay=press_delay,
                #     #                         release_delay=release_delay)
                #     #     # time.sleep(3)
                #     # else:
                #     #     action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                #     #                     release_delay=release_delay)
                #
                #     #没有识别到 则向右走
                #     action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                #                         release_delay=release_delay)
                #     # break
                #
                # # 捡材料
                # if "monster" not in cls_object and "hero" in cls_object and (
                #         "material" in cls_object or "money" in cls_object):
                #     min_distance = float("inf")
                #     hero_xywh[1] = hero_xywh[1] + (hero_xywh[3] // 2) * 0.7
                #     thx = thx / 2
                #     thy = thy / 2
                #     for idx, (c, box) in enumerate(zip(cls_object, img_object)):
                #         if c == 'material' or c == "money":
                #             dis = ((hero_xywh[0] - box[0]) ** 2 + (hero_xywh[1] - box[1]) ** 2) ** 0.5
                #             if dis < min_distance:
                #                 material_box = box
                #                 material_index = idx
                #                 min_distance = dis
                #     if abs(material_box[1] - hero_xywh[1]) < thy and abs(material_box[0] - hero_xywh[0]) < thx:
                #         if not action_cache:
                #             pass
                #         elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                #             ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                #             ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                #             action_cache = None
                #         else:
                #             ReleaseKey(direct_dic[action_cache])
                #             action_cache = None
                #         time.sleep(1)
                #         directkeys.key_press("X")
                #         print("捡东西")
                #         # break
                #
                #     elif material_box[1] - hero_xywh[1] < 0 and material_box[0] - hero_xywh[0] > 0:
                #
                #         if abs(material_box[1] - hero_xywh[1]) < thy:
                #             action_cache = move(direct="RIGHT", material=True, action_cache=action_cache,
                #                                 press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif hero_xywh[1] - material_box[1] < material_box[0] - hero_xywh[0]:
                #             action_cache = move(direct="RIGHT_UP", material=True, action_cache=action_cache,
                #                                 press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif hero_xywh[1] - material_box[1] >= material_box[0] - hero_xywh[0]:
                #             action_cache = move(direct="UP", material=True, action_cache=action_cache,
                #                                 press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #     elif material_box[1] - hero_xywh[1] < 0 and material_box[0] - hero_xywh[0] < 0:
                #         if abs(material_box[1] - hero_xywh[1]) < thy:
                #             action_cache = move(direct="LEFT", material=True, action_cache=action_cache,
                #                                 press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif hero_xywh[1] - material_box[1] < hero_xywh[0] - material_box[0]:
                #             action_cache = move(direct="LEFT_UP", material=True, action_cache=action_cache,
                #                                 press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif hero_xywh[1] - material_box[1] >= hero_xywh[0] - material_box[0]:
                #             action_cache = move(direct="UP", material=True, action_cache=action_cache,
                #                                 press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #     elif material_box[1] - hero_xywh[1] > 0 and material_box[0] - hero_xywh[0] < 0:
                #         if abs(material_box[1] - hero_xywh[1]) < thy:
                #             action_cache = move(direct="LEFT", material=True, action_cache=action_cache,
                #                                 press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif material_box[1] - hero_xywh[1] < hero_xywh[0] - material_box[0]:
                #             action_cache = move(direct="LEFT_DOWN", material=True, action_cache=action_cache,
                #                                 press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif material_box[1] - hero_xywh[1] >= hero_xywh[0] - material_box[0]:
                #             action_cache = move(direct="DOWN", material=True, action_cache=action_cache,
                #                                 press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #     elif material_box[1] - hero_xywh[1] > 0 and material_box[0] - hero_xywh[0] > 0:
                #         if abs(material_box[1] - hero_xywh[1]) < thy:
                #             action_cache = move(direct="RIGHT", material=True, action_cache=action_cache,
                #                                 press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif material_box[1] - hero_xywh[1] < material_box[0] - hero_xywh[0]:
                #             action_cache = move(direct="RIGHT_DOWN", material=True, action_cache=action_cache,
                #                                 press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                #         elif material_box[1] - hero_xywh[1] >= material_box[0] - hero_xywh[0]:
                #             action_cache = move(direct="DOWN", material=True, action_cache=action_cache,
                #                                 press_delay=press_delay,
                #                                 release_delay=release_delay)
                #             # break
                # # 开箱子
                # if "box" in cls_object:
                #     box_num = 0
                #     for b in cls_object:
                #         if b == "box":
                #             box_num += 1
                #     if box_num >= 4:
                #         directkeys.key_press("ESC")
                #         print("打开箱子ESC")
                #         # break62

                # 重新开始
                time_option = -20
                if "options" in cls_object:
                    if not action_cache:
                        pass
                    elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                        ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                        ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                        action_cache = None
                    else:
                        ReleaseKey(direct_dic[action_cache])
                        action_cache = None
                    if time.time() - time_option > 10:
                        directkeys.key_press("NUM0")
                        print("移动物品到脚下")
                        directkeys.key_press("X")
                        time_option = time.time()
                    directkeys.key_press("F2")
                    print("重新开始F2")
                    # break
            t_end = time.time()
            print("一帧游戏操作所用时间：", (t_end - t_start) / fs)

            img0 = cv2.resize(img0, (600, 375))
            # Stream results
            if view_img:
                cv2.imshow('window', img0)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    raise StopIteration

    # Setting pause and unpause
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
