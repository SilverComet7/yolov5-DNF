import numpy as np
from grabscreen import grab_screen
import cv2
import time
import directkeys
import torch
from torch.autograd import Variable
from directkeys import PressKey, ReleaseKey, key_down, key_up
from getkeys import key_check
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from models.experimental import attempt_load
import random

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

weights = r'E:\Computer_vision\yolov5\YOLO5\yolov5-master\runs\exp0\weights\best.pt'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = attempt_load(weights, map_location=device)  # load FP32 model
window_size = (0,0,1200,750)
last_time = time.time()
for i in list(range(5))[::-1]:
    print(i + 1)
    time.sleep(1)
img_size = 608
paused = False
half = device.type != 'cpu'
view_img = True
save_txt = False
conf_thres = 0.3
iou_thres = 0.2
classes = None
agnostic_nms = True
names = ['hero', 'small_map', "monster", 'money', 'material', 'door', 'BOSS', 'box', 'options']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
if half:
    model.half()  # to FP16

while (True):
    if not paused:
        img0 = grab_screen(window_size)
        print('loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)
        # Padded resize
        img = letterbox(img0, new_shape=img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device).unsqueeze(0)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        t1 = time_synchronized()
        # print(img.shape)
        pred = model(img, augment=False)[0]

        # Apply NMS
        det = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()
        print("inference and NMS time: ", t2 - t1)
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        det = det[0]
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # if save_txt:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     with open(txt_path + '.txt', 'a') as f:
                #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                if view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)

        img0 = cv2.resize(img0, (600, 375))
        # Stream results
        if view_img:
            cv2.imshow('window', img0)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                raise StopIteration

        # Setting pause and unpause
        keys = key_check()
        if 'P' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                time.sleep(1)