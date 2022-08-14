import os
import shutil

# 将标注好的图像和标签转移到新路径下
root_path = "datasets/guiqi/patch1"
yolo5_data_dir = "datasets/guiqi/patch1_yolo5"

json_list = []
imgs_list = []
dir = os.listdir(root_path)
for d in dir:
    if d.endswith(".json"):
        imgs_list.append(d.strip().split(".")[0] + ".jpg")
        json_list.append(d)
print(imgs_list)
print(json_list)

for img_name, json in zip(imgs_list, json_list):
    shutil.copy(os.path.join(root_path + "/" + img_name), os.path.join(yolo5_data_dir + '/imgs'))
    shutil.copy(os.path.join(root_path + "/" + json), os.path.join(yolo5_data_dir + '/labels_json'))

# # 选一部分数据作为验证集
# img_train_path = r"F:\Computer_vision\yolov5\YOLO5\DNF\train\images"
# img_valid_path = r"F:\Computer_vision\yolov5\YOLO5\DNF\valid\images"
# label_train_path = r"F:\Computer_vision\yolov5\YOLO5\DNF\train\labels"
# label_valid_path = r"F:\Computer_vision\yolov5\YOLO5\DNF\valid\labels"
# eval_ratio = 0.1
# dir = os.listdir(img_train_path)
# eval_nums = int(eval_ratio * len(dir))
# import random
# random.shuffle(dir)
# for d in dir[:eval_nums]:
#     shutil.move(os.path.join(img_train_path + "\\" + d), os.path.join(img_valid_path + "\\" + d))
#     shutil.move(os.path.join(label_train_path + "\\" + d.strip().split(".")[0] + ".txt"),
#                 os.path.join(label_valid_path + "\\" + d.strip().split(".")[0] + ".txt"))

# undict生成
#
# name2id = {'hero': 0, 'small_map': 1, "monster": 2, 'money': 3, 'material': 4, 'door': 5, 'BOSS': 6, 'box': 7, 'options': 8}
# id2name = {}
# for key, val in name2id.items():
#     id2name[val] = key
# print(id2name)