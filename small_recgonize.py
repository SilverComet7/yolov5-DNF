import cv2 as cv
import numpy as np


img_path = "test/DNF.png"
img = cv.imread(img_path)

def img_show(img):
    cv.imshow("win", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def current_door(img, stride = 17):
    crop = img[45:65, 1107:1270, 0]
    # img_show(crop)
    index = np.unravel_index(crop.argmax(), crop.shape)
    i = int((index[1] // stride) + 1)
    return i  # 返回的是在第几个房间

def next_door(img):
    img_temp = np.load("问号模板.npy")
    # img_show(img_temp)
    target = img[45:65, 1107:1270]
    result = cv.matchTemplate(target, img_temp, cv.TM_SQDIFF_NORMED)
    cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    next_door_id = 0
    if min_val < 1e-10:
        # print(min_val, max_val, min_loc, max_loc)
        strmin_val = str(min_val)
        theight, twidth = img_temp.shape[:2]
        # cv.rectangle(target, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (225, 0, 0), 2)
        # cv.imshow("MatchResult----MatchingValue=" + strmin_val, target)
        # cv.waitKey()
        # cv.destroyAllWindows()
        next_door_id = int(((min_loc[0] + 0.5 * twidth) // 18.11) + 1)
    return next_door_id

if __name__ == "__main__":
    print(current_door(img))
    print(next_door(img))
    # img_show(img[45:65, 1144:1162])
    # np.save("问号模板", img[45:65, 1144:1162])


