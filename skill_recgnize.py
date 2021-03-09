import cv2 as cv
import numpy as np

def score(img):
    counter = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > 127:
                counter += 1
    return counter/(img.shape[0] * img.shape[1])

def img_show(img):
    cv.imshow("win", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

skill_height = int((793-733)/2)
skill_width = int((750-538)/7)

dict = {"A": (733+skill_height, 538), "S": (733+skill_height, 538+skill_width), "D": (733+skill_height, 538+2*skill_width),
        "F": (733+skill_height, 538+3*skill_width), "G": (733+skill_height, 538+4*skill_width),
        "H": (733+skill_height, 538+5*skill_width), "Q": (733, 538), "W": (733, 538+skill_width), "E": (733, 538+2*skill_width),
        "R": (733, 538+3*skill_width), "T": (733, 538+4*skill_width), "Y": (733, 538+5*skill_width)}


def skill_rec(skill_name, img):
    if skill_name == "X":
        return True
    skill_img = img[dict[skill_name][0]: dict[skill_name][0]+skill_height,
                dict[skill_name][1]: dict[skill_name][1]+skill_width, 2]
    if score(skill_img) > 0.1:
        return True
    else:
        return False

if __name__ == "__main__":
    img_path = "datasets/guiqi/test/20_93.jpg"
    img = cv.imread(img_path)
    print(skill_height, skill_width)
    print(img.shape)
    skill_img = img[733: 793, 538:750, 2]
    img_show(skill_img)


    skill_imgA = img[dict["A"][0]: dict["A"][0]+skill_height, dict["A"][1]: dict["A"][1]+skill_width, 2]
    skill_imgH= img[dict["H"][0]: dict["H"][0]+skill_height, dict["H"][1]: dict["H"][1]+skill_width, 2]
    skill_imgG= img[dict["G"][0]: dict["G"][0]+skill_height, dict["G"][1]: dict["G"][1]+skill_width, 2]
    skill_imgE= img[dict["E"][0]: dict["E"][0]+skill_height, dict["E"][1]: dict["E"][1]+skill_width, 2]
    skill_imgQ= img[dict["Q"][0]: dict["Q"][0]+skill_height, dict["Q"][1]: dict["Q"][1]+skill_width, 2]
    skill_imgS= img[dict["S"][0]: dict["S"][0]+skill_height, dict["S"][1]: dict["S"][1]+skill_width, 2]
    skill_imgY= img[dict["Y"][0]: dict["Y"][0]+skill_height, dict["Y"][1]: dict["Y"][1]+skill_width, 2]
    skill_imgD = img[dict["D"][0]: dict["D"][0]+skill_height, dict["D"][1]: dict["D"][1]+skill_width, 2]
    skill_imgF = img[dict["F"][0]: dict["F"][0]+skill_height, dict["F"][1]: dict["F"][1]+skill_width, 2]
    skill_imgW = img[dict["W"][0]: dict["W"][0]+skill_height, dict["W"][1]: dict["W"][1]+skill_width, 2]
    skill_imgR = img[dict["R"][0]: dict["R"][0]+skill_height, dict["R"][1]: dict["R"][1]+skill_width, 2]

    # print("A", np.mean(skill_imgA))
    # print("H", np.mean(skill_imgH))
    # print("G", np.mean(skill_imgG))
    # print("E", np.mean(skill_imgE))
    # print("Q", np.mean(skill_imgQ))
    # print("S", np.mean(skill_imgS))
    # print("Y", np.mean(skill_imgY))

    print("A", score(skill_imgA))
    print("Q", score(skill_imgQ))
    print("S", score(skill_imgS))
    print("D", score(skill_imgD))
    print("F", score(skill_imgF))
    print("W", score(skill_imgW))
    print("R", score(skill_imgR))
    print("Y", score(skill_imgY))
    print("H", score(skill_imgH))
    print("G", score(skill_imgG))
    print("E", score(skill_imgE))

    print(skill_rec("W", img))

