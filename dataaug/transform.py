import cv2
import numpy as np
import os
import glob
from imagecorruptions import corrupt
from imagecorruptions import get_corruption_names

from skimage import transform
origin_path = "D:\\datasets\\dataaug\\data\\foreign_matter"
save_path = r"D:\datasets\dataaug\data\augfore"

def show(img):
    cv2.namedWindow("img",0)
    cv2.imshow("img", img)
    cv2.waitKey(0)

def rotate(img):
    copyimg = img.copy()
    res90 = cv2.rotate(copyimg,cv2.ROTATE_90_CLOCKWISE)
    res180 = cv2.rotate(copyimg,cv2.ROTATE_180)
    res270 = cv2.rotate(copyimg,cv2.ROTATE_90_COUNTERCLOCKWISE)
    return (res90,res180,res270)
def Affine(img,mode="offset"):
    copyimg = img.copy()
    width, height = img.shape[:2]
    angle = [45,135,225,315]
    imglist = []
    if mode == "angle":
        for i in angle:
            m_ratation = cv2.getRotationMatrix2D((width / 2, height / 2), i, 1)
            res = cv2.warpAffine(copyimg, m_ratation, dsize=(width, height))
            imglist.append(res)
    if mode == "offset":
        # 向左下各平移50
        m_ratation = np.float32([[1, 0, 10], [0, 1, 10]])
        res = cv2.warpAffine(copyimg, m_ratation, dsize=(width, height))
        imglist.append(res)
        # 利用平移矩阵进行仿射变换
        # lena_3 = cv2.warpAffine(img, m_move, dsize=(width, height))
    # 利用旋转矩阵进行仿射变换
    return imglist


def flip(img,flipCode = 1):
    # 水平翻转，正数水平，负数竖直，0为水平加竖直
    copyimg = img.copy()
    img1 = cv2.flip(copyimg,1)
    img2 = cv2.flip(copyimg,-1)
    img3 = cv2.flip(copyimg,0)
    return (img1,img2,img3)

def saveimg(method,img):
    if method == "rotate":
        res = rotate(img)
    elif method == "flip":
        res = flip(img)
    elif method == "offset":
        res = Affine(img,"offset")
    elif method == "angle":
        res = Affine(img, "angle")
    elif method == "corrup":
        res = corrup(img)
    for num, i in enumerate(res):
        img_file = method + str(num) + path
        save_file = os.path.join(save_classespath, img_file)
        print(save_file)
        cv2.imwrite(save_file, i)


def corrup(img):
    copyimg = img.copy()
    imglist = []
    for corruption in get_corruption_names():
        for severity in range(3):
            corrupted = corrupt(copyimg, corruption_name=corruption, severity=severity+1)
            imglist.append(corrupted)
    return imglist

# 如果文件夹不存在则创建
if not os.path.exists(save_path):
    os.mkdir(save_path)
# 遍历得到分类名字的文件夹
for calsses in os.listdir(origin_path):
    calsses_path = os.path.join(origin_path,calsses)
    save_classespath = os.path.join(save_path,calsses)
    # 如果文件夹不存在则创建
    if not os.path.exists(save_classespath):
        os.mkdir(save_classespath)
    # 遍历得到每个图片的名字
    for num,path in enumerate(os.listdir(calsses_path)):
        img_path = os.path.join(calsses_path,path)
        img = cv2.imread(img_path)
        saveimg("rotate",img) #3
        saveimg("flip",img)  # 3
        saveimg("offset",img) # 1
        saveimg("angle",img) # 4
        saveimg("corrup",img)# 15


        # show(img)


