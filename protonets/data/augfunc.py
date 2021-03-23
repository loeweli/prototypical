import  cv2
import numpy
import os
import glob
from PIL import Image
import random
import matplotlib.pyplot as plt
import torch
def pairing(p1,p2):
    p1 =  p1.numpy()
    p2 =  p2.numpy()
    return cv2.add(p1 * 0.5, p2 * 0.5)
def samplepairing(sample):
    xs = sample["xs"]
    all_data = xs.view(xs.shape[0]*xs.shape[1],xs.shape[2],xs.shape[3],xs.shape[4])
    order = random.sample(range(9,all_data.shape[0]),int(all_data.shape[0] * 0.25))
    listway = []
    for way in range(0,xs.shape[0]):  # 5
        listshot = []
        for shot in xs[way]: # 5
            for i in order:
                newdata = pairing(shot,all_data[i])
                listshot.append(newdata)
        listway.append(listshot)

    sample["xs"] = torch.Tensor(listway)
    return sample





if __name__ == "__main__":
    path1 = r"D:\datasets\dataaug\data\vitium\0\027_49.bmp"
    path2 = r"D:\datasets\dataaug\data\vitium\1\0168_36.bmp"
    img1 = cv2.imread(path1,1)
    # cv2.namedWindow("img1",0)
    # cv2.imshow("img1", img1)
    img2 = cv2.imread(path2,1)

    img5 = (img1 * 0.5)
    img3 = cv2.add(img1 * 0.5 , img2 * 0.5)
    # img3.astype(int)
    # img3 = img1 * 0.5 + img2 * 0.5
    # print(img1 * 0.5)

    img = Image.fromarray(img3.astype('uint8'))#.convert('L')
    plt.figure("tt")
    plt.imshow(img)
    plt.show()
    # cv2.namedWindow("img3",0)
    # cv2.imshow("img3", img3)
    # cv2.waitKey(0)