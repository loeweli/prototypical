import  cv2
import numpy as np
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


def mixup(sample, alpha=1.0, use_cuda=False):
    # 对数据的mixup 操作 x = lambda*x_i+(1-lamdda)*x_j
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    x = sample["xs"]
    xa ,xb = x[0] ,x[1]
    label = sample["class"]
    ya = label[0]
    yb = label[1]
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    # residue = x[index, :]
    # mixed_x = lam * x + (1 - lam) * x[index, :]  # 此处是对数据x_i 进行操作
    mixed_x = lam * xa + (1 - lam) * xb  # 此处是对数据x_i 进行操作
    mixed_y = lam * xb + (1 - lam) * xa  # 此处是对数据x_i 进行操作
    xq = torch.cat([mixed_x.unsqueeze(0),mixed_y.unsqueeze(0)],0)
    # y_a, y_b = y, y[index]  # 记录下y_i 和y_j
    print(mixed_x.shape)
    y = torch.arange(0, len(label)).view(len(label),1).expand(len(label), mixed_x.shape[0]).long()
    ya = y[0]
    yb = y[1]
    # sample["xs"] = xa.unsqueeze(0)
    # sample["xq"] = mixed_x.unsqueeze(0)
    sample["xq"] = xq
    return sample, ya, yb, lam  # 返回y_i 和y_j 以及lambda



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