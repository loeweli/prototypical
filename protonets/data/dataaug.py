import os
import torch
from torch.utils.data import DataLoader ,Dataset
import numpy as np
import cv2
OMNIGLOT_DATA_DIR  =  'D:\datasets\dataau'
class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes # 100

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


class Augdata(Dataset):
    def __init__(self,datapath):
        "初始化数据"
        self.path_set = 0
        with open(datapath , "r") as da:
            self.path_set = da.readlines()
        self.lenth = len(self.path_set)
    def __getitem__(self,index):
        path,label = self.path_set[index].strip().split(" ")
        img = cv2.imread(path)
        return img,label
    def __len__(self):
        return self.lenth

save_traintxt = "D:\\datasets\\dataaug\\data\\vitium_split\\train.txt"
augdata = Augdata(save_traintxt)

train_data = DataLoader(dataset=augdata,batch_size=2,shuffle=True)
for data in train_data:
    input , label = data
    print(input.shape)
    print(label)

def load(opt, splits):
    split_dir = os.path.join(OMNIGLOT_DATA_DIR, 'splits', opt['data.split']) # splits/vinyals

    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way'] #60

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot'] # 5

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query'] # 5

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes'] # 100