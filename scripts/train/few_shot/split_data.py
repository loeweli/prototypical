import os
import numpy as np
import glob
import random
# save_path = r"D:\datasets\dataaug\data\vitium"
save_traintxt = "D:\\datasets\\dataaug\\data\\vitium_split\\train.txt"
save_valtxt = r"D:\datasets\dataaug\data\vitium_split\val.txt"
datadir = r"D:\datasets\dataaug\data\vitium"

class_floder = os.listdir(datadir)
classes_len = len(class_floder)
order = random.sample(range(0,classes_len),int(classes_len * 0.7))
print(order)
rotate = ["000","090","180","270"]
# for i in order:
#     clases = class_floder[i]
#     path = os.path.join(datadir,clases)
#     print(path)
def path_way():
    with open(save_traintxt,"w") as tr,open(save_valtxt,"w") as va:
        for i,classes in enumerate(class_floder):
            path = os.path.join(datadir, classes) # 全路径
            file_name = os.listdir(path)
            if i in order:
                for file in file_name:
                    file_path = os.path.join(path , file)
                    # print(file_path)
                    tr.writelines(file_path+" " + classes + "\n")
            else:
                for file in file_name:
                    file_path = os.path.join(path , file)
                    # print(file_path)
                    va.writelines(file_path+" " + classes + "\n")

def class_way():
    with open(save_traintxt,"w") as tr,open(save_valtxt,"w") as va:
        for i,classes in enumerate(class_floder):
            if i in order:
                for j in rotate:
                    print(classes)
                    tr.writelines(classes + "/rot" + j + "\n")
            else:
                for j in rotate:
                    va.writelines(classes + "/rot" + j + "\n")
class_way()