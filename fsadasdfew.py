import torch.nn as nn
import torch
# x = torch.randn(60,1,3,3)
# # conv = torch.nn.Conv2d(1,64,3,padding=1)
# max = nn.MaxPool2d(2)
#
# res = max(x)
#
# print(res.shape)


# import numpy as np
#
# a = torch.arange(0,6)
# b = a.view(2,3)
# c = b.unsqueeze(1).expand(2,2,3)
# print(c)

# import torch as t
# a = t.arange(0, 16).view(4, 4)
#
# print(a)
# # 选取对角线的元素
# index = t.LongTensor([[0, 1, 2, 3],[0, 1, 2, 3]])
# c = a.gather(0, index)
# print(c)

# t = torch.Tensor([[1, 2], [3,4]])
# print(t)
# g = torch.gather(t,1, torch.LongTensor([ [1, 0],[0,1]]))
# print(g)



a = torch.Tensor(range(0,30)).view(2,3,5)
print(a)

index = torch.LongTensor([[[0,1,2,0,2],
                           [0,0,0,0,0],
                           [1,1,1,1,1]],
                          [[1,2,2,2,2],
                           [0,0,0,0,0],
                           [2,2,2,2,2]]])
index2 = torch.LongTensor([[[0,1,1,0,1],
                            [0,1,1,1,1],
                            [1,1,1,1,1]],
                           [[1,0,0,0,0],
                            [0,0,0,0,0],
                            [1,1,0,0,0]],
                           [[1,0,0,0,0],
                            [0,0,0,0,0],
                            [1,1,0,0,0]]])
b = torch.gather(a,2,index)
print("dim=1:\n",b)



# import torch
# a = torch.randint(0, 30, (2, 3, 5))
# print(a)
#
# index2 = torch.LongTensor([[[0,1,1,0,1],
#                           [0,1,1,1,1],
#                           [1,1,1,1,1]],
#                         [[1,0,0,0,0],
#                          [0,0,0,0,0],
#                          [1,1,0,0,0]]])
# d = torch.gather(a, 0,index2)
# print(d)

