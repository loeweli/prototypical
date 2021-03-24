from imagecorruptions import corrupt
import cv2
from imagecorruptions import get_corruption_names
                        # 闪光             脉冲               失焦
# ‘gaussian_noise’, ‘shot_noise’, ‘impulse_noise’, ‘defocus_blur’,
# ‘glass_blur’, ‘motion_blur移动’, ‘zoom推近或拉远_blur’, ‘snow’, ‘frost霜’, ‘fog’,烟雾
# ‘brightness’, ‘contrast对比度’, ‘elastic_transform弹性变换’, ‘pixelate滤镜效果’, ‘jpeg_compression’,jpeg压缩


img = cv2.imread(r"D:\datasets\dataaug\data\foreign_matter\0\qq.bmp")
# print(get_corruption_names())
# cv2.namedWindow("img1", 0)

# for corruption in get_corruption_names():
#     for severity in range(3):
#         corrupted = corrupt(img, corruption_name=corruption, severity=severity+1)
#         cv2.imshow("img", corrupted)
#
#         cv2.waitKey(0)

# img = corrupt(img,corruption_name= "gaussian_blur",severity=1)
# cv2.namedWindow("img1",0)
cv2.cv2.imshow("img1",img)
# cv2.cv2.imshow()
cv2.waitKey(0)