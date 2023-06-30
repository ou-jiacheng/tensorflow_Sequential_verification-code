import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# 读取文件夹的图片
base_path = r'./crop_img'
one_level_path = os.listdir(base_path)

x_train, y_train = [], []
for one_level in one_level_path:
    two_level_path = os.path.join(base_path, one_level)  # 得到完整的文件夹名称
    for two_level in os.listdir(two_level_path):
        picture_path = os.path.join(two_level_path, two_level)  # 得到具体图片文件的完整路径
        img = cv2.imread(picture_path, 0) / 255.0  # 灰度模式读取图片并归一化
        x_train.append(img)
        y_train.append(ord(one_level))  # 将一级文件夹的名称转换为对应的ASCII码值

x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = y_train.astype(np.int64)  # 转换为np.int64数据类型

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)  # random_state设置随机种子

# 保存数据
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)

print('数据保存完成')
