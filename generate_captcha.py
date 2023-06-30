from captcha.image import ImageCaptcha, random_color
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2, string, random, os


# 生成验证码
def generate():
    letters = string.ascii_letters + string.digits
    chars = ''.join(random.sample(letters, k=4))  # 随机选择4个字符，以构成验证码的文本
    image = ImageCaptcha()  # 创建验证码图像
    # 生成验证码
    captcha_ = image.create_captcha_image(chars=chars,
                                          color=random_color(0, 256),
                                          background=random_color(0, 256))
    # 转成灰度图后再转成numpy格式
    img = np.array(captcha_.convert('L'))
    # 计算平均阈值
    threshold = img.sum() // img.size
    # 将其转成二值化
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)  # threshold是阈值
    # 将其转成白底黑字
    if img[0, 0] == 0:  # 判断左上角的颜色,如果是黑色就转换
        # 找出目前黑色的索引
        black_index = np.where(img == 0)
        # 找出目前白色的索引
        white_index = np.where(img != 0)
        img[black_index] = 255
        img[white_index] = 0
    # 将全部小于255的值全部设置为0
    img[np.where(img < 255)] = 0
    projection_coordinate(img, chars)


# 根据投影切割图片的坐标
def projection_coordinate(img, chars='images'):
    cols_sum = []
    for index in range(img.shape[1]):
        col = img[:, index]  # 获取图像中第index列的像素值
        cols_sum.append(len(np.where(col == 0)[0]))
    cols_sum = np.array(cols_sum)
    index = 0  # 记录要裁剪的区域的起始列
    crop_lis = []  # 记录要裁剪的区域坐标
    for ind, num in enumerate(cols_sum):
        # 如果num为0,那么说明当前列没有黑点,index=ind
        if num == 0:
            index = ind
        else:
            if ind - index > 5 and all(cols_sum[ind + 1:ind + 4] == 0):
                crop_lis.append((index - 2, ind + 2))
                index = ind
    if len(crop_lis) != 4:
        crop_lis = []
    crop_img(img, coordinate=crop_lis, chars=chars)


# 切割图片
def crop_img(img, coordinate=None, chars='images'):
    if not coordinate:
        return
    for coord, c in zip(coordinate, chars):
        # 切割
        c_img = img[0:, coord[0]:coord[1]]
        # 重新调整图片大小,(20,60)
        c_img = cv2.resize(c_img, (20, 60))
        save_img(c_img, chars, c)


# 保存图片
def save_img(img, chars, c):
    base_path = r'./crop_img'
    # 标签路径
    c = c.lower()
    tag_path = os.path.join(base_path, c)
    if not os.path.exists(tag_path):
        os.makedirs(tag_path)
    # 图片路径
    captcha_path = os.path.join(tag_path, chars + '-' + c + '.jpg')
    cv2.imwrite(captcha_path, img)
    print(captcha_path + '生成成功')


if __name__ == "__main__":
    for _ in range(10000):  # 执行10000次
        generate()
        print(_)
