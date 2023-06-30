# 导入所需库
import keras
from captcha.image import ImageCaptcha, random_color
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import string
import random
import os
from PIL import Image, ImageDraw, ImageFont
import random

# 创建 "photo" 目录（如果不存在）
if not os.path.exists("photo"):
    os.makedirs("photo")
# 加载模型
model = keras.models.load_model('captcha_model')

# 验证码字符集
CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


# 生成随机字符串
def generate_code(length):
    return ''.join(random.choice(CHARS) for _ in range(length))


# 生成验证码图片
def generate_image(code, font_path, font_size, width, height, noise_points=100):
    # 创建画布
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 加载字体
    font = ImageFont.truetype(font_path, font_size)

    # 计算字符间距
    char_width, char_height = draw.textsize(code[0], font=font)
    gap = int((width - char_width * len(code)) / (len(code) + 1))

    # 绘制字符
    x = gap
    for c in code:
        y = random.randint(0, height - char_height)
        draw.text((x, y), c, font=font, fill=random_color())
        x += char_width + gap

    return image


# 生成随机颜色
def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))


# 生成验证码图片
def generate():
    code = generate_code(4)
    image = generate_image(code, 'arial.ttf', 36, 200, 80)
    image.show()

    # 转换为灰度图并转换为 numpy 数组格式
    image = np.array(image.convert('L'))
    # 计算平均阈值
    threshold = image.sum() // image.size
    # 将图像二值化
    _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    # 转换为白底黑字
    if image[0, 0] == 0:  # 判断左上角的颜色，如果是黑色则转换
        # 找出所有黑色像素的索引
        black_index = np.where(image == 0)
        # 找出所有白色像素的索引
        white_index = np.where(image != 0)
        image[black_index] = 255
        image[white_index] = 0
    # 将所有小于255的值设置为0
    image[np.where(image < 255)] = 0

    # 保存验证码图片
    image_path = os.path.join("photo", f"{code}.png")
    Image.fromarray(image).save(image_path)

    return (image, code)


# 投影切割图片的坐标
def projection_coordinate():
    img, chars = generate()

    # 计算每一列为0的个数
    cols_sum = np.sum(img == 0, axis=0)

    # 记录要裁剪的区域坐标
    crop_lis = []
    index = 0
    for ind, num in enumerate(cols_sum):
        if num == 0:
            index = ind
        else:
            if ind - index > 5 and np.all(cols_sum[ind + 1:ind + 4] == 0):
                crop_lis.append((index - 2, ind + 2))
                index = ind

    return img, crop_lis, chars


# 切割图片
def crop_img():
    img, coordinate, chars = projection_coordinate()
    if not coordinate:
        return
    # 保存切割的4份图片
    results = []
    for coord, c in zip(coordinate, chars):
        # 切割
        c_img = img[0:, coord[0]:coord[1]]
        # 重新调整图片大小,(20,60)
        c_img = cv2.resize(c_img, (20, 60))
        results.append(c_img)
    return (results, chars)


# 将模型输出转换为字符索引
def decode_predictions(prediction):
    predicted_label = np.argmax(prediction, axis=1)  # 沿着第一个轴（axis=1）找到具有最大值的元素的索引
    return predicted_label


# 将标签索引转换为字符
def label_to_char(label, charset):
    index = label % len(charset)  # 将标签限制在字符集的有效索引范围内
    return charset[index]


if __name__ == "__main__":
    chars_set = string.ascii_letters + string.digits
    results, chars = crop_img()
    if results is None:
        print("无法成功切割验证码")
    elif len(results) != 4:
        print("切割的验证码数量不正确")
    else:
        predicted_chars = ""
        for img in results:
            img = np.expand_dims(img, axis=0)  # 将图片数据 img 扩展为一个维度
            img = np.expand_dims(img, axis=-1)  # 将图片数据 img 扩展为一个额外的灰度通道
            img = img / 255.0  # 归一化
            prediction = model.predict(img)
            predicted_labels = decode_predictions(prediction)  # 解码为标签列表
            predicted_chars += "".join(label_to_char(label, chars_set) for label in predicted_labels)
        print("识别结果:", predicted_chars)
        print("真实结果:", chars)
