import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 读取数据并随机
np.random.seed(111)
x_train = np.load('./x_train.npy')
# 随机排序
np.random.shuffle(x_train)
x_test = np.load('./x_test.npy')
np.random.shuffle(x_test)

np.random.seed(111)
y_train = np.load('./y_train.npy')
np.random.shuffle(y_train)
y_test = np.load('./y_test.npy')
np.random.shuffle(y_test)

# 修正标签值超出有效范围
num_classes = 62
y_train = np.clip(y_train, 0, num_classes - 1)
y_test = np.clip(y_test, 0, num_classes - 1)

# 创建模型
model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(62, activation='softmax')
])

# 编译模型
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy'])

# 模型训练
history = model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_train[:10000], y_train[:10000]))

# 评估模型
eva = model.evaluate(x_test, y_test)
print('评估准确率:', eva[1])

# 打印分类报告
y_pred = np.argmax(model.predict(x_test), axis=1)
print('分类报告:\n', classification_report(y_test, y_pred))

# 创建一个列表用来保存错误分类的数据
misclassified_index = []
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        misclassified_index.append(i)

# 创建错误分类图像保存目录
save_dir = "error"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 打印错误分类的数据
print('Misclassified data:')
for index in misclassified_index:
    print("标签索引:", index, "标签真实值:", y_test[index], "标签预测值:", y_pred[index])

    # 获取错误分类的图像数据
    misclassified_image = x_test[index].reshape((60, 20))

    # 生成保存路径和文件名
    save_path = os.path.join(save_dir, f"misclassified_{index}.png")

    # 保存图像
    plt.imshow(misclassified_image, cmap='gray')
    plt.title(f"Actual: {y_test[index]}, Predicted: {y_pred[index]}")
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

# 完成后提示保存的目录
print("已经错误的数据图像保存在本地目录中.")

# 打印混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print('混淆矩阵:\n', cm)

# 绘制混淆矩阵图像
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=np.arange(num_classes),
            yticklabels=np.arange(num_classes))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')

# 绘制损失函数曲线
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')

# 绘制准确率曲线
plt.figure()
plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_curve.png')

# 保存模型
model.save('captcha_model')
print('模型保存完成')
