import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 数据准备，构造训练集
x_t = np.arange(-23 / 18, (2 * np.pi - 23) / 18, 2 * np.pi / 18 / 2000, dtype=np.float32)
x_t = x_t[:, np.newaxis]
x_train = np.concatenate((x_t, np.power(x_t, 2), np.power(x_t, 3)), axis=1)
y_train = np.cos(18 * x_t + 23)


def test_fun():
    # 定义模型
    inputs = tf.keras.Input(shape=(3,), name='data')
    outputs = tf.keras.layers.Dense(units=1, input_dim=3)(inputs)
    Lm1 = tf.keras.Model(inputs=inputs, outputs=outputs, name='linear')

    # 定义存储模型的回调函数，请补充完整
    ########## Begin ###########
    checkpoint_path = r"C:\Users\YunRW\Desktop\12\cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=0, period=150)
    ########## End ###########

    # 编译模型，定义相关属性
    Lm1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mse', metrics=['mse'])

    # 在训练的过程中使用回调函数保存模型，请补充完整
    ########## Begin ###########
    Lm1.fit(x_train, y_train, epochs=450, callbacks=[cp_callback])
    ########## End ###########

    loss, acc = Lm1.evaluate(x_train, y_train)
    print("saved model, loss: {:5.2f}".format(loss))

    # 取出最后一次保存的断点
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    # 构建模型加载参数，请补充完整
    ########## Begin ###########
    Lm2 = tf.keras.Model(inputs=inputs, outputs=outputs)
    Lm2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mse')
    Lm2.load_weights(latest)

    return Lm2
    ########## End ###########

##################以下将出现在测试文件中，这里不需要运行，请勿取消注释！！！！！！！！！！！！！！！
# 用模型进行预测，并画出拟合曲线
Lm2 = test_fun()
loss= Lm2.evaluate(x_train, y_train)
print("Restored model, loss: {:5.2f}".format(loss))

forecast=Lm2(x_train)
plt.figure()
plot1 = plt.plot(x_t, y_train, 'b', label='original values')
plot2 = plt.plot(x_t, forecast, 'r', label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)
# plt.savefig('./test_figure/ls_recall/fig.jpg')
