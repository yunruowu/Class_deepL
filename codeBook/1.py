import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import time
import os

# 数据准备，构造训练集
x_t = np.arange(-23 / 18, (2 * np.pi - 23) / 18, 2 * np.pi / 18 / 2000, dtype=np.float32)
x_t = x_t[:, np.newaxis]
x_train = np.concatenate((x_t, np.power(x_t, 2), np.power(x_t, 3)), axis=1)
y_train = np.cos(18 * x_t + 23)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(64)

def test_fun():
    # 模型定义，请补全构建模型的代码
    ########## Begin ###########
    # 步骤1 定义输入层
    inputs = tf.keras.Input(shape=(3,), name='data')
    # 步骤2 定义输出层，提供线性回归的参数
    outputs = tf.keras.layers.Dense(units=1, input_dim=3)(inputs)
    # 步骤3 创建模型包装输入层和输出层
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linear')
    ########## End ###########
    
    
    # 创建损失函数
    loss_object = tf.keras.losses.MeanSquaredError()
    # 创建优化器
    optimizer = tf.keras.optimizers.Adam(0.1)
    # 创建训练损失计算方法
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    # 创建测试损失计算方法
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    
    
    # 定义 python 函数封装训练迭代过程，请补充
    ########## Begin ###########
    @tf.function
    def train_step(data, labels):
        with tf.GradientTape() as tape:
            predictions = model(data, training = True)
            loss = train_loss(data, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss.update_state(loss)
        train_metric
        return
    ########## End ###########
    
    
    # 定义 python 函数为静态图封装测试迭代过程，请补充
    ########## Begin ###########
    @tf.function
    def test_step(data, labels):
        
        return
    ########## End ###########
    
    
    EPOCHS = 800
    
    # 自定义循环对模型进行训练
    ########## Begin ###########
    for epoch in range(EPOCHS):
        start = time.time()
        # 在下一个epoch开始时，重置评估指标
        train_loss.reset_states()
        test_loss.reset_states()
    
        # 模型在训练集上进行一次训练，请补充
        for data, labels in train_dataset:
            
    
        # 模型在测试集上进行一次验证，请补充
        for test_data, test_labels in train_dataset:
            
            
    
        end = time.time()
        # 输出训练情况
        template = 'Epoch {}, Loss: {:.3f},  Test Loss: {:.3f}，Time used: {:.2f}'
        # print(template.format(epoch + 1,
        #                       train_loss.result(),
        #                       test_loss.result(), end - start))
    return model
    ########## End ###########


##################以下将出现在测试文件中，这里不需要运行，请勿取消注释！！！！！！！！！！！！！！！
# 模型预测，并画出拟合曲线
#model.compile(loss='mse')
#loss = model.evaluate(x_train, y_train)
#predictions = model(x_train)
#plt.figure()
#plot1 = plt.plot(x_t, y_train, 'b', label='original values')
#plot2 = plt.plot(x_t, predictions, 'r', label='polyfit values')
#plt.xlabel('x axis')
#plt.ylabel('y axis')
#plt.legend(loc=4)
#plt.savefig('./test_figure/train_loop/fig.jpg')