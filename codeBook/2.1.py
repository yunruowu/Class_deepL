# coding=utf-8

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, concatenate, Input, add

import numpy as np

import os

num_classes = 19


# 由于需要本地读取MNIST,不用MNISTdatasets
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
def PreparePlusData():
    # 本地读取MNIST流程
    # path = './dataset/mnist.npz'
    # f = np.load(path)
    # x_train, y_train = f['x_train'], f['y_train']
    # x_test, y_test = f['x_test'], f['y_test']
    #
    # f.close()
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    h = x_train.shape[1] // 2
    w = x_train.shape[2] // 2

    # 为便于评测，图像尺寸缩小为原来的一半
    x_train = np.expand_dims(x_train, axis=-1)
    x_train = tf.image.resize(x_train, [h, w]).numpy()  # if we want to resize
    x_test = np.expand_dims(x_test, axis=-1)
    x_test = tf.image.resize(x_test, [h, w]).numpy()  # if we want to resize

    # 图像归一化,易于网络学习
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 注意，即使同一个数字也有很多不同图像，
    # 需要产生的是尽可能多的数字图像样例对的组合，
    # 下面会采用两个随机列输入配对的方式去产生
    # 因此，为扩充更多的图像对加法实例，先扩充两个随机输入列的长度
    len_train = len(x_train)
    len_test = len(x_test)
    len_ext_train = len_train * 3
    len_ext_test = len_test * 3

    # 由于本实训采用线性全连接网络，需要将图片拉伸为一维向量
    x_train = x_train.reshape((len_train, -1))
    x_test = x_test.reshape((len_test, -1))

    # 由于MNIST是按数字顺序排列，故将其打乱，通过随机交叉样本产生更多随机的图片数字加法组合
    left_train_choose = np.random.choice(len_train, len_ext_train, replace=True)
    right_train_choose = np.random.choice(len_train, len_ext_train, replace=True)
    left_test_choose = np.random.choice(len_test, len_ext_test, replace=True)
    right_test_choose = np.random.choice(len_test, len_ext_test, replace=True)

    x_train_l = x_train[left_train_choose]
    x_train_r = x_train[right_train_choose]
    x_test_l = x_test[left_test_choose]
    x_test_r = x_test[right_test_choose]
    # print(x_train_r)
    # print(x_train)
    # ！！！！！！注意，本题标签不采用one-hot编码
    y_train = y_train[left_train_choose] + y_train[right_train_choose]
    y_test = y_test[left_test_choose] + y_test[right_test_choose]
    # WORK1: --------------BEGIN-------------------
    # 请补充完整训练集和测试集的产生方法：
    x_train_l_r = tf.data.Dataset.from_tensor_slices((x_train_l, x_train_r))
    y_train_data = tf.data.Dataset.from_tensor_slices(y_train)
    # print(x_train_l_r)
    train_datasets = tf.data.Dataset.zip((x_train_l_r, y_train_data)).batch(64)
    print(train_datasets)
    x_test_l_r = tf.data.Dataset.from_tensor_slices((x_test_l, x_test_r))
    y_test_data = tf.data.Dataset.from_tensor_slices(y_test)

    test_datasets = tf.data.Dataset.zip((x_test_l_r, y_test_data)).batch(64)
    # # WORK1: ---------------END--------------------
    # print(train_datasets)
    # print(test_datasets)
    return train_datasets, test_datasets


# PreparePlusData()
# WORK2: --------------BEGIN-------------------
# 请补充完整自定义层实现 BiasPlusLayer([input1,input2])=input1+input2+bias：
class BiasPlusLayer(keras.layers.Layer):
    # 2.1如果变量不随输入维度的改变而改变，可以在初始化__init__中用add_weight添加bias变量
    # 否则，变量的添加在build方法中根据input_shape实现
    # 请在__init__中添加变量self.bias，实现BiasPlusLayer([input1,input2])=input1+input2+bias 功能
    # 注意，bias维度需和需要相加的input1，input2一致
    def __init__(self, num_outputs, **kwargs):
        super(BiasPlusLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.bias = self.add_weight(shape=(1, num_outputs), initializer='random_normal', trainable=True)

    def build(self, input_shape):
        super(BiasPlusLayer, self).build(input_shape)  # Be sure to call this somewhere!

    # 2.2在调用中实现input1+input2+bias
    def call(self, input):
        return tf.add(tf.add(input[0] ,input[1]), self.bias)


# WORK2: ---------------END--------------------

# WORK3: --------------BEGIN-------------------
# 请参考所给网络结构图，补充完整共享参数孪生网络siamese_net的实现：
# 注意，我们用比较图片的方法来评测网络结构是否正确
# 所以网络结构中的参数维度、名称等需和参考图中一致，否则不能通过评测
def BuildModel():
    # 3.1 shared_base是共享参数的骨干网，用sequential方式搭建
    # 其中包含两层64个节点的Dense全连接层，激活用relu

    # 注意！！！如果要让plot_model打印出嵌套的Sequential内部结构,
    # 需要给出输入的维度，例如，在第一层Desse中加入参数：input_shape=(xxx,)
    # (注意一维向量大小这里一定写为"xxx,")
    shared_base = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(196,)),
        tf.keras.layers.Dense(64, input_shape=(196, 0), activation='relu'),
        tf.keras.layers.Dense(64, input_shape=(196, 0), activation='relu')
    ], name='seq1')

    # 3.2 x1,x2 分别表示一对图片中两个图像输入，请补充完整输入维度信息
    x1 = Input(shape=(196,))
    x2 = Input(shape=(196,))
    # 3.3 b1,b2 表示应用共享骨干网的两个处理通道
    b1 = shared_base(x1)
    b2 = shared_base(x2)
    # 3.4 b1,b2 的处理结果放入我们的自定义层做b1+b2+bias处理
    # 注意，对于多个输入通道，输入用列表表示
    # 请补充BiasPlusLayer参数及输入
    b = BiasPlusLayer(64, name='BiasPlusLayer')([b1, b2])

    # 3.5 加法实际用分类实现，用softmax激活，这之前有个全连接，请补充相关参数和输入
    output = Dense(19, input_shape=(64,), activation='softmax')(b)
    # 3.6 最后构建 Keras.Model,请补充完整输入输出
    siamese_net = Model(inputs=[x1,x2],
                        outputs = output)
    # 打印网络结构用于测试，请不要修改地址和参数
    # plot_model(siamese_net, to_file='./test_figure/step1/siamese_net.png', show_shapes=True, expand_nested=True)

    return siamese_net


# WORK3: ---------------END--------------------


# WORK4: --------------BEGIN-------------------
# 实例化网络并进行训练
def test_fun():
    siamese_net = BuildModel()
    # 4.1 配置模型，我们的加法用分类实现，故选择分类loss (注意根据标签y的形式，选择合适的loss)，及评测metric，
    # 其他训练参数不用变
    siamese_net.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                        metrics=['accuracy'])
    # 在给定训练参数下，一般12个迭代就可以完成训练任务（val_acc>0.7），用时200多秒
    epochs = 12
    train_datasets, test_datasets = PreparePlusData()
    # 4.2 配置训练参数，开始训练，
    history = siamese_net.fit(train_datasets, epochs=epochs, validation_data=test_datasets, verbose=2)
    # 返回要素都是评测所需，请不要更改
    return siamese_net, history, test_datasets


# WORK4: ---------------END--------------------


# 以下为测试代码

'''
from NeuroPlus_work import PreparePlusData,test_fun'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import os

siamese_net, history, test_datasets = test_fun()

# 1.用图片对比的方法测试网络结构是否正确
test_img = mpimg.imread('./test_figure/step1/siamese_net.png')
answer_img = mpimg.imread('./answer/step1/answer.png')
assert ((answer_img == test_img).all())
print('Network pass!')
# 2.测试BiasPlusLayer层功能
l = siamese_net.get_layer('BiasPlusLayer')
bias = l.get_weights()
r = l([1., 2.]).numpy()
r_np = 1. + 2. + bias[0]
assert ((r == r_np).all())
print('BiasPlusLayer pass!')
# 3.打印样例结果
iter_test = iter(test_datasets)
b_test = next(iter_test)
r_test = siamese_net.predict(b_test[0])

fig, ax = plt.subplots(nrows=2, ncols=5, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(5):
    img = b_test[0][0][i].numpy().reshape(14, 14)
    ax[i].set_title('Label: ' + str(b_test[1][i].numpy()))
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
for i in range(5):
    img = b_test[0][1][i].numpy().reshape(14, 14)
    ax[i + 5].set_title('Prediction: ' + str(np.argmax(r_test[i])))
    ax[i + 5].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.savefig("./test_figure/step1/PredictionExample.png")
print('Result pass!')
# 4.测试网络训练是否达标
if history.history['val_accuracy'][-1] > 0.7:
    print("Success!")
# '''
