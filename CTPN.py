import keras
import numpy as np

class my3x3Layer(keras.layers.Layer):
    def __init__(self):
        pass

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, input):
        shape = keras.backend.shape(input)
        arr = np.zeros(())

input = keras.layers.Input(shape=(None, None, 3))

#VGG16
conv1_1 = keras.layers.Conv2D(filters=64,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(input)
conv1_2 = keras.layers.Conv2D(filters=64,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv1_1)
pool1 = keras.layers.MaxPool2D(pool_size=(2, 2),
                               strides=(2, 2))(conv1_2)
conv2_1 = keras.layers.Conv2D(filters=128,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(pool1)
conv2_2 = keras.layers.Conv2D(filters=128,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv2_1)
pool2 = keras.layers.MaxPool2D(pool_size=(2, 2),
                               strides=(2, 2))(conv2_1)
conv3_1 = keras.layers.Conv2D(filters=256,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(pool2)
conv3_2 = keras.layers.Conv2D(filters=256,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv3_1)
conv3_3 = keras.layers.Conv2D(filters=256,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv3_2)
pool3 = keras.layers.MaxPool2D(pool_size=(2, 2),
                               strides=(2, 2))(conv3_3)
conv4_1 = keras.layers.Conv2D(filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(pool3)
conv4_2 = keras.layers.Conv2D(filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv4_1)
conv4_3 = keras.layers.Conv2D(filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv4_2)
pool4 = keras.layers.MaxPool2D(pool_size=(2, 2),
                               strides=(2, 2))(conv4_3)
conv5_1 = keras.layers.Conv2D(filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(pool4)
conv5_2 = keras.layers.Conv2D(filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv5_1)
conv5_3 = keras.layers.Conv2D(filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv5_2)
#VGG16 end

#RPN
rpnConv3x3 = keras.layers.Conv2D(filters=512,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 activation="relu")(conv5_3)

Bilstm = keras.layers.Bidirectional(keras.layers.LSTM(128))(rpnConv3x3)
