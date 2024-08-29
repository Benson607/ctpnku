import keras
import keras.layers
import numpy as np
import tensorflow as tf

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

#理論上conv3x3應使用自訂義函數完成，這裡暫時以捲積替代
conv3x3 = keras.layers.Conv2D(filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv5_3)

reshape = tf.reshape(conv3x3, [tf.shape(conv3x3)[0]*tf.shape(conv3x3)[1], tf.shape(conv3x3)[2], tf.shape(conv3x3)[3]])

Bilstm = keras.layers.Bidirectional(keras.layers.LSTM(128))(reshape)

reshape2 = tf.reshape(Bilstm, [tf.shape(conv3x3)[0], tf.shape(conv3x3)[1], tf.shape(conv3x3)[2], tf.shape(conv3x3)[3]])

fc = keras.layers.Dense(units=512,
                        activation="relu")(reshape2)

rpn_bbox_pred = keras.layers.Conv2D(filters=20,
                                    kernel_size=(1, 1),
                                    strides=(1, 1))(fc)

rpn_cls_score = keras.layers.Conv2D(filters=20,
                                    kernel_size=(1, 1),
                                    strides=(1, 1))(fc)

rpn_cls_score_reshape = tf.reshape(rpn_cls_score, (tf.shape(rpn_bbox_pred)[0], 10*tf.shape(rpn_bbox_pred)[1], tf.shape(rpn_bbox_pred)[2], 2))

rpn_cls_prob = keras.layers.Softmax()(rpn_cls_score_reshape)

rpn_cls_prob_reshape = tf.reshape(rpn_cls_prob, (tf.shape(rpn_cls_prob)[0], -1, tf.shape(rpn_cls_prob)[2], 20))

print(tf.shape(rpn_cls_score))
print(tf.shape(rpn_cls_score_reshape))
print(tf.shape(rpn_cls_prob))