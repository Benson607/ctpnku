import keras.regularizers
import keras.layers
import keras.utils
import numpy as np
import tensorflow as tf

weight_decay = 0.0005

def loss():
    pass

input = keras.layers.Input(shape=(None, None, 3))

#VGG16
conv1_1 = keras.layers.Conv2D(name="conv1_1",
                              kernel_regularizer=keras.regularizers.l2(weight_decay),
                              filters=64,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(input)
conv1_2 = keras.layers.Conv2D(name="conv1_2",
                              kernel_regularizer=keras.regularizers.l2(weight_decay),
                              filters=64,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv1_1)
pool1 = keras.layers.MaxPool2D(name="pool1",
                               pool_size=(2, 2),
                               strides=(2, 2))(conv1_2)
conv2_1 = keras.layers.Conv2D(name="conv2_1",
                              kernel_regularizer=keras.regularizers.l2(weight_decay),
                              filters=128,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(pool1)
conv2_2 = keras.layers.Conv2D(name="conv2_2",
                              kernel_regularizer=keras.regularizers.l2(weight_decay),
                              filters=128,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv2_1)
pool2 = keras.layers.MaxPool2D(name="pool2",
                               pool_size=(2, 2),
                               strides=(2, 2))(conv2_1)
conv3_1 = keras.layers.Conv2D(name="conv3_1",
                              kernel_regularizer=keras.regularizers.l2(weight_decay),
                              filters=256,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(pool2)
conv3_2 = keras.layers.Conv2D(name="conv3_2",
                              kernel_regularizer=keras.regularizers.l2(weight_decay),
                              filters=256,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv3_1)
conv3_3 = keras.layers.Conv2D(name="conv3_3",
                              kernel_regularizer=keras.regularizers.l2(weight_decay),
                              filters=256,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv3_2)
pool3 = keras.layers.MaxPool2D(name="pool3",
                               pool_size=(2, 2),
                               strides=(2, 2))(conv3_3)
conv4_1 = keras.layers.Conv2D(name="conv4_1",
                              kernel_regularizer=keras.regularizers.l2(weight_decay),
                              filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(pool3)
conv4_2 = keras.layers.Conv2D(name="conv4_2",
                              kernel_regularizer=keras.regularizers.l2(weight_decay),
                              filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv4_1)
conv4_3 = keras.layers.Conv2D(name="conv4_3",
                              kernel_regularizer=keras.regularizers.l2(weight_decay),
                              filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv4_2)
pool4 = keras.layers.MaxPool2D(name="pool4",
                               pool_size=(2, 2),
                               strides=(2, 2))(conv4_3)
conv5_1 = keras.layers.Conv2D(name="conv5_1",
                              kernel_regularizer=keras.regularizers.l2(weight_decay),
                              filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(pool4)
conv5_2 = keras.layers.Conv2D(name="conv5_2",
                              kernel_regularizer=keras.regularizers.l2(weight_decay),
                              filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv5_1)
conv5_3 = keras.layers.Conv2D(name="conv5_3",
                              kernel_regularizer=keras.regularizers.l2(weight_decay),
                              filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv5_2)
#VGG16 end

#rpn
#理論上conv3x3應使用自訂義函數完成，這裡暫時以捲積替代
conv3x3 = keras.layers.Conv2D(name="3x3",
                              kernel_regularizer=keras.regularizers.l2(weight_decay),
                              filters=512,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation="relu")(conv5_3)
N, H, W, C = tf.shape(conv3x3,
                      name="get3x3Shape")
reshape = tf.reshape(conv3x3,
                     [N*H, W, C],
                     name="reshapeForLstm")
Bilstm = keras.layers.Bidirectional(name="lstm",
                                    layer=keras.layers.LSTM(128))(reshape)
reshape2 = tf.reshape(Bilstm,
                      [N, H, W, C],
                      name="reshapeForFc")
fc = keras.layers.Dense(name="fc",
                        units=512,
                        activation="relu")(reshape2)

rpn_bbox_pred = keras.layers.Conv2D(name="rpn_bbox_pred",
                                    kernel_regularizer=keras.regularizers.l2(weight_decay),
                                    filters=20,
                                    kernel_size=(1, 1),
                                    strides=(1, 1))(fc)

rpn_cls_score = keras.layers.Conv2D(name="rpn_cls_score",
                                    kernel_regularizer=keras.regularizers.l2(weight_decay),
                                    filters=20,
                                    kernel_size=(1, 1),
                                    strides=(1, 1))(fc)
N, H, W, C = tf.shape(rpn_bbox_pred,
                      name="getRpn_bbox_predShape")
rpn_cls_score_reshape = tf.reshape(rpn_cls_score,
                                   [N, 10*H, W, 2],
                                   name="rpn_cls_score_reshape")
rpn_cls_prob = keras.layers.Softmax(name="rpn_cls_prob")(rpn_cls_score_reshape)
N, H, W, C = tf.shape(rpn_cls_prob,
                      name="getRpn_cls_probShape")
rpn_cls_prob_reshape = tf.reshape(rpn_cls_prob,
                                  [N, -1, W, 20],
                                  name="rpn_cls_prob_reshape")

print(tf.shape(rpn_bbox_pred))
print(tf.shape(rpn_cls_score))
print(tf.shape(rpn_cls_score_reshape))
print(tf.shape(rpn_cls_prob))

model = keras.Model(inputs=input, outputs=[rpn_bbox_pred, rpn_cls_prob_reshape])

model.summary()
keras.utils.plot_model(model, "model.png")

model.compile(optimizer='adam', 
              loss={'output_1': 'categorical_crossentropy', 
                    'output_2': 'binary_crossentropy'})