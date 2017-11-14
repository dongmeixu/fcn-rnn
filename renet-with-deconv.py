#  coding: utf-8
import matplotlib
import time

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
import random
import cv2
import os
import keras
from shapely.geometry import MultiPolygon, Polygon
from shapely.wkt import loads as wkt_loads
from keras.layers import Input, Dropout, LSTM, UpSampling2D, ConvLSTM2D, Convolution2D
from keras.layers import merge, Flatten, TimeDistributed, Bidirectional, CuDNNLSTM
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose
from keras.models import Model
from keras.layers.core import Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score

from collections import defaultdict
import tensorflow as tf

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

FLAGS = tf.flags.FLAGS
# tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_integer("image_size", "160", "image size for training")
tf.flags.DEFINE_integer("image_channels", "3", "image channels for training")
tf.flags.DEFINE_integer("mask_channels", "9", "mask channels for output")

tf.flags.DEFINE_string("data_dir", "/media/files/xdm/ningxia-hn1/dataset/", "path to dataset")
tf.flags.DEFINE_string("model_dir", "renet-with-deconv/model/", "Path to vgg model mat")
tf.flags.DEFINE_string("npy_dir", "/media/files/xdm/ningxia-hn1/dataset/npy/", "path to dataset")
tf.flags.DEFINE_string("mask_dir", "renet-with-deconv/mask/", "Path to mask-data")

TWF_FILE = FLAGS.data_dir + 'hn1.tfw'
DF = pd.read_csv(FLAGS.data_dir + 'hn1_train_wkt.csv')
GS = pd.read_csv(FLAGS.data_dir + 'hn1_grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

# tf.flags.DEFINE_string("model_dir", "D:\\Project\\ML_DL\\DeepLearning\\kaggle\ningxia-hn1\\model\\", "Path to vgg model mat")
# tf.flags.DEFINE_string("npy_dir", "D:\\Project\\ML_DL\\DeepLearning\\kaggle\\ningxia-hn1\\npy\\", "Path to npy-data")
# tf.flags.DEFINE_string("mask_dir", "D:\\Project\\ML_DL\\DeepLearning\\kaggle\\ningxia-hn1\\mask\\", "Path to mask-data")
#
# windows_dir = r"F:\三百米裁切\hn1"
# TWF_FILE = windows_dir + '\hn1.tfw'
# DF = pd.read_csv(windows_dir + '\hn1_train_wkt.csv')
# GS = pd.read_csv(windows_dir + '\hn1_grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)


# 参数设置
smooth = 1e-12


def read_tif(image_id):
    filename = FLAGS.data_dir + '{}.tif'.format(image_id)
    img = tiff.imread(filename)
    return img


# “对比度增强”，类似于使用QGIS打开图像时的默认设置
# "Contrast enhancement", similar to the default when opening an image using QGIS.
def stretch_n(bands, lower_percent=0, higher_percent=100):
    out = np.zeros_like(bands, dtype=np.float32)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.float32)


# running jaccard_coef function............
# y_true.shape:  (?, ?, ?, ?)
# y_pred.shape:  (?, 160, 160, 9)
def jaccard_coef(y_true, y_pred):
    print("running jaccard_coef function............")
    print("y_true.shape: ", y_true.shape)
    print("y_pred.shape: ", y_pred.shape)
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


# running jaccard_coef_int function............
# y_true.shape:  (?, ?, ?, ?)
# y_pred.shape:  (?, 160, 160, 9)
# y_pred_pos.shape:  (?, 160, 160, 9)
def jaccard_coef_int(y_true, y_pred):
    print("running jaccard_coef_int function............")
    print("y_true.shape: ", y_true.shape)
    print("y_pred.shape: ", y_pred.shape)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    print("y_pred_pos.shape: ", y_pred_pos.shape)

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def get_patches(img, msk, amt=10000, aug=True):
    is2 = int(1.0 * FLAGS.image_size)
    xm, ym = img.shape[0] - is2, img.shape[1] - is2

    x, y = [], []

    tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001]
    for i in range(amt):
        xc = random.randint(0, xm)
        yc = random.randint(0, ym)

        im = img[xc:xc + is2, yc:yc + is2]
        ms = msk[xc:xc + is2, yc:yc + is2]

        for j in range(FLAGS.mask_channels):
            sm = np.sum(ms[:, :, j])
            if 1.0 * sm / is2 ** 2 > tr[j]:
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        im = im[::-1]
                        ms = ms[::-1]
                    if random.uniform(0, 1) > 0.5:
                        im = im[:, ::-1]
                        ms = ms[:, ::-1]

                x.append(im)
                y.append(ms)

    # this is standard normalization step to set range [-1, 1] for all image.
    x, y = 2 * np.transpose(x, (0, 1, 2, 3)) - 1, np.transpose(y, (0, 1, 2, 3))
    print(x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y))
    # (11645, 160, 160, 3) (11645, 160, 160, 9) 1.0 -1.0 1.0 0.0
    return x, y


# Method1 : with deconvnet
# def get_fcnrnn():
#     with tf.device("/gpu:0"):
#         inputs = Input((FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels))
#         conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(inputs)
#         conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(conv1_1)
#         conv1_2 = BatchNormalization()(conv1_2)
#         pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1_2)
#
#         conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(pool1)
#         conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(conv2_1)
#         conv2_2 = BatchNormalization()(conv2_2)
#         pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2_2)
#
#         conv3_1 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(pool2)
#         conv3_2 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv3_1)
#         conv3_3 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv3_2)
#         conv3_3 = BatchNormalization()(conv3_3)
#         pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3_3)
#
#         conv4_1 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool3)
#         conv4_2 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv4_1)
#         conv4_3 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv4_2)
#         conv4_3 = BatchNormalization()(conv4_3)
#         pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4_3)
#
#         conv5_1 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool4)
#         conv5_2 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv5_1)
#         conv5_3 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv5_2)
#         conv5_3 = BatchNormalization()(conv5_3)
#         pool5 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(conv5_3)
#         print(pool5.shape)  # (?, 9, 9, 512)
#
#         # fc6 = Conv2D(filters=4096, kernel_size=(5, 5), activation="relu")(pool5)
#         # fc7 = Conv2D(filters=4096, kernel_size=(1, 1), activation="relu")(fc6)
#
#         reshape = Reshape((1, 9, 9, 512))(pool5)
#         print(reshape.shape)
#
#         # # the first ReNet
#         # # LSTM(input_shape=(16, 16), return_sequences=True, units=240)
#         lstm1 = ConvLSTM2D(filters=240, kernel_size=(2, 2), dropout=0.25, recurrent_dropout=0.25,
#                            return_sequences=True)(reshape)
#         lstm1 = ConvLSTM2D(filters=240, kernel_size=(2, 2), dropout=0.25, recurrent_dropout=0.25,
#                            return_sequences=True)(lstm1)
#         lstm1 = ConvLSTM2D(filters=240, kernel_size=(2, 2), dropout=0.25, recurrent_dropout=0.25,
#                            return_sequences=True)(lstm1)
#         lstm1 = ConvLSTM2D(filters=240, kernel_size=(2, 2), dropout=0.25, recurrent_dropout=0.25,
#                            return_sequences=True)(lstm1)
#         print(lstm1.shape)
#
#         lstm2 = ConvLSTM2D(filters=240, kernel_size=(2, 2), dropout=0.25, recurrent_dropout=0.25,
#                            return_sequences=True)(lstm1)
#         lstm2 = ConvLSTM2D(filters=240, kernel_size=(2, 2), dropout=0.25, recurrent_dropout=0.25,
#                            return_sequences=True)(lstm2)
#         lstm2 = ConvLSTM2D(filters=240, kernel_size=(2, 2), dropout=0.25, recurrent_dropout=0.25,
#                            return_sequences=True)(lstm2)
#         lstm2 = ConvLSTM2D(filters=240, kernel_size=(2, 2), dropout=0.25, recurrent_dropout=0.25)(lstm2)
#
#         print(lstm2.shape)
#
#         deconv_fc6 = Conv2DTranspose(filters=512, kernel_size=(5, 5))(lstm2)
#         deconv_fc6 = BatchNormalization()(deconv_fc6)
#         unpool5 = UpSampling2D(size=(2, 2))(deconv_fc6)
#
#         deconv5_1 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(unpool5)
#         deconv5_2 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(deconv5_1)
#         deconv5_3 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(deconv5_2)
#         deconv5_3 = BatchNormalization()(deconv5_3)
#         unpool4 = UpSampling2D(size=(2, 2))(deconv5_3)
#
#         deconv4_1 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(unpool4)
#         deconv4_2 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(deconv4_1)
#         deconv4_3 = Conv2DTranspose(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(deconv4_2)
#         deconv4_3 = BatchNormalization()(deconv4_3)
#         unpool3 = UpSampling2D(size=(2, 2))(deconv4_3)
#
#         deconv3_1 = Conv2DTranspose(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(unpool3)
#         deconv3_2 = Conv2DTranspose(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(deconv3_1)
#         deconv3_3 = Conv2DTranspose(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(deconv3_2)
#         deconv3_3 = BatchNormalization()(deconv3_3)
#         unpool2 = UpSampling2D(size=(2, 2))(deconv3_3)
#
#         deconv2_1 = Conv2DTranspose(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(unpool2)
#         deconv2_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(deconv2_1)
#         deconv2_2 = BatchNormalization()(deconv2_2)
#         unpool1 = UpSampling2D(size=(2, 2))(deconv2_2)
#
#         deconv1_1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(unpool1)
#         deconv1_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(deconv1_1)
#         deconv1_2 = BatchNormalization()(deconv1_2)
#         output = Conv2DTranspose(filters=FLAGS.mask_channels, kernel_size=(1, 1), activation='sigmoid')(deconv1_2)
#
#         model = Model(input=inputs, output=output)
#         model.compile(optimizer=Adam(), loss='binary_crossentropy',
#                       metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
#         return model


# with deconv and 双向RNN
# def get_fcnrnn():
#     with tf.device("/gpu:0"):
#         inputs = Input((FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels))
#         conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(inputs)
#         conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(conv1_1)
#         conv1_2 = BatchNormalization()(conv1_2)
#         pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1_2)
#
#         conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(pool1)
#         conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(conv2_1)
#         conv2_2 = BatchNormalization()(conv2_2)
#         pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2_2)
#
#         conv3_1 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(pool2)
#         conv3_2 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv3_1)
#         conv3_3 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv3_2)
#         conv3_3 = BatchNormalization()(conv3_3)
#         pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3_3)
#
#         conv4_1 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool3)
#         conv4_2 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv4_1)
#         conv4_3 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv4_2)
#         conv4_3 = BatchNormalization()(conv4_3)
#         pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4_3)
#
#         conv5_1 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool4)
#         conv5_2 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv5_1)
#         conv5_3 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv5_2)
#         conv5_3 = BatchNormalization()(conv5_3)
#         pool5 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(conv5_3)
#         print(pool5.shape)  # (?, 9, 9, 512)
#
#         # fc6 = Conv2D(filters=4096, kernel_size=(5, 5), activation="relu")(pool5)
#         # fc7 = Conv2D(filters=4096, kernel_size=(1, 1), activation="relu")(fc6)
#
#         reshape = Reshape((1, 9, 9, 512))(pool5)
#         print(reshape.shape)
#
#         # # the first ReNet
#         # # LSTM(input_shape=(16, 16), return_sequences=True, units=240)
#         lstm1 = ConvLSTM2D(filters=512, kernel_size=(2, 2), activation='relu', dropout=0.25, recurrent_dropout=0.25,
#                            return_sequences=True)(reshape)
#         lstm1 = ConvLSTM2D(filters=512, kernel_size=(2, 2), activation='relu', dropout=0.25, recurrent_dropout=0.25,
#                            return_sequences=True)(lstm1)
#         lstm1 = ConvLSTM2D(filters=512, kernel_size=(2, 2), activation='relu', dropout=0.25, recurrent_dropout=0.25,
#                            return_sequences=True)(lstm1)
#         lstm1 = ConvLSTM2D(filters=512, kernel_size=(2, 2), activation='relu', dropout=0.25, recurrent_dropout=0.25,
#                            return_sequences=True)(lstm1)
#         print(lstm1.shape)
#
#         lstm2 = ConvLSTM2D(filters=512, kernel_size=(2, 2), activation='relu', dropout=0.25, recurrent_dropout=0.25,
#                            return_sequences=True)(lstm1)
#         lstm2 = ConvLSTM2D(filters=512, kernel_size=(2, 2), activation='relu', dropout=0.25, recurrent_dropout=0.25,
#                            return_sequences=True)(lstm2)
#         lstm2 = ConvLSTM2D(filters=512, kernel_size=(2, 2), activation='relu', dropout=0.25, recurrent_dropout=0.25,
#                            return_sequences=True)(lstm2)
#         lstm2 = ConvLSTM2D(filters=512, kernel_size=(2, 2), activation='relu', dropout=0.25, recurrent_dropout=0.25)(lstm2)
#
#         print(lstm2.shape)
#
#         deconv_fc6 = Conv2DTranspose(filters=512, kernel_size=(5, 5))(lstm2)
#         deconv_fc6 = BatchNormalization()(deconv_fc6)
#         unpool5 = UpSampling2D(size=(2, 2))(deconv_fc6)
#
#         deconv5_1 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(unpool5)
#         deconv5_2 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(deconv5_1)
#         deconv5_3 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(deconv5_2)
#         deconv5_3 = BatchNormalization()(deconv5_3)
#         unpool4 = UpSampling2D(size=(2, 2))(deconv5_3)
#
#         deconv4_1 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(unpool4)
#         deconv4_2 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(deconv4_1)
#         deconv4_3 = Conv2DTranspose(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(deconv4_2)
#         deconv4_3 = BatchNormalization()(deconv4_3)
#         unpool3 = UpSampling2D(size=(2, 2))(deconv4_3)
#
#         deconv3_1 = Conv2DTranspose(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(unpool3)
#         deconv3_2 = Conv2DTranspose(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(deconv3_1)
#         deconv3_3 = Conv2DTranspose(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(deconv3_2)
#         deconv3_3 = BatchNormalization()(deconv3_3)
#         unpool2 = UpSampling2D(size=(2, 2))(deconv3_3)
#
#         deconv2_1 = Conv2DTranspose(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(unpool2)
#         deconv2_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(deconv2_1)
#         deconv2_2 = BatchNormalization()(deconv2_2)
#         unpool1 = UpSampling2D(size=(2, 2))(deconv2_2)
#
#         deconv1_1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(unpool1)
#         deconv1_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(deconv1_1)
#         deconv1_2 = BatchNormalization()(deconv1_2)
#         output = Conv2DTranspose(filters=FLAGS.mask_channels, kernel_size=(1, 1), activation='sigmoid')(deconv1_2)
#
#         model = Model(input=inputs, output=output)
#         model.compile(optimizer=Adam(), loss='binary_crossentropy',
#                       metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
#         return model


def get_fcnrnn():
    # tensorflow channel-last
    with tf.device("/gpu:0"):
        inputs = Input((FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels))
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="elu")(inputs)
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="elu")(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="elu")(pool1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="elu")(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="elu")(pool2)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="elu")(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(pool3)
        conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="elu")(pool4)
        conv5 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="elu")(conv5)
        conv5 = BatchNormalization()(conv5)
        print(conv5.shape)  # (?, 10, 10, 512)#
        reshape = Reshape((1, -1))(conv5)
        # reshape = Flatten()(reshape)
        print("reshape.shape: ", reshape.shape)

        #  the first ReNet
        # blstm1 = Bidirectional(ConvLSTM2D(filters=512, kernel_size=(3, 3), activation='relu', dropout=0.25,
        #                                   recurrent_dropout=0.25, return_sequences=True), merge_mode='sum')(reshape)
        # blstm1 = Bidirectional(ConvLSTM2D(filters=512, kernel_size=(2, 2), activation='relu', dropout=0.25,
        #                                   recurrent_dropout=0.25, return_sequences=True), merge_mode='sum')(blstm1)
        blstm1 = CuDNNLSTM(800, return_sequences=True)(reshape)
        blstm1 = CuDNNLSTM(800, return_sequences=True)(blstm1)
        print("blstm1.shape: ", blstm1.shape)
        # (?, ?, 7, 7, 1024)

        # blstm2 = Bidirectional(ConvLSTM2D(filters=512, kernel_size=(2, 2), activation='relu', dropout=0.25,
        #                                   recurrent_dropout=0.25, return_sequences=True), merge_mode='sum')(blstm1)
        # blstm2 = Bidirectional(ConvLSTM2D(filters=512, kernel_size=(2, 2), activation='relu', dropout=0.25,
        #                                   recurrent_dropout=0.25), merge_mode='sum')(blstm2)
        blstm2 = CuDNNLSTM(800, return_sequences=True)(blstm1)
        blstm2 = CuDNNLSTM(800, return_sequences=False)(blstm2)
        print("blstm2.shape: ", blstm2.shape)
        # (?, ?, 5, 5, 1024)
        print("conv4.shape: ", conv4.shape)
        # (?, 20, 20, 256)
        blstm2 = Reshape((5, 5, 32))(blstm2)
        print("blstm2.shape: ", blstm2.shape)
        blstmcombine = merge([UpSampling2D(size=(4, 4))(blstm2), conv4], mode='concat')
        # (?, 20, 20, 1280)
        print("conv5.shape: ", conv5.shape)
        # (?, 10, 10, 512)
        up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4, blstmcombine], mode='concat')
        # (?, 20, 20, 2048)
        conv6 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="elu")(up6)
        conv6 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="elu")(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat')
        conv7 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="elu")(up7)
        conv7 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="elu")(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat')
        conv8 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="elu")(up8)
        conv8 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="elu")(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat')
        print(up9.shape)
        conv9 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="elu")(up9)
        conv9 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="elu")(conv9)
        conv9 = BatchNormalization()(conv9)
        print(conv9.shape)

        conv10 = Conv2D(filters=FLAGS.mask_channels, kernel_size=(3, 3), padding="same", activation="sigmoid")(conv9)
        conv10 = BatchNormalization()(conv10)
        model = Model(input=inputs, output=conv10)
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
        return model


def calc_jacc(model):
    img = np.load(FLAGS.npy_dir + 'x_tmp_%d.npy' % FLAGS.mask_channels)
    msk = np.load(FLAGS.npy_dir + 'y_tmp_%d.npy' % FLAGS.mask_channels)

    prd = model.predict(img, batch_size=4)
    print("prd.shape, msk.shape: ", prd.shape, msk.shape)
    avg, trs = [], []

    for i in range(FLAGS.mask_channels):
        t_msk = msk[:, :, :, i]
        t_prd = prd[:, :, :, i]
        t_msk = t_msk.reshape(msk.shape[0] * msk.shape[1], msk.shape[2])
        t_prd = t_prd.reshape(msk.shape[0] * msk.shape[1], msk.shape[2])
        m, b_tr = 0, 0
        for j in range(10):
            tr = j / 10.0
            pred_binary_mask = t_prd > tr

            jk = jaccard_similarity_score(t_msk, pred_binary_mask)
            if jk > m:
                m = jk
                b_tr = tr
        print(i, m, b_tr)
        avg.append(m)
        trs.append(b_tr)

    score = sum(avg) / 10.0
    return score, trs


def mask_for_polygons(polygons, im_size):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    print(exteriors)
    print("********")
    print(interiors)
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    print(img_mask)
    return img_mask


def mask_to_polygons(mask, epsilon=5, min_area=1.):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    print(all_polygons)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def get_scalers(im_size, x_max, y_min):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    h, w = float(h), float(w)
    w_ = 1.0 * w * (w / (w + 1))
    h_ = 1.0 * h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


def train_net():
    start_time = time.time()
    print("start train net")
    # 读取训练以及验证数据
    '''linux'''
    x_val, y_val = np.load(FLAGS.npy_dir + 'x_tmp_%d.npy' % FLAGS.mask_channels), \
                   np.load(FLAGS.npy_dir + 'y_tmp_%d.npy' % FLAGS.mask_channels)
    img = np.load(FLAGS.npy_dir + 'x_trn_%d.npy' % FLAGS.mask_channels)
    msk = np.load(FLAGS.npy_dir + 'y_trn_%d.npy' % FLAGS.mask_channels)

    print("img.shape: ", img.shape)
    print("msk.shape: ", msk.shape)
    # img.shape:  (2035, 2035, 3)
    # msk.shape:  (2035, 2035, 9)

    x_trn, y_trn = get_patches(img, msk)

    model = get_fcnrnn()
    print(model.summary())
    # model.load_weights(FLAGS.model_dir + 'epoch_3_unet_9_jk0.8385')
    model_checkpoint = ModelCheckpoint(FLAGS.model_dir + 'unet_tmp.hdf5', monitor='loss', save_best_only=True)
    nb_epoch = 50
    for i in range(1):
        history = model.fit(x_trn, y_trn, batch_size=16, nb_epoch=nb_epoch, verbose=1,
                            shuffle=True, callbacks=[model_checkpoint], validation_data=(x_val, y_val))

        plt.plot()
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig(FLAGS.model_dir + "model accuracy with dropout and elu.png")
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig(FLAGS.model_dir + "model loss with dropout and elu.png")

        del x_trn
        del y_trn
        x_trn, y_trn = get_patches(img, msk)
        score, trs = calc_jacc(model)
        print('val jk', score)
        model.save_weights(FLAGS.model_dir + 'rnn_with_dropout_and_elu_unet_9_jk%.4f' % score)

    end_time = time.time()
    print("模型训练时间：", (end_time - start_time) * 1000, "ms")
    return model


def predict_id(id, model, trs):
    img = read_tif(id)
    x = stretch_n(img)
    cnv = np.zeros((2400, 2400, 3)).astype(np.float32)
    prd = np.zeros((2400, 2400, FLAGS.mask_channels)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x

    for i in range(0, 15):
        line = []
        for j in range(0, 15):
            line.append(cnv[i * FLAGS.image_size:(i + 1) * FLAGS.image_size, j * FLAGS.image_size:(j + 1) * FLAGS.image_size])

        x = 2 * np.transpose(np.array(line), (0, 1, 2, 3)) - 1
        print("x.shape: ", x.shape)
        tmp = model.predict(x, batch_size=4)
        print("tmp.shape: ", tmp.shape[0])
        for j in range(tmp.shape[0]):
            prd[i * FLAGS.image_size:(i + 1) * FLAGS.image_size, j * FLAGS.image_size:(j + 1) * FLAGS.image_size, :] = tmp[j]

    # trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    for i in range(FLAGS.mask_channels):
        prd[i] = prd[i] > trs[i]

    return prd[:img.shape[0], :img.shape[1], :]


def predict_test(model, trs):
    print("predict test")
    for i, id in enumerate(sorted(set(GS['ImageId'].tolist()))):
        msk = predict_id(id, model, trs)
        np.save('msk/9_%s' % id, msk)
        if i % 100 == 0:
            print(i, id)


def check_predict(id='hn1'):
    model = get_fcnrnn()
    # model.load_weights(FLAGS.model_dir + 'epoch_with_biRNN_unet_9_jk0.4729')
    model.load_weights(FLAGS.model_dir + 'rnn_unet_9_jk0.8023')
    msk = predict_id(id, model, trs=[0.95, 0.95, 0.95, 0.1, 0.1, 0.1, 0.4, 0.4, 0.001])
    np.save(FLAGS.model_dir + 'msk_hn1_road', msk)
    img = read_tif(id)
    # plt.figure(figsize=(25, 25))
    plt.figure()
    ax1 = plt.subplot(131)
    ax1.set_title('image ID:hn1')
    ax1.imshow(img[:, :, 1], cmap=plt.get_cmap('gist_ncar'))
    ax2 = plt.subplot(132)
    ax2.set_title('predict bldg pixels')
    ax2.imshow(msk[:, :, 5], cmap=plt.get_cmap('gray'))
    ax3 = plt.subplot(133)
    ax3.set_title('predict bldg polygones')
    ax3.imshow(mask_for_polygons(mask_to_polygons(msk[:, :, 5], epsilon=1), img.shape[:2]), cmap=plt.get_cmap('gray'))

    plt.show()
    plt.savefig(FLAGS.model_dir + "hn1-road.png")


if __name__ == '__main__':
    model = train_net()
    score, trs = calc_jacc(model)
    # predict_test(model, trs)
    check_predict()
    print(get_fcnrnn().summary())