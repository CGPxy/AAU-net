# coding=utf-8
from tensorflow.keras.layers import *
import cv2
import tensorflow.keras.backend as K
from tensorflow.keras.models import *


def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat

# Channel attentation
def Channelblock(data, filte):
    conv1 = Conv2D(filte, (3, 3), padding="same",dilation_rate=(3,3))(data)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = ReLU()(batch1)

    conv2 = Conv2D(filte, (5, 5), padding="same")(data)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = ReLU()(batch2)

    data3 = concatenate([LeakyReLU1, LeakyReLU2])
    data3 = GlobalAveragePooling2D()(data3)
    data3 = Dense(units=filte)(data3)
    data3 = BatchNormalization()(data3)
    data3 = ReLU()(data3)
    data3 = Dense(units=filte)(data3)
    data3 = Activation('sigmoid')(data3)

    a = Reshape((1, 1, filte))(data3)

    a1 = 1-data3
    a1 = Reshape((1, 1, filte))(a1)

    y = multiply([LeakyReLU1, a])

    y1 = multiply([LeakyReLU2, a1])

    data_a_a1 = concatenate([y, y1])

    conv3 = Conv2D(filte, (1, 1), padding="same")(data_a_a1)
    batch3 = BatchNormalization()(conv3)
    LeakyReLU3 = ReLU()(batch3)
    return LeakyReLU3

# spatial attentation
def Spatialblock(data, channel_data, filte, size):
    conv1 = Conv2D(filte, (3, 3), padding="same")(data)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = ReLU()(batch1)

    conv2 = Conv2D(filte, (1, 1), padding="same")(LeakyReLU1)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = ReLU()(batch2)


    data3 = add([channel_data, spatil_data])
    data3 = ReLU()(data3)
    data3 = Conv2D(1, (1, 1), padding='same')(data3)
    data3 = Activation('sigmoid')(data3)

    a = expend_as(data3, filte)
    y = multiply([a, channel_data])

    a1 = 1-data3
    a1 = expend_as(a1, filte)
    y1 = multiply([a1, spatil_data])

    data_a_a1 = concatenate([y, y1])

    conv3 = Conv2D(filte, size, padding='same')(data_a_a1)
    batch3 = BatchNormalization()(conv3)

    return batch3

def HAAM(data, filte,size):

    channel_data = Channelblock(data=data, filte=filte)

    haam_data = Spatialblock(data, channel_data, filte, size)

    return haam_data
