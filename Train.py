# coding=utf-8
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from PIL import Image
import cv2
import random
import os
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
import xlsxwriter
import xlrd
from model_net import *

matplotlib.use("Agg")
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"



def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
    return img


filepath ='/media/dy/Data_2T/CGP/Unet_Segnet/data/new-kidney/3/Train_images/Augementa/'

def get_train_data():
    train_url = []
    train_set = []
    for pic in os.listdir(filepath + 'Train_images/'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    print(total_num)
    for i in range(len(train_url)):
        train_set.append(train_url[i])

    return train_set


def get_train_val(val_rate=0.25):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(filepath + 'Train_images/'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


# data for training
def generateData(batch_size, data=[]):
    # print 'generateData...'
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'Train_images/' + url)
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(filepath + 'Train_labels/' + url, grayscale=True)
            label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0


# data for validation
def generateValidData(batch_size, data=[]):
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'Train_images/' + url)
            img = img_to_array(img)
            valid_data.append(img)
            label = load_img(filepath + 'Train_labels/' + url, grayscale=True)
            label = img_to_array(label)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0  


def train(args):
    EPOCHS = 50
    BS = 12

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # Open a strategy scope.
    with strategy.scope():
        model = AAUnet()
        model.summary()

    model.compile(loss=BCE(),optimizer='adam',metrics=['accuracy'])

    checkpointer = ModelCheckpoint(os.path.join(
        args['save_dir'], 'model_{epoch:03d}.hdf5'), monitor='val_acc', save_best_only=False, mode='max')

    tensorboard = TensorBoard(log_dir='./logs/kidney/3/', histogram_freq=0, write_graph=True, write_images=True)

    train_set,val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,  
                    validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS,callbacks=[checkpointer, tensorboard])   #,max_q_size=1

    # k_fold = 3
    # Train_data = get_train_data()
    # num_val_samples = len(Train_data)//k_fold
    # all_scores = []

    # for  i in range(k_fold):
    #     val_set = Train_data[i*num_val_samples:(i+1)*num_val_samples]
    #     train_set = np.concatenate([Train_data[:i*num_val_samples],Train_data[(i+1)*num_val_samples:]])

    #     # train_set, val_set = get_train_val()
    #     train_numb = len(train_set)
    #     valid_numb = len(val_set)
    #     print("the number of train data is", train_numb)
    #     print("the number of val data is", valid_numb)
    #     H = model.fit_generator(generator=generateData(BS, train_set), steps_per_epoch=train_numb // BS, epochs=EPOCHS, verbose=1,
    #                             validation_data=generateValidData(BS, val_set), validation_steps=valid_numb // BS, callbacks=[checkpointer, tensorboard], max_q_size=1)
    # history_dict = H.history_dict
    # print(history_dict.keys())

    # mae = history_dict['val_loss']
    # all_scores.append(mae)
    # print(all_scores)



def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--augment", help="using data augment or not",
                    action="store_true", default=False)
    ap.add_argument("-m", "--save_dir", default="/media/dy/Data_2T/CGP/Unet_Segnet/AAUnet/model/kidney/3/",
                    help="path to output model")
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    args = args_parse()
    if args['augment'] == True:
        filepath = '/media/dy/Data_2T/CGP/Unet_Segnet/data/'

    train(args)
    # predict()
