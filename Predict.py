import cv2
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import argparse
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder  
import tensorflow as tf
from our_train import *
mpl.use('TkAgg')
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

TEST_SET = []

def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        TEST_SET.append(filename)


def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to trained model model")
    args = vars(ap.parse_args())    
    return args



def original_predict(args):
    # load the trained convolutional neural network
    # print("[INFO] loading network...")
    model = AAUnet()
    model.summary()

    model.load_weights("/media/dy/Data_2T/CGP/Unet_Segnet/AAUnet/model/kidney/2/" + args["model"])
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        image = cv2.imread('/media/dy/Data_2T/CGP/Unet_Segnet/data/new-kidney/2/Test_images/images/384/' + path)
        image = np.array(image,dtype=np.uint8)
        h,w,_ = image.shape
        padding_h = h
        padding_w = w
        padding_img = np.zeros((padding_h, padding_w, 3),dtype=np.uint8)
        padding_img[0:h,0:w,:] = image[:,:,:]
        padding_img = padding_img.astype("float") / 255.0
        padding_img = img_to_array(padding_img)
        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        crop = padding_img[:,:,:3]
        crop = np.expand_dims(crop, axis=0) 
        pred = model.predict(crop,verbose=2)
        preimage = pred.reshape((384,384))
        h,w = preimage.shape
        for i in range(0, h):
            for j in range(0, w):
                if (preimage[i, j] > 0.5):
                    preimage[i, j] = 1
                else:
                    preimage[i, j] = 0
        pred = preimage.reshape((384,384)).astype(np.uint8)
        mask_whole[:,:] = pred[:,:]
        cv2.imwrite('/media/dy/Data_2T/CGP/Unet_Segnet/AAUnet/result/mask/kidney/2/'+path,mask_whole[0:h,0:w])

if __name__ == '__main__':
    read_directory("/media/dy/Data_2T/CGP/Unet_Segnet/data/new-kidney/2/Test_images/images/384/")
    args = args_parse()
    original_predict(args)


