# coding=utf-8
import matplotlib as mpl
import cv2
import random
import os
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
mpl.use('TkAgg')

img_w = 384
img_h = 384

image_sets = []

def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        # image
        image_sets.append(filename)

def elastic_transform(image, label, alpha=10, sigma=2, alpha_affine=2, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1: 仿射变换前的点(3个点)
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])
    # pts2: 仿射变换后的点
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    # 仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    # 对image进行仿射变换.
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    imageA = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # generate random displacement fields
    # random_state.rand(*shape)会产生一个和shape一样打的服从[0,1]均匀分布的矩阵
    # *2-1是为了将分布平移到[-1, 1]的区间, alpha是控制变形强度的变形因子
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # generate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # x+dx,y+dy
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # bilinear interpolation
    xb = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)
    yb = map_coordinates(imageA, indices, order=1, mode='constant').reshape(shape)
    return xb,yb
    
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img

def add_noise(xb):
    sigma = random.uniform(5, 10)
    h,w = xb.shape
    gauss = np.random.normal(0, sigma, (h, w))
    gauss = gauss.reshape(h, w)
    noisy = xb + gauss
    noisy = np.clip(noisy, 0, 255).astype('uint8')
    return noisy

def data_augment(xb, yb):
    if np.random.random() < 0.20:
        xb, yb = rotate(xb, yb, random.uniform(0, 1)*20)
    # if np.random.random() < 0.20:
    #     xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.20:
        xb, yb = rotate(xb, yb, random.uniform(1,1.05)*340)
    if np.random.random() < 0.30:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.30:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.30:
        xb = blur(xb)

    if np.random.random() < 0.25:
        xb = add_noise(xb)

    if np.random.random() < 0.35:
        xb,yb = elastic_transform(xb,yb)

    return xb, yb
