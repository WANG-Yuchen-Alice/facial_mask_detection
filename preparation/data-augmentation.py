'''
This is the image data enhancement code, can be realized on the picture:
1. Zoom in and out
2. Random tailoring
Step 3: Transform
4. Rotation (any Angle, such as 45°, 90°, 180°, 270°)
5. Flip (horizontal flip, vertical flip)
6. Brightness changes (lightening, darkening)
7. Pixel translation (Translate the pixel in one direction, and the empty part will automatically fill the black)
8. Add noise (salt and pepper noise, Gaussian noise)
'''
import os
import cv2
import numpy as np
# import tensorflow as tf
import random as rd
import matplotlib
from tqdm import tqdm


matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

'''
缩放
'''


# Scale up and down
def Scale(image, scale):
    return cv2.resize(image, (500, 500), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


'''
Crop
'''


def crop(image, min_ratio=0.6, max_ratio=1.0):
    h, w = image.shape[:2]
    ratio = rd.random()
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    new_h = int(h * scale)
    new_w = int(w * scale)
    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)
    image = image[y:y + new_h, x:x + new_w, :]
    return image


def change(image):
    x, y = image.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(image, M, (y, x), borderValue=(255, 255, 255))
    return dst


'''
Rotate
'''


# Flip horizontally
def Horizontal(image):
    return cv2.flip(image, 1, dst=None)


# Flip vertically
def Vertical(image):
    return cv2.flip(image, 0, dst=None)


# Rotate
def Rotate(image, angle=15, scale=0.9):
    w = image.shape[1]
    h = image.shape[0]
    # rotate matrix
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    # rotate
    image = cv2.warpAffine(image, M, (w, h))
    return image


'''  
Brightness
'''


# Darkness
def Darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


# Brighter
def Brighter(image, percetage=1.1):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


# Move
def Move(img, x, y):
    img_info = img.shape
    height = img_info[0]
    width = img_info[1]

    mat_translation = np.float32([[1, 0, x], [0, 1, y]])  # 变换矩阵：设置平移变换所需的计算矩阵：2行3列
    # [[1,0,20],[0,1,50]]   表示平移变换：其中x表示水平方向上的平移距离，y表示竖直方向上的平移距离。
    dst = cv2.warpAffine(img, mat_translation, (width, height))  # 变换函数
    return dst


'''
Add noise
'''


# Salt/Pepper noise
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


# Gaussian Noise
def GaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


def Blur(img):
    blur = cv2.GaussianBlur(img, (7, 7), 1.5)
    return blur


# Multi photo augmentation
def AllData(rootpath, save_loc):
    # root_path = "data/"
    # save_loc = root_path
    for a, b, c in os.walk(root_path):
        for file_i in tqdm(c):
            file_i_path = os.path.join(a, file_i)
            # print(file_i_path)
            if '.DS_Store' in file_i_path:
                continue
            split = os.path.split(file_i_path)
            # print('split',split)
            dir_loc = os.path.split(split[0])[1]
            # print('dir_loc',dir_loc)
            save_path = os.path.join(save_loc, dir_loc)
            if not os.path.exists(save_path):  # If no such dir, create one
                os.mkdir(save_path)
            # print('save_path',save_path)

            img_i = cv2.imread(file_i_path)

            img_scale = Scale(img_i,1.5)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_scale.jpg"), img_scale)

            img_crop = crop(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_crop.jpg"), img_crop)

            img_change = change(img_i)
            cv2.imwrite(os.path.join(save_path,file_i[:-4] + "_change.jpg"),img_change)

            img_horizontal = Horizontal(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_horizontal.jpg"), img_horizontal)
            #
            img_vertical = Vertical(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_vertical.jpg"), img_vertical)
            # #
            img_rotate = Rotate(img_i, 90)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_rotate90.jpg"), img_rotate)
            # #
            img_rotate = Rotate(img_i, 180)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_rotate180.jpg"), img_rotate)
            # #
            img_rotate = Rotate(img_i, 270)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_rotate270.jpg"), img_rotate)
            # #
            img_move = Move(img_i, 15, 15)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_move.jpg"), img_move)
            # #
            img_darker = Darker(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_darker.jpg"), img_darker)
            # #
            img_brighter = Brighter(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_brighter.jpg"), img_brighter)
            # #
            img_blur = Blur(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_blur.jpg"), img_blur)
            # #
            img_salt = SaltAndPepper(img_i, 0.05)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_salt.jpg"), img_salt)
            #
            img_salt = GaussianNoise(img_i, 0.1)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_GaussianNoise.jpg"), img_salt)


if __name__ == "__main__":
    root_path = r"C:\temp_can\delete2\better_face_new_data\new_aug_data\irrelevant-small"
    save_loc = r'C:\temp_can\delete2\better_face_new_data\new_aug_data\small-irr-aug'
    AllData(root_path, save_loc)
