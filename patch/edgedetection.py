# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:20:57 2022

@author: eyxysdht
"""

import cv2


def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)   # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return absX, absY, dst
 
def scharr(img):
    x_Scharr = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=-1)
    y_Scharr = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=-1)
    # ksize=-1 Scharr算子
    # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
    # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
    Scharr_absX = cv2.convertScaleAbs(x_Scharr)  # convert 转换  scale 缩放
    Scharr_absY = cv2.convertScaleAbs(y_Scharr)
    dst = cv2.addWeighted(Scharr_absX, 0.5, Scharr_absY, 0.5, 0)
    return Scharr_absX, Scharr_absY, dst

def laplacian(img):
    #blur = cv2.GaussianBlur(img, (3, 3), 0)
    gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize = 3)
    dst = cv2.convertScaleAbs(gray_lap)
    return dst

def canny(img):
    img_gau = cv2.GaussianBlur(img, (3,3), 0) #用高斯平滑处理原图像降噪。
    dst = cv2.Canny(img_gau, 18, 6) #最大最小阈值, best setup for the assembly dataset.
    return dst