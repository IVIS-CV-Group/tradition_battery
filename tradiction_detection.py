import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import signal

'''高斯差分'''
# 高斯核
def gaussConv(I, size, sigma):
    # 卷积核的高和宽
    H, W = size
    # 构造水平方向上非归一化的高斯卷积核
    xr, xc = np.mgrid[0:1, 0:W]
    xc -= np.int((W - 1) / 2)
    xk = np.exp(-np.power(xc, 2.0) / (2.0 * pow(sigma, 2)))
    # I 与 xk 卷积
    I_xk = signal.convolve2d(I, xk, 'same', 'symm')
    # 构造垂直方向上的非归一化的高斯卷积核
    yr, yc = np.mgrid[0:H, 0:1]
    yr -= np.int((H - 1) / 2)
    yk = np.exp(-np.power(yr, 2.0) / (2.0 * pow(sigma, 2.0)))
    # I_xk 与 yk 卷积
    I_xk_yk = signal.convolve2d(I_xk, yk, 'same', 'symm')
    I_xk_yk *= 1.0 / (2 * np.pi * pow(sigma, 2.0))
    return I_xk_yk
# 高斯差分
def DoG(I, size, sigma, k=1.1):
    # 标准差为 sigma 的非归一化的高斯卷积
    Is = gaussConv(I, size, sigma)
    # 标准差为 k*sigma 的非归一化高斯卷积
    Isk = gaussConv(I, size, k * sigma)
    # 两个高斯卷积的差分
    doG = Isk - Is
    doG /= (pow(sigma, 2.0) * (k - 1))
    return doG
def detect_DoG(image):
    # if len(sys.argv) > 1:
    #     image = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
    # else:
    #     print "Usge:python DoG.py imageFile"
    # image = cv.imread('1.jpg', cv.IMREAD_GRAYSCALE)

    # 显示原图
    cv.imshow("image", image)
    # 高斯差分边缘检测
    sigma = 2
    k = 1.1
    size = (25, 25)
    imageDoG = DoG(image, size, sigma, k)
    # 二值化边缘，对 imageDoG 阈值处理
    edge = np.copy(imageDoG)
    edge[edge > 0] = 255
    edge[edge <= 0] = 0
    edge = edge.astype(np.uint8)
    cv.imshow("edge", edge)
    # cv2.imwrite("edge.jpg", edge)
    asbstraction(imageDoG)
def asbstraction(imageDoG):
    # 图像边缘抽象化
    asbstraction = -np.copy(imageDoG)
    asbstraction = asbstraction.astype(np.float32)
    asbstraction[asbstraction >= 0] = 1.0
    asbstraction[asbstraction < 0] = 1.0 + np.tanh(asbstraction[asbstraction < 0])

    asbstraction = asbstraction * 255
    asbstraction = asbstraction.astype(np.uint8)
    # cv2.imwrite("asbstraction.jpg", asbstraction)
    cv.imshow("asbtraction", asbstraction)
    cv.waitKey(0)
    cv.destroyAllWindows()

'''黑帽顶帽'''
def yahen_detection(img_):
    """测黑条压痕"""
    img_crop = cv.GaussianBlur(img_, (7, 7), 2)
    img_tophat = cv.morphologyEx(img_crop, cv.MORPH_BLACKHAT, (3, 3))
    img_tophat = cv.dilate(img_tophat, (3, 3), iterations=1)
    _, img_tophat = cv.threshold(img_tophat, 1, 255, cv.THRESH_BINARY)
    img_hstack = np.hstack([img_,img_tophat])
    # img_plus=img_+img_tophat
    # cv.imshow('img_tophat', img_hstack)
    # cv.waitKey(0)
    return img_hstack,img_tophat

'''边缘检测'''
class EdgeDetect:
    def __init__(self,img):
        self.img=img
        # self.img= cv.GaussianBlur(img, (3, 3), -1)
    def prewitt(self):
        # Prewitt 算子
        kernelX = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=int)
        kernelY = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=int)
        # 对图像滤波
        x = cv.filter2D(self.img, cv.CV_16S, kernelX)
        y = cv.filter2D(self.img, cv.CV_16S, kernelY)
        # 转 uint8 ,图像融合
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        return cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    def sobel(self):
        # Sobel 算子
        kernelX = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=int)
        kernelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=int)
        # 对图像滤波
        x = cv.filter2D(self.img, cv.CV_16S, kernelX)
        y = cv.filter2D(self.img, cv.CV_16S, kernelY)
        # 转 uint8 ,图像融合
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        return cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    # Laplace 算子
    def laplace(self):
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=int)
        img = cv.filter2D(self.img, cv.CV_16S, kernel)
        return cv.convertScaleAbs(img)
    # LoG算子
    def LoG(self):
        kernel = np.array([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]],
                          dtype=int)
        img = cv.filter2D(self.img, cv.CV_16S, kernel)
        return cv.convertScaleAbs(img)
