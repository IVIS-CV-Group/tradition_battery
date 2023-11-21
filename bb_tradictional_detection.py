import cv2 as cv
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import shutil
from PIL import Image
from scipy import signal
import math

# 全局变量，统计缺陷数目
global_flag = 0
global_num_boxing = 0
global_num_zhezhou = 0
global_num_yahen = 0



class myequalize:
    """
    抛物线均衡化
    """
    def __init__(self, img):
        self.params = self.myequalize_hist(img)

    
    # 二次函数的标准形式
    def func(self, params, x):
        a, b, c = params
        x = np.array(x)
        return a * x * x + b * x + c


    # 误差函数，即拟合曲线所求的值与实际值的差
    def error(self, params, x, y):
        return func(params, x) - y


    # 对参数求解
    def slovePara(self, x, y):
        p0 = [1, 1, 1]
        para_ = leastsq(error, p0, args=(x, y))
        return para_


    def myequalize_hist(self, img_):
        """输入模板图得到抛物线参数"""
        xsum_ = np.sum(img_, axis=0)
        xsum_ = xsum_ / img_.shape[0]
        xsum_func_x = []
        xsum_func_y = []
        for i in range(50, 7800):
            if xsum_[i] < 90:
                xsum_func_x.append(i)
                xsum_func_y.append(xsum_[i])
                # print("(" + str(i) + "," + str(xsum_[i]) + ")")

        para_ = slovePara(xsum_func_x, xsum_func_y)
        a, b, c = para_[0]
        print("已完成抛物线拟合！")
        print("参数为:a=" + str(a) + "  b=" + str(b) + "  c=" + str(c))
        return para_


    def myequalize_prepross(self, img_):
        """输入图像和抛物线参数得到均衡化后的图像"""
        a, b, c = self.params[0]
        # 全图预处理
        for i in range(img_.shape[0]):
                img_[:, i] = img_[:, i] - (a*i*i+b*i+c-50)

        # for i in range(img_.shape[0]):
        #     for j in range(img_.shape[1]):
        #         if img_[i][j] > a*j*j+b*j+c-30:
        #             img_[i][j] = img_[i][j] - (a*j*j+b*j+c-30)
        #         else:
        #             img_[i][j] = 5
        #     print("已完成一行预处理！")
        print("  已完成当前图像灰度均衡化！")
        return img_



def mylinecrop_img(img_original):
    """
    ROI提取:
    采用二值化线扫描的方式
    """
    img_original = cv.GaussianBlur(img_original, (3, 3), -1)    # 使用时无需事先滤波
    img_ada_thred = cv.adaptiveThreshold(img_original, 100, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, -3)
    # cv.imshow('img_ada_thred', img_ada_thred)

    # 进一步完整二值化白条
    contour_x_ = []
    for i in range(img_original.shape[0]):
        # x轴投影
        line_gray_value = np.sum(img_ada_thred[:, i], axis=0)
        if line_gray_value > 100*300:
            contour_x_.append(i)
            # print(contour_x)
            # 若当前检测出的白条边缘已是下一个的则先二值化上一个白条
            if max(contour_x_) - min(contour_x_) > 40:
                for j in range(contour_x_[0], contour_x_[-2]):
                    img_ada_thred[:, j] = 255
                max_num = contour_x_[-1]
                contour_x_ = [max_num]
        # 二值化最后一个白条
        if i == img_original.shape[0]-1:
            for j in range(contour_x_[0], contour_x_[-1]):
                img_ada_thred[:, j] = 255
            contour_x_ = []
    # 得到黑白
    _, img_mask_white_ = cv.threshold(img_ada_thred, 200, 255, cv.THRESH_BINARY)
    _, img_mask_black_ = cv.threshold(img_ada_thred, 200, 255, cv.THRESH_BINARY_INV)

    # 找出边缘坐标
    for i in range(img_mask_white_.shape[0]-1):
        if abs(int(img_mask_white_[40, i+1]) - int(img_mask_white_[40, i])) != 0:
            contour_x_.append(i)

    # cv.imshow('img_ada_thred_after', img_ada_thred)
    # cv.waitKey(0)
    return contour_x_, img_mask_black_, img_mask_white_



def boxing_detect(img_):
    """
    波形缺陷检测
    """
    img_ = cv.resize(img_, (0, 0), fx=0.1, fy=0.1, interpolation=cv.INTER_AREA)
    contour_x_, _, _ = mylinecrop_img(img_)
    if len(contour_x_) == 20:
        # 裁剪
        for i in range(9):
            # 黑条裁剪
            img_crop = img_[:, contour_x_[2*i+1]+5:contour_x_[2*i+2]-5]  # 缩小了裁剪的宽度
            img_crop = cv.GaussianBlur(img_crop, (3, 3), -1)

            # 计算横向投影
            y_sum = np.sum(img_crop, axis=1)

            # 计算方差
            y_col = y_sum / (contour_x_[2*i+2] - 10 - contour_x_[2*i+1])
            y_col_std = np.std(y_col)
            if y_col_std > 1.8:
                global global_flag
                global_flag = 8
                global global_num_boxing
                global_num_boxing = global_num_boxing + 1
                break



def zhezhou_detection(img_):
    """
    褶皱检测:
    测黑条褶皱
    """
    img_ = cv.resize(img_, (0, 0), fx=0.1, fy=0.1, interpolation=cv.INTER_AREA)

    contour_x_, _, _ = mylinecrop_img(img_)
    for i in range(9):
        # 黑条裁剪 + CLAHE均值化
        img_crop = img_[:, contour_x_[2*i+1]+5:contour_x_[2*i+2]-5]  # 缩小了裁剪的宽度
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_crop = clahe.apply(img_crop)

        # 滤波方法
        img_crop = cv.medianBlur(img_crop, 7)

        # 边缘检测
        edge_image = cv.Canny(img_crop, 12, 15)

        # 截断竖线
        for j in range(edge_image.shape[0] - 1):
            edge_image[j, :] = edge_image[j, :] - edge_image[j+1, :]

        # 连通域寻找
        connectivity = 8
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(edge_image, connectivity, cv.CV_16S)
        sizes = stats[1:, -1]  # 得到各连通域的面积
        nb_components = nb_components - 1  # 第一个连通域是背景

        # 找到output中值为label的区域赋值为255,min_size原始为120
        for j in range(0, nb_components):
            if sizes[j] >= 25:
                global global_flag
                global_flag = 6
                global global_num_zhezhou
                global_num_zhezhou = global_num_zhezhou + 1
                break



"""以下都是压痕检测"""
def gen_ROI(img_cv):
    # 自适应核大小：11
    img_ada_thred = cv.adaptiveThreshold(img_cv, 100, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, -3)
    xsum = np.sum(img_ada_thred, axis=0)
    xsum = xsum / int(img_cv.shape[1])
    for i in range(img_cv.shape[1]):  # 遍历宽度
        if xsum[i] > 40:
            img_ada_thred[:, i] = 255
    # 膨胀核大小：11
    img_ada_thred = cv.morphologyEx(img_ada_thred, cv.MORPH_CLOSE, kernel=np.ones((11, 11), np.uint8))
    # img_ada_thred = cv.morphologyEx(img_ada_thred, cv.MORPH_CLOSE, kernel=np.ones((50, 50), np.uint8)) #原图
    _, mask_w = cv.threshold(img_ada_thred, 200, 255, cv.THRESH_BINARY)
    _, mask_b = cv.threshold(img_ada_thred, 200, 255, cv.THRESH_BINARY_INV)
    # vis_cv(mask_b,'mask')
    return mask_w, mask_b


def gen_ROI_idx(img_mask): # m_b和m_w是一样的
    pla = []
    for i in range(img_mask.shape[1]-1):  # 遍历图像宽度
        # 前个数减后个数
        if int(img_mask[1, i])-int(img_mask[1, i+1]) != 0:  # 是边界
            pla.append(i)
    return pla


def crop(img, range = [400, 430]):
    img_crop = img[:, range[0]:range[1]]
    return img_crop


def mycrop_hstack(img_hstack_, img_crop_, i_):
    if i_ == 0:
        img_hstack_crop = img_crop_
    else:
        img_hstack_crop = np.hstack([img_hstack_, img_crop_])
    return img_hstack_crop


class Filter:  # 频域变换
    def __init__(self, img):
        self.img = img
        self.shape = img.shape
        self.f = np.fft.fft2(img)  # quick fft
        self.fshift = np.fft.fftshift(self.f)

    def PHOT(self):
        # get  magnitude
        M = np.abs(self.f)
        # Get phase
        P = self.f / M
        # Reconstruct image from phase
        R = np.abs(np.fft.ifft2(P))
        return R


def yahen_detection(img_):
    img_gray_small = cv.resize(img_, (0, 0), fx=0.1, fy=0.1, interpolation=cv.INTER_AREA)
    img_gray_small = cv.GaussianBlur(img_gray_small, (3, 3), -1)  # 高斯去噪
    '''gen_ROI'''
    m_w, m_b = gen_ROI(img_gray_small)
    edge = gen_ROI_idx(m_w)

    j = 0
    img_h = []
    for i in range(len(edge) - 1):
        if edge[i + 1] - edge[i] > 30:  # 边缘的两个边界
            pla = [edge[i] + 10, edge[i + 1] - 10]
            img_b = crop(img_gray_small, pla)
            # img_b = cv.GaussianBlur(img_w, (3, 3), -1)  # 高斯去噪
            img_h = mycrop_hstack(img_h, img_b, j)
            j += 1

    '''PHOT'''
    phot = Filter(img_h).PHOT()

    # 调大σ即提高了远处像素对中心像素的影响程度，滤波结果也就越平滑
    img_Ed = cv.GaussianBlur(phot, (3, 3), 3)  # 高斯去噪:参数3为x轴高斯核标准差，y轴默认和x轴相同
    # calculate mean and variance
    mean, stddv = cv.meanStdDev(img_Ed)
    # Compute distance Euclidean
    img_Ed = np.abs(img_Ed - mean)
    img_normal = np.zeros(img_Ed.shape)
    cv.normalize(img_Ed, img_normal, 0, 255, cv.NORM_MINMAX)
    img_normal = cv.convertScaleAbs(img_normal)  # 范围内,计算绝对值,并将结果转换为8位。//数据增强

    det, img_binary = cv.threshold(img_normal, 150, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    img_dilated = cv.dilate(img_binary, kernel)

    contours, hierarchy = cv.findContours(img_dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if contours:
        global global_flag
        global_flag = 5
        global global_num_yahen
        global_num_yahen = global_num_yahen + 1



if __name__ == '__main__':

    # 图片的主路径
    imgs_dir = os.path.join(os.curdir, 'battery/all defect(1)')
    template_dir = os.path.join(os.curdir, 'battery/template/template.jpg')
    boxing_dir = os.path.join(os.curdir, 'battery/boxing')
    zhezhou_new_dir = os.path.join(os.curdir, 'battery/zhezhou_new')
    zhezhou_more_dir = os.path.join(os.curdir, 'battery/zhezhou_more')
    yahen_dir = os.path.join(os.curdir, 'battery/yahen')


    for _, _, files in os.walk(imgs_dir):
        for file in files:
            print("当前图片：", file)
            img_dir = os.path.join(imgs_dir, file)

            # 读取模板图并抛物线均衡化
            myequalize_ = myequalize(cv.imread(template_dir))
            img_ = myequalize_.myequalize_prepross(cv.imread(img_dir))

            # 读取图像并灰度化
            image_bgr = cv.imread(img_dir)
            image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
            global_flag = 0

            # 波形缺陷分类
            boxing_detect(image_gray)
            if global_flag == 8:
                continue

            # 褶皱检测
            zhezhou_detection(image_gray)
            if global_flag == 6:
                to_path = os.path.join(zhezhou_new_dir, file)
                shutil.copy(img_dir, to_path)

            # 压痕检测
            # yahen_detection(image_gray)
            # if global_flag == 5:
            #     to_path = os.path.join(yahen_dir, file)
            #     shutil.copy(img_dir, to_path)

    # print("已成功转移褶皱缺陷图片", global_num_zhezhou, "张")
