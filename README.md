# 1 预处理
## 1.1 抛物线均衡化
&emsp;纵向投影计算均值，得到直方图，通过最小二乘法拟合出抛物线；通过图像减去抛物线的方式完成亮度均衡化。

## 1.2 前处理
&emsp;CLAHE均衡化——提高对比度、滤波

# 2 ROI提取
&emsp;双阈值直线扫描裁剪方式：
&emsp;高斯模糊、自适应二值化、记录灰度值高的左右坐标，将这之间的全置255

# 3 检测算法
## 3.1 波形检测
&emsp;高斯模糊后，计算横向投影均值，采用均值和方差判断ROI是否有波形缺陷
## 3.2 褶皱检测
&emsp;ROI提取、CLAHE均衡化、平均滤波、Canny、断直线、连通域、面积判断缺陷
