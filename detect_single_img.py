import cv2
import math
import numpy as np
import sys
img = cv2.imread('./pic/17.jpg')
img_h,img_w,_ = img.shape
r = 4
img = cv2.resize(img,(img_w//r,img_h//r))
row = 6
column = 8
# 检测角点
# 7*10,ret为bool变量，corners为角点矩阵(row*column个角点，包括x,y坐标)
ret, corners = cv2.findChessboardCorners(img, (row, column), None)
print(ret,corners.shape) # True , (40,1,2)
if not ret:
    print('没有检测到')
else:

    point1 = corners[0,0,:]
    point2 = corners[(row - 1),0,:]
    point3 = corners[-1,0,:]
    point4 = corners[-row,0,:]

    delta_1 = (point2 - point1) / (row - 1)
    delta_2 = (point4 - point1) / (column - 1)

    real_point1 = point1 - delta_1 - delta_2
    real_point2 = point2 + delta_1 - delta_2
    real_point3 = point3 + delta_1 + delta_2
    real_point4 = point4 - delta_1 + delta_2

    real_point1 = np.array(real_point1,dtype=int)
    real_point2 = np.array(real_point2,dtype=int)
    real_point3 = np.array(real_point3,dtype=int)
    real_point4 = np.array(real_point4,dtype=int)
    # 将四个真实边缘点绘制出来，矩形绘制
    cv2.circle(img, ((real_point1[0]),(real_point1[1])), 10, (255, 0, 0), 2)
    cv2.circle(img, ((real_point2[0]),(real_point2[1])), 10, (255, 0, 0), 2)
    cv2.circle(img, ((real_point3[0]),(real_point3[1])), 10, (255, 0, 0), 2)
    cv2.circle(img, ((real_point4[0]),(real_point4[1])), 10, (255, 0, 0), 2)

    # cv2.rectangle(img, (real_point1[0],real_point1[1]) , (real_point3[0],real_point3[1]), (0,255,0), 2)
    # 绘制矩形
    point_color = (0, 255, 0) # BGR
    thickness = 1
    lineType = 4
    cv2.line(img, (real_point1[0],real_point1[1]), (real_point2[0],real_point2[1]), point_color, thickness, lineType)
    cv2.line(img, (real_point2[0],real_point2[1]), (real_point3[0],real_point3[1]), point_color, thickness, lineType)
    cv2.line(img, (real_point3[0],real_point3[1]), (real_point4[0],real_point4[1]), point_color, thickness, lineType)
    cv2.line(img, (real_point4[0],real_point4[1]), (real_point1[0],real_point1[1]), point_color, thickness, lineType)


    cv2.drawChessboardCorners(img, (row, column ), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值

# 标定的参数


f = 3200.86
print(f)
W = 0.182 # 靶标真实宽度
# 根据标定结果，计算坐标
def cal_w(real_point1,real_point2):
    dis = real_point1 - real_point2
    w = math.sqrt(dis[0]*dis[0] + dis[1]*dis[1])
    return w
# 计算距离
def dis_cal(f,W,w):
    '''

    :param fc: 焦距
    :param W: 靶标高度
    :param w: 靶标相机成像高度
    :return: 距离
    '''
    return f * W / w /r

w = cal_w(real_point1,real_point2)
print(dis_cal(f,W,w))

cv2.imshow('img',img)
cv2.waitKey(0)
sys.exit()