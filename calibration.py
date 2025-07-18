#库
import numpy as np
import cv2
import glob


# 设置工作目录和保存目录

savedir = "./para/"

# 定义迭代算法的终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备对象点，rowxcolumn棋盘格，每个格子len厘米
row = 6
column = 8
len = 2.6
objp = np.zeros((row*column,3), np.float32)
objp[:,:2] = np.mgrid[0:row,0:column].T.reshape(-1,2)*len
'''
.T 是对网格进行转置，使得坐标对按照列排列，即先变化列坐标，再变化行坐标。
.reshape(-1,2) 将转置后的网格重塑为一个1维数组,其中每行包含两个元素,分别表示x和y坐标。
*len 将每个坐标值乘以len,这是因为棋盘格的每个格子的实际大小是len厘米。
'''
# 存储对象点和图像点的列表
objpoints = []
imgpoints = []

# 获取所有标定图像的路径
images = glob.glob("./pic/*.bmp")



# 读取并处理图像，检测棋盘格角点
print("getting images")
for fname in images:
    img = cv2.imread(fname)
    print(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#将读取的彩色图像转换为灰度图像。

    ret, corners = cv2.findChessboardCorners(gray, (row,column), None)
    print(ret,corners.shape)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, (row,column), corners2, ret)
        print(fname)




# 开始相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#shape[::-1]：这表示将图像的尺寸（宽和高）作为输入参数传递给calibrateCamera函数。gray是灰度图像，shape属性返回一个包含图像维度的元组，[::-1]用于反转这个元组，即先高度后宽度。
# 打印并保存相机内参矩阵
print("Camera Matrix")
print(camera_matrix)
np.save(savedir+'cam_mtx.npy', camera_matrix)

# 打印并保存畸变系数
print("Distortion Coeff")
print(dist_coeffs)
np.save(savedir+'dist.npy', dist_coeffs)

# 打印旋转向量和平移向量
print("r vecs")
#会输出第三张标定图像对应的旋转向量。
print(rvecs[2])
print("t Vecs")
#输出第三张标定图像对应的平移向量
print(tvecs[2])

print(">==> Calibration ended")

# 获取图像尺寸
h, w = img.shape[:2]
print("Image Width, Height")
print(w, h)

# 获取新的相机矩阵和ROI
newcam_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))

# 打印并保存ROI和新的相机矩阵
print("Region of Interest")
print(roi)
# np.save(savedir+'roi.npy', roi)
print("New Camera Matrix")
np.save(savedir+'newcam_mtx.npy', newcam_mtx)
print(np.load(savedir+'newcam_mtx.npy'))

# 计算新相机矩阵的逆矩阵
inverse = np.linalg.inv(newcam_mtx)
print("Inverse New Camera Matrix")
print(inverse)

# 对图像进行畸变矫正
undst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcam_mtx)
u_h,u_w = undst.shape[:2]
# 显示原始和矫正后的图像
r = 3
undst = cv2.resize(undst,(u_w // r, u_h // r))
img = cv2.resize(img, ((w // r), (h // r)))
cv2.imshow('img1', img)
cv2.waitKey(5000)      
cv2.destroyAllWindows()
cv2.imshow('undst', undst)
cv2.waitKey(5000)      
cv2.destroyAllWindows()

