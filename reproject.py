import cv2
import numpy as np
# def reproject(x = 1684.,y = 1798. ):
    # 相机内参矩阵和畸变系数，需要从相机标定过程中获得
def reproject(x,y):
    camera_matrix = np.array([[2040, 0, 960],
     [0.00000000e+00, 2041, 540],
     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_coeffs = np.array([0, 0, 0, 0,0],dtype=np.float32)
    # 已知的Z常数（橙色球的高度）
    Z_const = 0
    # 图像中的点（2D）
    image_points = np.array([
        (969,1710),   #(0.,0.,0.),       
        (862,1333),   #(0.,800.,0.),
        (804,1139), #(0.,1600.,0.),
        (772,1020), #(0.,2400.,0.),
        (733,886), #(0.,3200.,0.),
        (196,1352),   #(800.,800.,0.),      14
        (303,1156),  #(800.,1600.,0.),      4
        (368,1036), #(800.,2400.,0.), 
        
                             
    ], dtype=np.float32)
    # 世界坐标中的点（3D），单位为毫米
    object_points = np.array([
        (0.,0.,0.),
        (0.,800. ,0.),
        (0.,1600.,0.),           #4
        (0.,2400.,0.),
        (0.,3200.,0.),
        (800.,800.,0.),           #14      
        (800.,1600.,0.),

        (800.,2400.,0.),          #4


    ], dtype=np.float32)
    # 求解PnP问题
    #rvec:旋转向量
    #tvec:平移向量
    '''
    object_points:物体在世界坐标系中的位置。
    image_points:表示物体在图像平面上的投影。
    camera_matrix:相机内参矩阵.
    dist_coeffs:相机畸变系数的数组
    '''
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    # 选择一个图像点进行计算
    uv_point = np.array([[x], [y], [1]], dtype=np.float32)
    # 计算s值
    left_side_mat = np.linalg.inv(rotation_matrix)@(np.linalg.inv(camera_matrix))@(uv_point)
    right_side_mat = np.linalg.inv(rotation_matrix)@(tvec)
    s = (Z_const + right_side_mat[2]) / left_side_mat[2]
    # 计算3D坐标
    point_3d = np.linalg.inv(rotation_matrix)@(s * np.linalg.inv(camera_matrix)@(uv_point) - tvec)
    return point_3d
