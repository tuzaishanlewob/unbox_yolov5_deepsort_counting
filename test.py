import numpy as np
from reproject import reproject
import csv
csv_file_path = './ppl/ppl_3d_test.csv' 

img_points = np.array([
        (969,1710),   #(0.,0.,0.),       
        (862,1333),   #(0.,800.,0.),
        (804,1139), #(0.,1600.,0.),
        (772,1020), #(0.,2400.,0.),
        (733,886), #(0.,3200.,0.),
        (196,1352),   #(800.,800.,0.),      14
        (303,1156),  #(800.,1600.,0.),      4
        (368,1036), #(800.,2400.,0.)
], dtype=np.float32)
obj_points = np.zeros((14,1,3), dtype=np.float32)
i = 1
obj_points = np.zeros((1,3), dtype=np.float32)
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(['Test ID', '3D_X', '3D_Y'])
    for point in img_points:
        x,y = point
        print('第'+str(i)+'帧')
        obj_point = reproject(x,y)
        print(obj_point)
        # writer.writerow([i, obj_point[0], obj_point[1]])
        i+=1
    
