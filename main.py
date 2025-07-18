import numpy as np
import csv
import tracker
from detector import Detector
import cv2

frame_id = 0
csv_file_path = './ppl/ppl_3d_idle.csv'  # 定义CSV文件路径
pic_file_path = './ppl/pic/'
savedir = './ppl/results/'
if __name__ == '__main__':
    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture('./ppl/5.mp4')
    # capture = cv2.VideoCapture('/mnt/datasets/datasets/towncentre/TownCentreXVID.avi')
    
    # 打开CSV文件准备写入
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['Frame ID', '3D_X', '3D_Y', 'Class ID', 'Position ID'])
        
        while True:#frame_id < 3:
            # 读取每帧图片
            _, im = capture.read()
            # im = cv2.imread(pic_file_path+str(frame_id)+'.bmp')
            if im is None:
                break
            
            # 缩小尺寸，1920x1080->960x540
            # im = cv2.resize(im, (960, 540))

            list_bboxs = []
            bboxes = detector.detect(im)

            # 如果画面中有bbox
            if len(bboxes) > 0:
                # 处理视频
                list_bboxs = tracker.update(bboxes, im)
                # 画框，并获取更新后的bboxes
                output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None, csv_writer=writer, frame_id=frame_id)
            else:
                # 如果画面中没有bbox
                output_image_frame = im
            frame_id += 1
            # cv2.imshow(str(frame_id), output_image_frame)
            cv2.imwrite(savedir+str(frame_id)+'.png', output_image_frame)
            # cv2.waitKey(1)

        pass
    pass

    capture.release()
    cv2.destroyAllWindows()
