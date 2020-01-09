import cv2
import numpy as np
import os

if __name__ == "__main__":
    VIDEO = "/home/nj/HBRS/Studies/Sem-3/CV/Dataset/Videos/CV19_video_79.avi"

    IMAGE_1 = cv2.imread("/home/nj/Desktop/temp/24.jpg",0)
    IMAGE_2 = cv2.imread("/home/nj/Desktop/temp/27.jpg",0)

    diff_image = cv2.absdiff(IMAGE_1, IMAGE_2)
    cv2.imwrite("/home/nj/Desktop/temp/result_same.jpg",diff_image)
    # cap = cv2.VideoCapture(VIDEO)
    #
    # count = 0
    # while True:
    #
    #     count += 1
    #     ret, frame = cap.read()
    #     if ret == True:
    #         print(count)
    #         if count % 3 == 0 :
    #             cv2.imwrite(os.path.join("/home/nj/Desktop/temp",str(count)+".jpg"),frame)