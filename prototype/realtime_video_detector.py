import argparse
import cv2
import csv
import math
from cv_object_detector import CVTFObjectDetector

if __name__ == "__main__" :

    IMAGE = "/home/nj/Desktop/CV/Dataset/Images/CV19_image_20.jpeg"
    FROZEN_GRAPH = "/home/nj/Desktop/CV/Trained_Models/FRCNN_Tray.pb"
    PBTEXT = "/home/nj/Desktop/CV/Dataset/Trained/FRCNN_TRAY/opencv_frcnn_tray.pbtxt"
    VIDEO = "/home/nj/HBRS/Studies/Sem-3/CV/Dataset/Videos/CV19_video_79.avi"
    LABELS = {0: "Walnut",
              1: "Peanut",
              2: "Hazelnut",
              3: "Tray"}

    vidcap = cv2.VideoCapture(VIDEO)
    success, image = vidcap.read()
    count = 0

    obj_detector = CVTFObjectDetector()
    obj_detector.set_parameters(FROZEN_GRAPH,PBTEXT)

    while success:
        success, image = vidcap.read()

        obj_detector.set_input_image(image)
        obj_detector.run_detection()
        obj_detector.set_labels(LABELS)
        print('Read a new frame: ', success,count)
        count += 1

        cv2.imshow('img', obj_detector.get_inference_image())
        cv2.waitKey(20)

        obj_detector.reset()

    cv2.destroyAllWindows()