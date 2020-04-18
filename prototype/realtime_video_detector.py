import argparse
import cv2
import csv
import math
from src.cv_object_detector import CVTFObjectDetector

if __name__ == "__main__" :

    FROZEN_GRAPH = "../model/FRCNN_Tray.pb"
    PBTEXT = "../model/opencv_frcnn_tray.pbtxt"
    VIDEO = "../test_data/test_video.avi"
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

        obj_detector.set_input_image(image,(640,480))
        obj_detector.run_detection()
        obj_detector.set_labels(LABELS)
        print('Read a new frame: ', success,count)
        count += 1

        cv2.imshow('img', obj_detector.get_inference_image())
        cv2.waitKey(20)

        obj_detector.reset()

    cv2.destroyAllWindows()