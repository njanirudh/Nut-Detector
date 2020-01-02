import argparse
import cv2
import csv
import math
import os
from src.cv_object_detector import CVTFObjectDetector

if __name__ == "__main__" :

    IMAGE = "/home/nj/Desktop/CV/Dataset/Images/CV19_image_20.jpeg"
    FROZEN_GRAPH = "/home/nj/Desktop/RnD/Datasets/lucy_dataset_mixed/Trained/FRCNN_MIXED/freeze/frozen_inference_graph.pb"
    PBTEXT = "/home/nj/Desktop/RnD/Datasets/lucy_dataset_mixed/Trained/FRCNN_MIXED/frcnn_mixed_handle_opencv_pipeline.config"
    VIDEO = "/home/nj/Desktop/RnD/Datasets/handles.avi"
    LABELS = {0: "pull_knob",
              1: "lever_knob",
              2: "round_knob"
              }

    vidcap = cv2.VideoCapture(VIDEO)
    success, image = vidcap.read()
    count = 0

    obj_detector = CVTFObjectDetector()
    obj_detector.set_parameters(FROZEN_GRAPH,PBTEXT)

    while success:
        success, image = vidcap.read()

        obj_detector.set_input_image(image)
        # obj_detector.run_detection()
        # obj_detector.set_labels(LABELS)
        # print('Read a new frame: ', success,count)
        count += 1

        # cv2.imshow('img', obj_detector.get_inference_image())
        if count % 3 == 0:
            cv2.imwrite(os.path.join("/home/nj/Desktop/RnD/video",str(count)+".jpg"),image)

        cv2.waitKey(10)

        obj_detector.reset()

    cv2.destroyAllWindows()
