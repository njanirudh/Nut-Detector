import argparse
import cv2
import csv
import math
import os
from src.cv_object_detector import CVTFObjectDetector

if __name__ == "__main__" :

    IMAGE = "/home/nj/HBRS/RnD/Datasets/lucy_dataset_mixed/images/handle_e9b4a39a-2afe-47b4-95fd-50096ea8f0c2.jpg"

    FROZEN_GRAPH = "/home/nj/Desktop/RnD/Datasets/lucy_dataset_mixed/Trained/FRCNN_MIXED/freeze/frozen_inference_graph.pb"
    PBTEXT = "/home/nj/Desktop/RnD/Datasets/lucy_dataset_mixed/Trained/FRCNN_MIXED/frcnn_mixed_handle_opencv_pipeline.config"
    VIDEO = "/home/nj/Desktop/RnD/Datasets/handles.avi"
    LABELS = {0: "pull_knob",
              1: "lever_knob",
              2: "round_knob"
              }

    img = cv2.imread(IMAGE)

    obj_detector = CVTFObjectDetector()
    obj_detector.set_parameters(FROZEN_GRAPH,PBTEXT)
    obj_detector.set_input_image(img,(640,480))
    obj_detector.run_detection()
    obj_detector.set_labels(LABELS)
    cv2.imshow('img', obj_detector.get_inference_image())
    cv2.waitKey(10000)

    # vidcap = cv2.VideoCapture(VIDEO)
    # success, image = vidcap.read()
    # count = 0
    #
    # obj_detector = CVTFObjectDetector()
    # obj_detector.set_parameters(FROZEN_GRAPH,PBTEXT)
    #
    # while success:
    #     success, image = vidcap.read()
    #
    #     obj_detector.set_input_image(image,(640,480))
    #     obj_detector.run_detection()
    #     obj_detector.set_labels(LABELS)
    #     # print('Read a new frame: ', success,count)
    #     count += 1
    #
    #     cv2.imshow('img', obj_detector.get_inference_image())
    #     # if count % 3 == 0:
    #     #     cv2.imwrite(os.path.join("/home/nj/Desktop/RnD/video",str(count)+".jpg"),image)
    #
    #     cv2.waitKey(10)
    #
    #     obj_detector.reset()
    #
    # cv2.destroyAllWindows()
