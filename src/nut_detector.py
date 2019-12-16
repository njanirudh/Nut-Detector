import cv2
import csv
import numpy as np
import math
from cv_object_detector import CVTFObjectDetector

class NutDetector:

    def __init__(self,video_path,result_path):
        self.video_path = video_path
        self.result_path = result_path

        self.stable_frame = None

        self.FROZEN_GRAPH = "/home/nj/Desktop/CV/Trained_Models/FRCNN_Tray.pb"
        self.PBTEXT = "/home/nj/Desktop/CV/Dataset/Trained/FRCNN_TRAY/opencv_frcnn_tray.pbtxt"

        self.obj_detector = CVTFObjectDetector()

        self.LABELS = {0: "Walnut",
                       1: "Peanut",
                       2: "Hazelnut",
                       3: "Tray"}

    def extract_most_stable_frame(self):
        vidcap = cv2.VideoCapture(self.video_path)

        count = 0
        old_frame = None
        full_frame_stable = None

        STATIONARY_FLAG = False

        while True:
            success, current_frame = vidcap.read()
            if success == True:
                count += 1

                full_frame_stable = current_frame
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                current_frame = cv2.resize(current_frame, (640, 480))
                current_frame = cv2.blur(current_frame, (5, 5))

                if old_frame is not None:
                    print("Frame No:", count)

                    diff_image = cv2.absdiff(old_frame, current_frame)
                    # print(np.sum(diff_image == 0) / (640 * 480))
                    unique = len(np.unique(diff_image))
                    print("//", unique)

                    if unique > 150:
                        STATIONARY_FLAG = True

                    if STATIONARY_FLAG:
                        if unique < 15:
                            print("--->", count)
                            self.stable_frame = full_frame_stable
                            break

                old_frame = current_frame

    def run_detection(self):
        self.obj_detector.set_parameters(self.FROZEN_GRAPH, self.PBTEXT)
        self.obj_detector.set_labels(self.LABELS)
        self.obj_detector.set_input_image(self.stable_frame)
        self.obj_detector.run_detection()

    def get_results(self):
        cv2.imwrite("/home/nj/Desktop/result.jpg",self.obj_detector.get_inference_image())
        result_dict = self.obj_detector.get_results()
        with open(self.result_path, 'w') as f:
            writer = csv.writer(f,delimiter=',')
            all_results_array = []
            for arr in result_dict:
                result_arr = []
                result_arr.append(15)
                x_mid,y_mid = self.__get_centre(arr["bbox"][0],arr["bbox"][1],
                                                arr["bbox"][2],arr["bbox"][3])
                result_arr.append(x_mid)
                result_arr.append(y_mid)
                result_arr.append(arr["label"])

                all_results_array.append(result_arr)

            writer.writerows(all_results_array)
                # print(result_arr)

    def __get_centre(self,left,top,right,bottom):
        return (math.ceil((left + right)/2), math.ceil((top + bottom)/2))


if __name__=="__main__":
    VIDEO = "/home/nj/HBRS/Studies/Sem-3/CV/Dataset/Videos/CV19_video_219.avi"
    Result_Path = "/home/nj/Desktop/result.csv"

    nut_detector = NutDetector(VIDEO,Result_Path)
    nut_detector.extract_most_stable_frame()
    nut_detector.run_detection()
    nut_detector.get_results()
