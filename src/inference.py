import cv2
import csv
import math
from cv_object_detector import CVTFObjectDetector

class NutDetector:

    def __init__(self,video_path,result_path):
        self.video_path = video_path
        self.result_path = result_path

        self.FROZEN_GRAPH = "/home/nj/Desktop/CV/Trained_Models/FRCNN_tray.pb"
        self.PBTEXT = "/home/nj/Desktop/CV/Dataset/frcnn_opencv.pbtext"

        self.obj_detector=None
        self.LABELS = {0: "Walnut",
                       1: "Peanut",
                       2: "Hazelnut",
                       4: "Tray"}

    def extract_stable_frame(self):
        pass

    def run_detector(self):
        vidcap = cv2.VideoCapture(self.video_path)
        success, image = vidcap.read()
        count = 0

        self.obj_detector = CVTFObjectDetector()
        self.obj_detector.set_parameters(self.FROZEN_GRAPH, self.PBTEXT)

        while success:
            success, image = vidcap.read()

            try:
                self.obj_detector.set_input_image(image)
                self.obj_detector.run_detection()
                self.obj_detector.set_labels(self.LABELS)
                print('Read a new frame: ', success, count)
                count += 1

                cv2.imshow('img', self.obj_detector.get_inference_image())
                cv2.waitKey(20)

                self.obj_detector.reset()

            except:
                print("[ERROR] Problem in running the detector")

        cv2.destroyAllWindows()

    def __get_centre(self,left,top,right,bottom):
        return (math.ceil((left + right)/2), math.ceil((top + bottom)/2))

    def __save_result(self):
        print(self.obj_detector.get_results())
        result_dict = self.obj_detector.get_results()
        with open(self.result_path, 'wb') as f:
            writer = csv.writer(f)
            for arr in result_dict:
                result_arr = []
                result_arr.append(15)
                x_mid,y_mid = self.__get_centre(arr["bbox"][0],arr["bbox"][1],
                                                arr["bbox"][2],arr["bbox"][3])
                result_arr.append(x_mid)
                result_arr.append(y_mid)
                result_arr.append(arr["label"])

                writer.writerows(result_dict)


if __name__=="__main__":
    nut_detector = NutDetector()

