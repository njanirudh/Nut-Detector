import cv2
import csv
import math
from cv_object_detector import CVTFObjectDetector

class NutDetector:

    def __init__(self,video,result):
        self.video_path = video
        self.result_path = result

        self.FROZEN_GRAPH = "/home/nj/Desktop/CV/Trained_Models/FRCNN_tray.pb"
        self.PBTEXT = "/home/nj/Desktop/CV/Dataset/frcnn_opencv.pbtext"

        self.obj_detector=None
        self.LABELS = {0: "Walnut",
                       1: "Peanut",
                       2: "Hazelnut",
                       4: "Tray"}

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

    IMAGE = "/home/nj/Desktop/CV/Dataset/Images/CV19_image_210.jpeg"
    FROZEN_GRAPH = "/home/nj/Desktop/CV/Trained_Models/FRCNN_Tray.pb"
    PBTEXT = "/home/nj/Desktop/CV/Dataset/Trained/FRCNN_TRAY/opencv_frcnn_tray.pbtxt"
    VIDEO = "/home/nj/HBRS/Studies/Sem-3/CV/Dataset/Videos/CV19_video_269.avi"
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

        # print(obj_detector.get_results())
        # with open("ress.csv", 'wb') as f:
        #     writer = csv.writer(f)
        #     # writer.writerows(obj_detector.get_results())
        #     for arr in obj_detector.get_results():
        #         result_arr = []
        #         result_arr.append(15)
        #         x_mid, y_mid = self.__get_centre(arr["bbox"][0], arr["bbox"][1],
        #                                          arr["bbox"][2], arr["bbox"][3])
        #         result_arr.append(x_mid)
        #         result_arr.append(y_mid)
        #         result_arr.append(arr["label"])
        #
        #         writer.writerows(result_arr)


        # except:
        #     print("Done!!")

    cv2.destroyAllWindows()
    # inference_on_image(FROZEN_GRAPH,PBTEXT,IMAGE)
    # print(cv_object_detector.CVTFObjectDetector)