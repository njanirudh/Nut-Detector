import os
import cv2
import csv
import numpy as np
from cv_object_detector import CVTFObjectDetector
from utility import *

class NutDetector:

    def __init__(self,video_path,result_path,frozen_graph,pbtxt_path):
        self.video_path = video_path
        self.result_path = result_path

        self.stable_frame = None
        self.stable_frame_count = 0

        self.FROZEN_GRAPH = frozen_graph
        self.PBTEXT = pbtxt_path

        print(DEBUG(" Input video path : "+str(self.video_path)))
        print(DEBUG(" Result path : "+str(self.result_path)))
        print(DEBUG(" Frozen graph path : "+str(self.FROZEN_GRAPH)))
        print(DEBUG(" Pbtext path : "+str(self.PBTEXT)))

        self.obj_detector = CVTFObjectDetector()

        self.LABELS = {0: "Walnut",
                       1: "Peanut",
                       2: "Hazelnut",
                       3: "Tray"}

    def extract_most_stable_frame(self):
        """
        Extract the frame with no moving objects ie. Stable frame.
        """
        vidcap = cv2.VideoCapture(self.video_path)

        count = 0
        old_frame = None
        full_frame_stable = None

        STATIONARY_FLAG = False

        # Extract frames from video
        while True:
            success, current_frame = vidcap.read()
            if success == True:
                # Preprocess image before sending it for inference
                full_frame_stable = current_frame
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                current_frame = cv2.resize(current_frame, (640, 480))
                current_frame = cv2.blur(current_frame, (5, 5))

                if old_frame is not None:
                    diff_image = cv2.absdiff(old_frame, current_frame)
                    unique = len(np.unique(diff_image))
                    print("Frame No:", count," | ", unique)

                    if unique > 150:
                        STATIONARY_FLAG = True

                    if STATIONARY_FLAG:
                        if unique < 15:
                            print(DEBUG(" The stable frame number is : " + str(count)))
                            self.stable_frame_count = count
                            self.stable_frame = full_frame_stable
                            break

                old_frame = current_frame
            count += 1

    def run_detection(self):
        """
        Run detection with input, parameters.
        """
        self.obj_detector.set_parameters(self.FROZEN_GRAPH, self.PBTEXT)
        self.obj_detector.set_labels(self.LABELS)
        self.obj_detector.set_input_image(self.stable_frame,(640,480))
        self.obj_detector.run_detection()

    def get_results(self):
        """
        Prints the results in the given folder.
        """
        cv2.imwrite(os.path.join(self.result_path,"result.jpg"),self.obj_detector.get_inference_image())
        result_dict = self.obj_detector.get_results()
        result_dict = self.__postprocess_result(result_dict)

        # Writes the result to '.csv' file
        with open(os.path.join(self.result_path,"result.csv"), 'w') as f:
            writer = csv.writer(f,delimiter=',')
            all_results_array = []
            for arr in result_dict:
                result_arr = []
                result_arr.append(self.stable_frame_count)
                x_mid,y_mid = get_rect_centre(arr["bbox"][0],arr["bbox"][1],
                                                arr["bbox"][2],arr["bbox"][3])
                result_arr.append(x_mid)
                result_arr.append(y_mid)
                result_arr.append(arr["label"])

                all_results_array.append(result_arr)

            writer.writerows(all_results_array)

    def __postprocess_result(self,input_array):
        """
        Removes detected items outside the 'Tray'.
        """
        tray_rect_values = None
        postprocess_result = []

        # Get the bounding box of the tray
        for res in input_array:
            if res['label'] == 'Tray' :
                tray_rect_values = res['bbox']
                break

        # Remove object bounding boxes that are
        # not present in the 'Tray'
        for res in input_array:
            overlap_area = bb_intersection_over_union(tray_rect_values,res['bbox'])
            if overlap_area != 0.0:
                postprocess_result.append(res)

        print("[INFO] Total detections before post-processing : ",len(input_array))
        print("[INFO] Total detections after  post-processing : ",len(postprocess_result))

        return postprocess_result

if __name__=="__main__":
    # VIDEO = "/home/nj/HBRS/Studies/Sem-3/CV/Dataset/Videos/CV19_video_73.avi"
    # Result_Path = "/home/nj/Desktop/result"
    #
    # nut_detector = NutDetector(VIDEO,Result_Path,"","")
    # nut_detector.extract_most_stable_frame()
    # nut_detector.run_detection()
    # nut_detector.get_results()
    pass
