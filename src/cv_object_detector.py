import cv2
import math
from utility import get_rect_centre

class CVTFObjectDetector:
    def __init__(self):
        self.input_image = None
        self.result_image = None
        self.cvNet = None
        self.cvOut = None
        self.Threshold = 0.7

        print("[INFO] Threshold : ",self.Threshold)

    def set_labels(self,label:dict):
        """
        Set dictionary of possible label names.
        """
        self.label_dict = label
        print("[INFO] Label dictionary : ",label)

    def set_parameters(self,frozen_graph:str,pb_text:str):
        """
        Setting the paths to the 'frozen graph' and 'pb text'.
        """
        self.cvNet = cv2.dnn.readNetFromTensorflow(frozen_graph, pb_text)
        print("[INFO] OpenCV object detector created.")

    def set_input_image(self,image,resize):
        """
        Set input image for inference.
        """
        self.rsize = resize
        self.input_image = cv2.resize(image,self.rsize,interpolation = cv2.INTER_AREA)
        #self.result_image = cv2.resize(image,self.rsize,interpolation = cv2.INTER_AREA)
        self.result_image = image

    def run_detection(self):
        """
        Running the detection after the values are set.
        """
        self.rows = self.result_image.shape[0]
        self.cols = self.result_image.shape[1]
        self.cvNet.setInput(cv2.dnn.blobFromImage(self.input_image, size=self.rsize,
                                                swapRB=True, crop=False))
        self.cvOut = self.cvNet.forward()
        print("[INFO] Inference successfully completed.")

    def get_inference_image(self):
        """
        Returns the inference image with bounding boxes.
        """
        for detection in self.cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > self.Threshold:
                left = int(detection[3]  * self.cols)
                top = int(detection[4]  * self.rows)
                right = int(detection[5]  * self.cols)
                bottom = int(detection[6]  * self.rows)

                # Draw the bounding-box on the image
                cv2.rectangle(self.result_image,(left, top),(right, bottom), (23, 230, 210), thickness=2)
                cv2.drawMarker(self.result_image,get_rect_centre(left, top,right, bottom),(255,0,0))
                cv2.putText(self.result_image, self.label_dict[int(detection[1])] + " : " + str(round(score,4)),\
                 (int(left-10),int(top-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        print("[INFO] Result image generated successfully.")
        return self.result_image

    def get_results(self):
        """
        Returns a list of a dictionary containing 'label' and 'bounding'.
        """
        result_array = list()
        for detection in self.cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > self.Threshold:
                left = int(detection[3] * self.cols)
                top = int(detection[4] * self.rows)
                right = int(detection[5] * self.cols)
                bottom = int(detection[6] * self.rows)

                single_result = {}
                single_result["label"]=self.label_dict[int(detection[1])]
                single_result["bbox"]=[left,top,right,bottom]
                single_result["score"]=score

                result_array.append(single_result)

        print("[INFO] Result dictionary generated successfully.")
        return result_array

    def reset(self):
        """
        Resets the variables.
        Run once after every detection.
        """
        self.input_image = None
        self.result_image = None
        self.rsize = (640,480)