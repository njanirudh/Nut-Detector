import cv2
import math

class CVTFObjectDetector:
    def __init__(self):
        self.input_image = None
        self.result_image = None
        self.cvNet = None
        self.cvOut = None
        self.rsize = (640,480)
        self.Threshold = 0.7

    def set_labels(self,label):
        """
        """
        self.label_dict = label

    def set_parameters(self,frozen_graph,pb_text):
        """
        """
        self.cvNet = cv2.dnn.readNetFromTensorflow(frozen_graph, pb_text)

    def set_input_image(self,image):
        """
        """
        self.input_image = cv2.resize(image,self.rsize,interpolation = cv2.INTER_AREA)
        #self.result_image = cv2.resize(image,self.rsize,interpolation = cv2.INTER_AREA)
        self.result_image = image

    def run_detection(self):
        """
        """
        self.rows = self.result_image.shape[0]
        self.cols = self.result_image.shape[1]
        self.cvNet.setInput(cv2.dnn.blobFromImage(self.input_image, size=self.rsize,
                                                swapRB=True, crop=False))
        self.cvOut = self.cvNet.forward()

    def __get_centre(self,left,top,right,bottom):
        """
        """
        return (math.ceil((left + right)/2), math.ceil((top + bottom)/2))

    def get_inference_image(self):
        """
        """
        for detection in self.cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > self.Threshold:
                left = int(detection[3]  * self.cols)
                top = int(detection[4]  * self.rows)
                right = int(detection[5]  * self.cols)
                bottom = int(detection[6]  * self.rows)
                cv2.rectangle(self.result_image,(left, top),(right, bottom), (23, 230, 210), thickness=2)
                cv2.drawMarker(self.result_image,self.__get_centre(left, top,right, bottom),(255,0,0))
                cv2.putText(self.result_image, self.label_dict[int(detection[1])], (int(left-10),int(top-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        return self.result_image

    def get_results(self):
        """
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

                result_array.append(single_result)

        return result_array

    def reset(self):
        """
        Resets the variables.
        Run once after every detection.
        """
        self.input_image = None
        self.result_image = None
        self.rsize = (640,480)