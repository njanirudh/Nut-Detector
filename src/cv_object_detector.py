import cv2

class CVTFObjectDetector:
    def __init__(self):
        self.input_image = None
        self.result_image = None
        self.cvNet = None
        self.cvOut = None
        self.rsize = (640,480)

    def set_labels(self,label):
        self.label_dict = label

    def set_input_image(self,image):
        self.input_image = cv2.resize(image,self.rsize,interpolation = cv2.INTER_AREA)
        self.result_image = cv2.resize(image,self.rsize,interpolation = cv2.INTER_AREA)

    def set_parameters(self,frozen_graph,pb_text):
        self.cvNet = cv2.dnn.readNetFromTensorflow(frozen_graph, pb_text)

    def run_detection(self):
        self.rows = self.input_image.shape[0]
        self.cols = self.input_image.shape[1]
        self.cvNet.setInput(cv2.dnn.blobFromImage(self.input_image, size=self.rsize,
                                                swapRB=True, crop=False))
        self.cvOut = self.cvNet.forward()

    def get_inference_image(self):
        for detection in self.cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > 0.6:
                left = detection[3] * self.cols
                top = detection[4] * self.rows
                right = detection[5] * self.cols
                bottom = detection[6] * self.rows
                cv2.rectangle(self.result_image, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
                cv2.putText(self.result_image, self.label_dict[int(detection[1])], (int(left-10),int(top-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        return self.result_image

    def get_results(self):
        result_array = list()
        for detection in self.cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > 0.6:
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
        self.input_image = None
        self.result_image = None
        self.rsize = (640,480)