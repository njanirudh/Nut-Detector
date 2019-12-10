import cv2
from cv_object_detector import CVTFObjectDetector

IMAGE = "/home/nj/Pictures/Nuts/2019-10-14@14-51-04.png"
FROZEN_GRAPH = "/home/nj/Desktop/CV/Trained_Models/FRCNN_Tray.pb"
PBTEXT = "/home/nj/Desktop/CV/Dataset/Trained/FRCNN_TRAY/opencv_frcnn_tray.pbtxt"
LABELS = {0: "Walnut",
              1: "Peanut",
              2: "Hazelnut",
              3: "Tray"}

image = cv2.imread(IMAGE)

obj_detector = CVTFObjectDetector()
obj_detector.set_parameters(FROZEN_GRAPH,PBTEXT)
obj_detector.set_labels(LABELS)
obj_detector.set_input_image(image)
obj_detector.run_detection()

output = obj_detector.get_inference_image()
cv2.imwrite("result.jpg",output)
print(obj_detector.get_results())
