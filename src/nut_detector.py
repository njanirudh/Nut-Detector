import cv2
import argparse
from cv_object_detector import CVTFObjectDetector

LABELS = {0: "Walnut",
          1: "Peanut",
          2: "Hazelnut",
          3: "Tray"}

def main(video_path:str,graph_path:str ,pbtext_path:str ,result_path:str):

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    obj_detector = CVTFObjectDetector()
    obj_detector.set_parameters(graph_path, pbtext_path)

    while success:
        success, image = vidcap.read()

        obj_detector.set_input_image(image)
        obj_detector.run_detection()
        obj_detector.set_labels(LABELS)
        print('Read a new frame: ', success, count)
        count += 1

        cv2.imshow('img', obj_detector.get_inference_image())
        cv2.waitKey(20)

        obj_detector.reset()
        print(obj_detector.get_results())
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

    cv2.destroyAllWindows()

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Commandline application to detect Nuts in a video.')
    parser.add_argument('--video_path',
                        help='path of the video to run detection')
    parser.add_argument('--result_path',
                        help='path of the folder to save evaluation results.')

    args = parser.parse_args()
    print(args)

    IMAGE = "/home/nj/Desktop/CV/Dataset/Images/CV19_image_20.jpeg"
    FROZEN_GRAPH = "/home/nj/Desktop/CV/Trained_Models/FRCNN_Tray.pb"
    PBTEXT = "/home/nj/Desktop/CV/Dataset/Trained/FRCNN_TRAY/opencv_frcnn_tray.pbtxt"
    VIDEO = "/home/nj/HBRS/Studies/Sem-3/CV/Dataset/Videos/CV19_video_29.avi"


