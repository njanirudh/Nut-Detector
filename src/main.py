from cv_object_detector import CVTFObjectDetector
import argparse

if __name__=="__main__":
    VIDEO = "/home/nj/HBRS/Studies/Sem-3/CV/Dataset/Videos/CV19_video_219.avi"
    Result_Path = "/home/nj/Desktop/result.csv"

    nut_detector = NutDetector(VIDEO,Result_Path)
    nut_detector.extract_most_stable_frame()
    nut_detector.run_detection()
    nut_detector.get_results()