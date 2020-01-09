import argparse
from pathlib import Path
from utility import *
from nut_detector import NutDetector

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Commandline application to detect Nuts in a video.')
    parser.add_argument('--video_path', type=str, default='',
                        help='path of the video to run detection')
    parser.add_argument('--result_path', type=str, default='',
                        help='path to the folder to save evaluation results.')
    parser.add_argument('--frozen_graph', type=str, default='..\model\FRCNN_Tray.pb',
                        help='path to the tensorflow frozen graph.')
    parser.add_argument('--pbtxt_path', type=str, default='..\model\opencv_frcnn_tray.pbtxt',
                        help='path to the pbtxt path.')

    args = parser.parse_args()

    # -------------------------------------------------------

    if not (args.video_path or args.result_path or args.frozen_graph or args.pbtxt_path):
        print("Valid '--video_path', '--result_path', --frozen_graph or --pbtxt_path not set.")
    # ------
    if Path(args.video_path).is_file():
        print(INFO("Video found successfully."))
    else:
        print(ERROR("Video file not valid."))
    # ------
    if Path(args.frozen_graph).is_file():
        print(INFO("Frozen graph found successfully."))
    else:
        print(ERROR("Frozen graph file not found."))
    # ------
    if Path(args.pbtxt_path).is_file():
        print(INFO("Pbtxt found successfully."))
    else:
        print(ERROR("Pbtxt file not found."))

    # -------------------------------------------------------

    nut_detector = NutDetector(args.video_path,args.result_path,args.frozen_graph,args.pbtxt_path)
    nut_detector.extract_most_stable_frame()
    nut_detector.run_detection()
    nut_detector.get_results()