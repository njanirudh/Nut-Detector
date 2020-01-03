from nut_detector import NutDetector
import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Commandline application to detect Nuts in a video.')
    parser.add_argument('--video_path', type=str,
                        help='path of the video to run detection')
    parser.add_argument('--result_path', type=str,
                        help='path of the folder to save evaluation results.')

    args = parser.parse_args()

    if not (args.video_path or args.result_path):
        print("Valid '--video_path' or '--result_path' not set.")
        exit(0)

    nut_detector = NutDetector(args.video_path,args.result_path)
    nut_detector.extract_most_stable_frame()
    nut_detector.run_detection()
    nut_detector.get_results()