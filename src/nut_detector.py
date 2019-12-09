import argparse

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Commandline application to detect Nuts in a video.')
    parser.add_argument('--video_path',
                        help='path of the video to run detection')
    parser.add_argument('--result_path',
                        help='path of the folder to save evaluation results.')

    args = parser.parse_args()
    print(args)