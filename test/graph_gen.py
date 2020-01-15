import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_video_unique_count(path):
    vidcap = cv2.VideoCapture(path)

    count = 0
    old_frame = None
    full_frame_stable = None

    STATIONARY_FLAG = False

    unique_elements = []
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
                print("Frame No:", count, " | ", unique)
                unique_elements.append(unique)
            old_frame = current_frame
            count += 1

        else :
            break

    print(len(unique_elements))
    if len(unique_elements) > 100:
        return [0 for i in range(0,100)]

    padded_arr = padarray(unique_elements,100)
    return padded_arr

def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')

if __name__ == "__main__":

    VIDEO_LIST = glob.glob("/home/nj/HBRS/Studies/Sem-3/CV/Dataset/Videos/*.avi")
    print(len(VIDEO_LIST))
    VIDEO = "/home/nj/HBRS/Studies/Sem-3/CV/Dataset/Videos/CV19_video_6.avi" #20
    # VIDEO = "/home/nj/HBRS/Studies/Sem-3/CV/Dataset/Videos/CV19_video_250.avi"

    Y_LIST = []
    X_LIST = [i for i in range(0,100)]

    for i,pth in enumerate(VIDEO_LIST):
        # print(get_video_unique_count(pth))
        if i == 4:
            break

        print(pth)
        Y_LIST.append(get_video_unique_count(pth))


    for y_list in Y_LIST:
        # plt.plot(X_LIST, y_list, color=np.random.rand(3,))
        plt.plot(X_LIST, y_list, color='red',alpha=0.3)
        plt.scatter(X_LIST, y_list,label='Frame')

    plt.title("Stable frame detection")

    plt.xlabel("Number of frames ")
    plt.ylabel("Number of unique elements")

    plt.axhline(y=15, color='r', linestyle='dotted')
    plt.axhline(y=150, color='r', linestyle='dotted')

    plt.axvline(x=2, color='r', linestyle='dotted')
    plt.axvline(x=13, color='r', linestyle='dotted')

    # Add ticks
    locs, labels = plt.xticks()

    print(locs)
    plt.legend()

    plt.show()




