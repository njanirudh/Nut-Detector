import cv2
import numpy as np

def find_box(img):
    pass

if __name__ == "__main__":
    img_path = "/home/nj/Desktop/CV/Dataset/Images/CV19_image_325.jpeg"

    # # Let's load a simple image with 3 black squares
    # img = cv2.imread(img_path)
    # img = cv2.resize(img,(640,480))
    #
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # binary = cv2.bitwise_not(gray)
    #
    # edges = cv2.Canny(binary, 100, 200)
    #
    # (_, contours) = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    # edges = cv2.drawContours(edges, contours, -1, (0,255,0))
    #
    # cv2.imshow("abc",edges)
    # cv2.waitKey(0)

    source_image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

    for threshold_itr in range(25, 100, 25):
        print("Current image threshold = " + str(threshold_itr))
        ret, thresh = cv2.threshold(gray_image, threshold_itr, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(source_image, contours, -1, (0, 0, 180), 3)

    cv2.imshow("abc",source_image)
    cv2.waitKey(0)