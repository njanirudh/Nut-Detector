import math

class bcolors:
    """
    Utility for colored terminal outputs.
    # https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def ERROR(input_str:str):
    return (bcolors.FAIL + "[ERROR] " + input_str + bcolors.ENDC)

def INFO(input_str:str):
    return ("[INFO] " + input_str)

def DEBUG(input_str:str):
    return (bcolors.OKGREEN + "[DEBUG] " + input_str + bcolors.ENDC)

#------------------------------------------------------------

def get_rect_centre(left, top, right, bottom):
    """
    Utility function to get centre of rectangle
    """
    return (math.ceil((left + right) / 2), math.ceil((top + bottom) / 2))


def bb_intersection_over_union(boxA, boxB):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou