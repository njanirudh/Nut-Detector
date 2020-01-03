import math

def get_rect_centre(left, top, right, bottom):
    """
    Utility function to get centre of rectangle
    """
    return (math.ceil((left + right) / 2), math.ceil((top + bottom) / 2))