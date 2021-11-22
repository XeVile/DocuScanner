# IMPORT
import numpy as np
import cv2 as cv



def init_border(zone):
    # Init a list of coordinates that wrap around the image border corners
    rect = np.zeros((4, 2), np.float32)
    
    # Define the coordinates
    # Find the total of the point (x,y) to calculate farthest points
    # Find the difference of point (x - y) to check if x is smaller
    s = zone.sum(axis=1)
    d = np.diff(zone, axis=1)

    # Assigning the coordinates ordered
    # Will be arranged in: p1   p4
    #                      p2   p3      <- Rectangle
    #
    # 
    rect[0] = zone[np.argmin(s)]
    rect[1] = zone[np.argmin(d)] # Since X < Y and X < other X values
    rect[2] = zone[np.argmax(s)]
    rect[3] = zone[np.argmax(d)] # Since X > Y and X > other X values

    return rect

def fp_transform(image, zone):
    # Order the border coordinates
    # Divide the transformable points
    rect = init_border(zone)
    (top_l, top_r, bot_l, bot_r) = rect

    # Calculate distance between points(right - left) at the top; points at the bottom
    # 
    widthA = np.sqrt(((bot_r[0] - bot_l[0]) ** 2) + ((bot_r[1] - bot_l[1]) ** 2))
    widthB = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    maxW = max(int(widthA), int(widthB))

    # Calculate distance between points(top and bottom) at the right; at the left
    #
    heightA = np.sqrt(((top_r[0] - bot_r[0]) ** 2) + ((top_r[1] - bot_l[1]) ** 2))
    heightB = np.sqrt(((top_r[0] - bot_r[0]) ** 2) + ((top_r[1] - bot_l[1]) ** 2))
    maxH = max(int(heightA), int(heightB))

    # Compose a top-down view for accurate tranformation
    #
    target = np.array([[0,0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], np.float32)

    # Use OpenCV to compute perspective transformation matrix
    #
    t_M= cv.getPerspectiveTransform(rect, target)
    final = cv.warpPerspective(image, t_M, (maxW, maxH))

    return final
