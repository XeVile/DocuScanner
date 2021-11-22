# Import
from pov_transform import fp_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2 as cv
import imutils

# FOR COMPATABILITY
import os

# For parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to scanned image")

args = vars(ap.parse_args())

im_loc = args['image']

def process_image(image):
    # Load image, find height and ratio of old vs new height.
    # Clone the image and then resize it into another im
    im = cv.imread(image)
    ratio = im.shape[0] / 500.0
    orig = im.copy()
    im = imutils.resize(im, height=500)
    
    return im

def detect_edge(image):
    # Filter the image in grayscale and then blur to locate edge
    # 
    gs = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Blur bg to gray
    gs = cv.GaussianBlur(gs, (5, 5), 0)
    im_edge = cv.Canny(gs, 75, 200)

    return im_edge

def find_doc(image):
    contour = cv.findContours(image.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    contour = sorted(contour, key=cv.contourArea, reverse=True)[:5]

    for c in contour:
        perimeter = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * perimeter, True)

        # See the length of the approx is 4 points if so then
        # Save and break loop
        if len(approx) == 4:
            docContour = approx
            break

    return docContour


def main():
    image = process_image(im_loc)
    result = detect_edge(image)

    print("Testing Edge detection")
    
    cv.imshow("Original", image)
    cv.imshow("Edge Detected", result)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

main()
