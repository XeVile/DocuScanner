# Import
from typing import final
from imutils.convenience import resize
from pov_transform import fp_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2 as cv
import imutils

# FOR COMPATABILITY
# import os

# For parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to scanned image")

args = vars(ap.parse_args())

im_loc = args['image']

class scanner:
    # Load image, find height and ratio of old vs new height.
    def __init__(self, image):
        self.image = cv.imread(image)
        self.ratio = self.image.shape[0] / 500.0

    def _countedContours(self, contours):
        # Approximate the High vertice curve into simpler curve for performance
        for c in contours:
            perimeter = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02 * perimeter, True)

            # See the length of the approx is 4 points if so then
            # Save and break loop
            if len(approx) == 4:
                countedContour = approx
                break
            else:
                countedContour = approx
            
        return countedContour
    
    # Clone the image and then resize it into another image
    def process_image(self, height):
        resizedImage = imutils.resize(self.image, height=height)
        return resizedImage
    
    # Filter the image in grayscale and then blur to locate edge
    def detect_edge(self, image):
        grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Blur bg to gray
        grayscale = cv.GaussianBlur(grayscale, (5, 5), 0)
        edgeDetected = cv.Canny(grayscale, 75, 200)
        return edgeDetected
    
    # Find the contour of GRAYSCALED image
    # Grab the maximum possible contours
    # Sort the contour in Descending order and take most apparent contours
    # Better performance ^^^
    def highlight_edge(self, edge_image, source_image):
        contours = cv.findContours(edge_image.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]
        self._count = self._countedContours(contours)

        docContour = cv.drawContours(source_image, [self._count], -1, (0,255,0), 4)
        return docContour
    
    def warp_image(self, highlighted_edge):
        reshapedContour = self._count.reshape(4,2) * self.ratio
        finalImage = fp_transform(image=self.image, zone=reshapedContour)

        finalImage = cv.cvtColor(finalImage, cv.COLOR_BGR2GRAY)
        T = threshold_local(finalImage, 9, 'gaussian', 11)
        finalImage = (finalImage > T).astype("uint8") * 255
        return finalImage

def main():
    scan = scanner(im_loc)

    processed = scan.process_image(height=500)
    edge_detected = scan.detect_edge(processed)
    edge_highlighted= scan.highlight_edge(edge_detected, processed)

    final = scan.warp_image(edge_highlighted)
    
    print("Testing Contour detection")
    
    cv.imshow("Original", imutils.resize(processed, height = 650))
    cv.imshow("Edge Detected", edge_detected)
    cv.imshow("Drawn Contour", edge_highlighted)
    cv.imshow("FINAL", imutils.resize(final, height = 650))
    
    cv.waitKey(0)
    cv.destroyAllWindows()

main()
