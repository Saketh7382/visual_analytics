from __future__ import print_function
import cv2 as cv
import numpy as np
import random as rng
rng.seed(12345)
def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Draw contours

    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 1, cv.LINE_8, hierarchy, 0)
    # Show in a window
    cv.imshow('Contours', drawing)
    cv.waitKey(0)
    cv.destroyAllWindows()
# Load source image
src = cv.imread('/home/saketh/Documents/Hackathons/AI-ML Tractor Analytics/SideLotImages/Scrub_Store_2.JPG',0)

# Convert image to gray and blur it
#src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#src_gray = cv.blur(src_gray, (3,3))

src_gray = cv.medianBlur(src,5)
#src_gray = cv.bilateralFilter(src_gray,3,25,25)
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey(0)