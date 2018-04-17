""" #.01
  learn how to convert images from one color-space to another, like BGR ↔ Gray, BGR ↔ HSV etc.
  In addition to that, we will create an application which extracts a colored object in a video
  functions : cv2.cvtColor(), cv2.inRange() etc.

"""

import cv2
import numpy as np

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
# Return possible color flags
print(flags)
print(len(flags))

# In our application, we will try to extract a blue colored object
# Object tracking
cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    k = cv2.waitKey(5) & 0xFF11
    if k == 27:
        break

cv2.destroyAllWindows()


# How to find HSV values to track?
green = np.unit8([[[0,255,0]]])
hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
print(hsv_green)

