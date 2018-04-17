""" #.01
  learn how to read an image, how to display it and how to save it back
  functions : cv2.imread(), cv2.imshow(), cv2.imwrite()
  optional : learn how to display images with Matplotlib
  2018.2.20
   
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

" ------------------------------------------------------------------ "
# Load an color image in grayscale

# img = cv2.imread('../factory/messi5.jpg', 0)

# Read image
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

" ------------------------------------------------------------------ "
# Write image

img = cv2.imread('../factory/messi5.jpg', 0)
cv2.imshow('image', img)
k = cv2.waitKey(0)
if k == 27:             # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):     # wait for 's' key to save and exit
    cv2.imwrite('messigray.png', img)
    cv2.destroyAllWindows()

" ------------------------------------------------------------------ "
# Show image using matplotlib

# from matplotlib import pyplot as plt
# img = cv2.imread('../factory/messi5.jpg', 0)
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

" ------------------------------------------------------------------ "


