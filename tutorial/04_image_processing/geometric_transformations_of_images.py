""" #.02
  Learn to apply different geometric transformation to images like translation, rotation, affine transformation etc.
  functions : cv2.getPerspectiveTransform

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Scaling
# img = cv2.imread('../factory/messi5.jpg')
# res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# height, width = img.shape[:2]
# res = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)

# Translation
# img = cv2.imread('../factory/messi5.jpg', 0)
# rows, cols = img.shape
# M = np.float32([[1, 0, 100], [0, 1, 50]])
# dst = cv2.warpAffine(img, M, (cols, rows))
# cv2.imshow('img', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Rotation
# img = cv2.imread('../factory/messi5.jpg', 0)
# rows, cols = img.shape
# M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)    # 90: degree
# dst = cv2.warpAffine(img, M, (cols, rows))
# cv2.imshow('img', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Affine Transformation
# img = cv2.imread('../factory/drawing.png')
# rows, cols, ch = img.shape
# pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
# pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
# M = cv2.getAffineTransform(pts1, pts2)
# dst = cv2.warpAffine(img, M, (cols, rows))
# plt.subplot(121), plt.imshow(img), plt.title('Input')
# plt.subplot(122), plt.imshow(dst), plt.title('Output')
# plt.show()

# Perspective Transformation
img = cv2.imread('../factory/sudokusmall.png')
rows, cols, ch = img.shape
pts1 = np.float32([[82, 95], [513, 73], [45, 535], [540, 542]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (300, 300))
plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()