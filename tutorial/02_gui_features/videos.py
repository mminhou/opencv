""" #.02
  Learn to read video, display video and save video.
  Learn to capture from Camera and display it.
  functions : cv2.VideoCapture(), cv2.VideoWriter()

"""

import numpy as np
import cv2

" ------------------------------------------------------------------ "

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    # Capture frame-by-frame // ret = Boolean value 로써, 제대로 frame를 받아오면 True 실패시 False를 반환한다.
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

" ------------------------------------------------------------------ "
# Saving a video

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame, 0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()