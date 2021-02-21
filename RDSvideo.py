import numpy as np
import cv2

camera=cv2.VideoCapture(0)
if not camera.isOpened():
    print("Cant Read the Video")
    exit()

while(1):
    ret,frame=camera.read()
    
    if not ret:
        print("Cant Recieve Frame from the Video")
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(blurred, 30, 150)
    (T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
    #cv2.imshow("gray",gray)
    #cv2.imshow("edged",edged)
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    frame2=frame.copy()
    cv2.drawContours(frame2, cnts, -1, (0, 0, 255), 2)
    cv2.imshow("thresh",thresh)
    cv2.imshow("contours",frame2)
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()