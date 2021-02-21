from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model
import argparse
import numpy as np
import cv2
import imutils

ap=argparse.ArgumentParser()
ap.add_argument("-c","--cascade",required=True,help="path to where the face cascade reside")
ap.add_argument("-m","--model",required=True,help="path to pre-trained model")
ap.add_argument("-v","--video",help="path to (optional) video")
args=vars(ap.parse_args())

detector=cv2.CascadeClassifier(args['cascade'])
model=load_model(args["model"])

if not args.get("video",False):
    camera=cv2.VideoCapture(0)
else:
    camera=cv2.VideoCapture(args["video"])

while True:
    grabbed,frame= camera.read()
    
    if args.get("video") and not grabbed:
        break
    
    frame=imutils.resize(frame,width=300)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frameClone=frame.copy()
    
    rects=detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (X,Y,W,H) in rects:
        roi=gray[Y:Y+H,X:X+H]
        roi=cv2.resize(roi,(28,28))
        roi=roi.astype("float")/255.0
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)
        
        (notSmiling,smiling)=model.predict(roi)[0]
        if notSmiling>smiling:
            label="Not Smiling"
        else:
            label="Smiling"
            
        cv2.putText(frameClone,label,(X,Y-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
        cv2.rectangle(frameClone,(X,Y),(X+W,Y+H),(255,0,0),2)
        
    cv2.imshow("Face",frameClone)
        
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
        
camera.release()
cv2.destroyAllWindows()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        