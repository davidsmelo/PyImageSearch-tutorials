import numpy as np
from scipy.spatial import distance as dist
import cv2
import argparse
import time
import dlib
import imutils
from imutils import face_utils
from imutils.video import FileVideoStream
from imutils.video import VideoStream



def eye_aspect_ratio(eye):
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    C =dist.euclidean(eye[0], eye[3])

    ear= (A+B)/(2*C)

    return ear

ap=argparse.ArgumentParser()
ap.add_argument("-v", "--video",type=str,default="", help="path")
ap.add_argument("-s", "--shape-predictor", required=True)

args=vars(ap.parse_args())

EYE_AR_THRESH=0.18
EYE_AR_CONSEC_FRAMES=3

COUNTER=0
TOTAL=0

print("[INFO] Initializing the predictor")
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(args["shape_predictor"])

(lStart,lEnd)=face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart,rEnd)=face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

print("Initialize file video stream")

vs=FileVideoStream(args["video"]).start()
filestream=True

vs=VideoStream(src=0).start()
filestream=False

time.sleep(1.0)

while True:

    if filestream and not vs.more():

        break

    frame=vs.read()
    frame=imutils.resize(frame,width=450)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rects=detector(gray,0)

    for rect in rects:
        shape=predictor(gray,rect)
        shape=face_utils.shape_to_np(shape)

        leftEye=shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR=eye_aspect_ratio(leftEye)
        rightEAR=eye_aspect_ratio(rightEye)

        ear= (leftEAR+rightEAR)/2.0

        leftEyeHull=cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame,[leftEyeHull], -1, (0,0,255),1)
        cv2.drawContours(frame,[rightEyeHull], -1, (0,0,255),1)

        if ear<EYE_AR_THRESH:
            COUNTER+=1

        else:
            if COUNTER >=EYE_AR_CONSEC_FRAMES:
                TOTAL+=1

            COUNTER=0
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Blink Eye Frame",frame)

    key=cv2.waitKey(1) & 0xFF

    if key== ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()