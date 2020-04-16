import numpy as np
import cv2
import argparse
import dlib
import imutils
import time
import playsound
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread


def sound_alarm(path):
    playsound.playsound(path)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2 * C)

    return ear


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark detection")
ap.add_argument("-a", "--audio", type=str, default="", help="path to .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam system")

args = vars(ap.parse_args())

EYE_AR_THRESH = 0.30
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0
ALARM_ON = False

print("Info loading facial landmark predictor")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255),1)

        if ear< EYE_AR_THRESH:
            COUNTER+=1
            print(COUNTER)

            if COUNTER> EYE_AR_CONSEC_FRAMES:


                if not ALARM_ON:
                    ALARM_ON=True

                    if(args["audio"]) != "":
                        t=Thread(target=sound_alarm, args=(args["audio"],))
                        t.daemon =True
                        t.start()

                cv2.putText(frame,"DROWSINESS DETECTED", (20,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (0,255,0), 2)

        else :
            COUNTER=0
            ALARM_ON=0

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()