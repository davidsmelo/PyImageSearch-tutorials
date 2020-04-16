import cv2
import argparse
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours

ap=argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Input Image")
args=vars(ap.parse_args())

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

image=cv2.imread(args["image"])
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred=cv2.GaussianBlur(gray, (5,5), 0)
edged=cv2.Canny(blurred,75,200)

cv2.imshow("Original", image)
cv2.imshow("Edged", edged)
cv2.waitKey()

cnts=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)

docCnt=None

if (len(cnts)>0):
    cnts=sorted(cnts,key=cv2.contourArea, reverse=True)

    for c in cnts:
        peri=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*peri,True)

        if len(approx)==4:
            docCnt=approx
            break

contourimg=image.copy()
cv2.drawContours(contourimg,[docCnt],-1, (0,0,255),2)
cv2.imshow("Test Sheet",contourimg)
cv2.waitKey()

paper=four_point_transform(image,docCnt.reshape(4,2))
warped=four_point_transform(gray,docCnt.reshape(4,2))

cv2.imshow("Paper transformed", paper)
cv2.imshow("Warped", warped)
cv2.waitKey()

thresh=cv2.threshold(warped,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]

cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)

questionsCnt=[]

for c in cnts:
    (x,y,w,h)=cv2.boundingRect(c)
    aspectR=w/float(h)

    if w>=20 and h>=20 and aspectR>=0.9 and aspectR<=1.1:
        questionsCnt.append(c)

papercnt=paper.copy()
cv2.drawContours(papercnt,questionsCnt,-1, (0,0,255),2)

cv2.imshow("Contours", papercnt)
cv2.waitKey()

questionsCnt=contours.sort_contours(questionsCnt,method="top-to-bottom")[0]

correct=0

for(q,i) in enumerate(np.arange(0,len(questionsCnt),5)):
    cnts=contours.sort_contours(questionsCnt[i:i + 5])[0]
    bubbled=None
    countFill=0

    for(j,c) in enumerate(cnts):
        mask=np.zeros_like(thresh, dtype="uint8")
        cv2.drawContours(mask,[c],-1,255,-1)

        mask=cv2.bitwise_and(thresh,thresh,mask=mask)
        total=cv2.countNonZero(mask)

        if total >500:
            countFill+=1

        if  (total>500 and (bubbled is None or total > bubbled[0])):
            bubbled=(total,j)

    print("Filled Circles =",  countFill)


    color=(0,0,255)
    k=ANSWER_KEY[q]
    if(bubbled is None):
        cv2.drawContours(paper, [cnts[k]], -1, color, 3)
    else:
        if (bubbled[1]==k):
            correct += 1
            color=(0,255,0)
    cv2.drawContours(paper, [cnts[k]],-1,color, 3)

score = (correct/5.0) * 100
print("INFO SOCRE : {:.2f}%".format(score))
cv2.putText(paper,"{:.2f}%".format(score),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)
cv2.imshow("Original", image)
cv2.imshow("Graded",paper)
cv2.waitKey()