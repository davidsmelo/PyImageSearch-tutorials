import numpy as np
import cv2
import argparse
import imutils
from imutils import contours
from imutils import perspective
from scipy.spatial import distance

def midpoint(ptA,ptB):
    return ((ptA[0] + ptB[0]) * 0.5), ((ptA[1] + ptB[1]) * 0.5)

ap=argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Input Image -Path")
ap.add_argument("-w", "--width",required=True, type=float, help = "Width of the left-most contour found")

args=vars(ap.parse_args())

image=cv2.imread(args["image"])
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred=cv2.GaussianBlur(gray,(11,11),0) #Test 11, change to 7,7 if wrong contour

edged=cv2.Canny(blurred,50,100) # Test different threshold values
edged=cv2.dilate(edged, None, iterations=2)
edged=cv2.erode(edged, None, iterations=2)

cv2.imshow("Contours", edged)
cv2.waitKey()

cnts= cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
[cnts,_]=contours.sort_contours(cnts)
pixelsPerMetric=None

for c in cnts:
    if cv2.contourArea(c) < 100:
        continue

    orig=image.copy()
    box=cv2.minAreaRect(c)
    box=cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box=np.array(box, dtype=int)

    box=perspective.order_points(box)
    cv2.drawContours(orig,[box.astype("int")], -1, (0,0,255),2)

    for(x,y) in box:
        cv2.circle(orig,(int(x), int(y)), 5, (0,255,0),-1)


    (tl,tr,br,bl) =box
    (tltrX,tltrY) = midpoint(tl,tr)
    (blbrX,blbrY) = midpoint(bl,br)

    (tlblX,tlblY)= midpoint(tl,bl)
    (trbrX, trbrY) = midpoint(tr, br)

    cv2.circle(orig,(int(tltrX),int(tltrY)),5,(255,0,0),-1)
    cv2.circle(orig,(int(blbrX),int(blbrY)),5,(255,0,0),-1)
    cv2.circle(orig,(int(trbrX),int(trbrY)),5,(255,0,0),-1)
    cv2.circle(orig,(int(tlblX),int(tlblY)),5,(255,0,0),-1)


    cv2.line(orig,(int (tltrX), int(tltrY)),(int(blbrX), int(blbrY)),(0,255,255),2)
    cv2.line(orig,(int (trbrX), int(trbrY)),(int(tlblX), int(tlblY)),(0,255,255),2)



    dA=distance.euclidean((tltrX,tltrY), (blbrX,blbrY))
    dB=distance.euclidean((tlblX,tlblY), (trbrX,trbrY))

    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]

    dimA=dA/pixelsPerMetric
    dimB=dB/pixelsPerMetric

    cv2.putText(orig, "{:.1f}mm".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}mm".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    # show the output image
    cv2.imshow("Image", orig)
    cv2.waitKey(0)