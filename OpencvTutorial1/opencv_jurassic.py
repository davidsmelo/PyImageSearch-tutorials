import imutils
import cv2

image=cv2.imread("jp.png")
(h,w,d)=image.shape
print("width={}, height={}, depth={}".format(w,h,d))

cv2.imshow("Image", image)
cv2.waitKey(0)

(B,G,R)= image[100,50]
print("B={}, G={}, R={}".format(B,G,R))

roi=image[0:160,320:420]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

resized=cv2.resize(image, (200,200))
cv2.imshow("Resized", resized)
cv2.waitKey()

r=300/w
dim=(300, int(r*h))
resized=imutils.resize(image,width=300)
cv2.imshow("Resized", resized)
cv2.waitKey()

center=(w//2, h//2)
M=cv2.getRotationMatrix2D(center,-45,1.0)
rotated=cv2.warpAffine(image,M,(w,h))
cv2.imshow("Rotated",rotated)
cv2.waitKey(0)

rotated=imutils.rotate_bound(image, 45)

cv2.imshow("Rotated",rotated)
cv2.waitKey(0)

blurred=cv2.GaussianBlur(image, (11,11),0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

output=image.copy()
cv2.rectangle(output,(320,60),(420,160),(0,0,255))
cv2.imshow("Rectangle", output)
cv2.waitKey(0)

output=image.copy()
cv2.circle(output,(300,150),20,(0,0,255),-1)
cv2.imshow("Circle", output)
cv2.waitKey(0)

output=image.copy()
cv2.line(output,(300,150),(20,200),(0,0,255),5)
cv2.imshow("Line", output)
cv2.waitKey(0)

output=image.copy()
cv2.putText(output,"Jurassic Park", (10,25),cv2.FONT_ITALIC,0.7,(0,255,0),2)
cv2.imshow("Text", output)
cv2.waitKey(0)