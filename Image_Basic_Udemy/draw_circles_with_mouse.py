import cv2




def draw_circle(event, x,y,flags,params):
  if event==cv2.EVENT_RBUTTONDOWN:
      cv2.circle(img,(x,y),200, (0,0,255),thickness= 20)

img=cv2.imread("C:/Users/admin/Downloads/Udemy_CV/Computer-Vision-with-Python/DATA/dog_backpack.jpg")
cv2.namedWindow(winname="draw puppy")

cv2.setMouseCallback("draw puppy",draw_circle)



while True:

    cv2.imshow("draw puppy", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()