import argparse
import numpy as np
import cv2
from skimage.exposure import rescale_intensity

cv2.boundingRect()
def convolve(image,kernel):
    (iH,iW)=image.shape[:2]
    (kH,kW)=kernel.shape[:2]

    cv2.imshow("Original Image",image)

    pad=int((kW -1) //2)
    image=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    output=np.zeros((iH,iW), dtype="float32")

    cv2.imshow("Padded Image",image)
    cv2.waitKey()

    for y in np.arange(pad, iH+pad):
        for x in np.arange(pad, iW+pad):
            roi=image[y-pad:y+pad+1, x-pad:x+pad+1]
            k=(roi*kernel).sum()
            output[y-pad,x-pad]=k
    output=rescale_intensity(output, in_range=(0,255))
    output=(output*255).astype("uint8")

    return output

ap=argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True, help="Path to input Image")
ap.add_argument("-k", "--kernel", required=False, help="Path to input Kernel")

args=vars(ap.parse_args())

small_blurs=np.ones((7,7), dtype="float") * (1.0/(7*7))
big_blurs=np.ones((21,21),dtype="float")* (1.0/(21*21))

sharpen=np.array((
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]), dtype="int")

laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")

sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")

sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")

kernelBank=(("small blur",small_blurs), ("big blurs", big_blurs), ("sharpen",sharpen), ("laplacian",laplacian), ("sobelx", sobelX), ("sobelY", sobelY))

image=cv2.imread(args["image"])
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

for (KernelName, kernel ) in kernelBank:
    convolveoutput=convolve(gray,kernel)
    opencvoutput=cv2.filter2D(gray,-1,kernel)

    cv2.destroyAllWindows()
    cv2.imshow("Original", gray)
    cv2.imshow("Convolve Output Kernel{}".format(KernelName), convolveoutput)
    cv2.imshow("OpenCv Output Kernel{}".format(KernelName), opencvoutput)
    cv2.waitKey()
    cv2.destroyAllWindows()

