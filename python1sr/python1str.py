import numpy as np
import cv2 as cv
import time
import imutils
import argparse
from imutils.video import VideoStream

ap =argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument ("-m", "--model", required=True, help="path to Caffe pre-trained model")
