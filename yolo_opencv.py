import cv2
import argparse
import numpy as np

# command line args
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required = True, help = 'path to input image') #add_argument( <command-line flags>, <requirement to run>, description of argument>)
ap.add_argument('-c', '--config', required = True, help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required = True, help = 'path to yolo pre-trained weights')
ap.add_argument('cl', '--classes', required = True, help = 'path to text file containing class names')
args = ap.parse_args()