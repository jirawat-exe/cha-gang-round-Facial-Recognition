import glob

import cv2
import numpy as np
import tqdm as t
import sklearn.svm as svm
from joblib import load, dump
import matplotlib.pyplot as plt
import os

IMAGE_SIZE = (64, 128)

Train_PATH = "./Image_Dataset/Train/"
Test_PATH = "./Image_Dataset/Test/"
