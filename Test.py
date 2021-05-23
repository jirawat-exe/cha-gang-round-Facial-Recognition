import glob

import cv2
import numpy as np
import tqdm as t
import sklearn.svm as svm
from joblib import load, dump
import matplotlib.pyplot as plt
import os


featureTr = []
labelTr = []

# Training Image



img1 = cv2.imread('Tr/emoji/i(1)/t (1).pgm',0)
cv2.imshow("Showcase",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

""" 
tr/emoji/i(n)
"""


#ฝากทำ TS โดยสุ่มเลือกรูปจากใน TR มาทำเป็นเทสเคส