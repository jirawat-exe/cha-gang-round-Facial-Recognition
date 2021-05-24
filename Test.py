import glob
import cv2
import numpy as np
# import tqdm as t
# import sklearn.svm as svm
# from joblib import load, dump
# import matplotlib.pyplot as plt
# import os

featureTr = []
labelTr = []
data = []
# Training Image

img = cv2.imread('Tr/emoji/i (1)/t (1).pgm')
# for _classname in range(1,16):
#   for _id in range(1, 64):
#     path = 'Tr/emoji/i (' + str(_classname) + ')/t (' + str(_id) + ').pgm';
#     img= cv2.imread(path,cv2.IMREAD_GRAYSCALE)
#     img= cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)
#     data.append(np.array(img).reshape(-1,1))
#     labelTr.append(_classname)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#ฝากทำ TS โดยสุ่มเลือกรูปจากใน TR มาทำเป็นเทสเคส