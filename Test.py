import glob
import cv2
import numpy as np
# import tqdm as t
# import sklearn.svm as svm
# from joblib import load, dump
# import matplotlib.pyplot as plt
# import os
import sklearn.neighbors as sn
featureTr = []
labelTr = []
data = []

# Training Image

for _classname in range(1,16):
  for _id in range(1, 9):
    path = 'Tr/emoji/i (' + str(_classname) + ')/t (' + str(_id) + ').pgm'
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    tmp = img.shape
    featureTr.append(tmp[0]/tmp[1])
    labelTr.append(_classname)
path = 'Tr/emoji/i (5)/t (5).pgm'
tmp = cv2.imread(path).shape
featureTs = [tmp[0]/tmp[1]]

labelTs = 2
classifier = sn.KNeighborsClassifier(n_neighbors=1)
classifier.fit(featureTr, labelTr)
out = classifier.predict(featureTs)
print('Answer is' + str(out))
cv2.waitKey(0)
cv2.destroyAllWindows()

#ฝากทำ TS โดยสุ่มเลือกรูปจากใน TR มาทำเป็นเทสเคส