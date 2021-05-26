import cv2
import numpy as np
import sklearn.neighbors as sn
labelTr = []
data = []

# Training Image
# read img path
for _classname in range(1,16):
  for _id in range(1, 9):
    path = 'Tr/emoji/i (' + str(_classname) + ')/t (' + str(_id) + ').pgm'
    img = cv2.imread(path,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
    # cv2.imshow('Pic of Expecting', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    data.append(np.array(img).reshape(-1,1))
    labelTr.append(_classname)
    
tmpShape = np.array(data).shape
data = np.reshape(np.array(data), (-1,tmpShape[1])).T

# ขั้นที่1 คำนวณ Mean
meanData= np.array(np.mean(data,1))
meanVector= np.array(meanData).reshape((meanData.shape[0],-1))
data0mean = data - meanVector

# ขั้นที่2 Covariance Matrix
cov= (1.0/(data0mean.shape[0]-1))*(np.dot(data0mean.T, data0mean))

# ขั้นที่3 สกัด Eigenvector และ Eigenvalue จํากนั้นทำการจัดเรียงจากมากไปน้อย
val, vec = np.linalg.eigh(cov)
idx= val.argsort()[::-1]
val= val[idx]
vec= vec[:,idx]

# ขั้นที่4 ระบุจำนวน Eigenvector ที่ต้องการ
PCs = 10
SelectedVec= vec[:,0:PCs]

# ขั้นที่5 Training Feature Extraction by using first ten eigenvectors
EigenFace= np.dot(data0mean, SelectedVec)
featureTr= np.dot(EigenFace.T, data0mean)
featureTr= featureTr.T

# ขั้นที่6 Testing Feature Extraction by using first ten eigenvectors
path = 'Tr/emoji/i (3)/t (8).pgm'
img2= cv2.imread(path,cv2.COLOR_BGR2GRAY)
img2= cv2.resize(img2, (128,128), interpolation = cv2.INTER_AREA)
tmpTs= np.array(img2).reshape(-1,1)
featureTs= np.dot(EigenFace.T, tmpTs-meanVector).T
labelTs= 2

# ขั้นที่7 Image Classification
classifier = sn.KNeighborsClassifier(n_neighbors=1)
classifier.fit(featureTr, labelTr)
out = classifier.predict(featureTs)
print('Answer is ' + str(out))



cv2.imshow('Pic of Expecting', img2)
cv2.waitKey(0)
path2 = 'Tr/emoji/i (' + str(out[0]) + ')/t (1).pgm'
predictImg = cv2.imread(path2,cv2.COLOR_BGR2GRAY)
predictImg = cv2.resize(predictImg, (128,128), interpolation = cv2.INTER_AREA)
cv2.imshow('Pic of Predicting', predictImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

#ฝากทำ TS โดยสุ่มเลือกรูปจากใน TR มาทำเป็นเทสเคส