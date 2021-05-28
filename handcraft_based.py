import cv2
import numpy as np
import sklearn.neighbors as sn
import matplotlib.pyplot as plt
import random

labelTr = []
data = []
listimg = []
# Training Image
# read img path
#-----------------------------------------------------------
rate = int((75*8)/100) #ใส่ % เรทสุ่มรูป
#-----------------------------------------------------------
if rate <= 0 :
  rate = 1
for _classname in range(1,16):
  for _id in range(rate):
    path = 'dataset/Tr/emoji/i (' + str(_classname) + ')/t (' + str(random.randint(1, 8)) + ').pgm'
    img = cv2.imread(path,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
    listimg.append(img)
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
# print(EigenFace)
featureTr= np.dot(EigenFace.T, data0mean)
# print(featureTr)
featureTr= featureTr.T
# print(featureTr)

# ขั้นที่6 Testing Feature Extraction by using first ten eigenvectors
#path = 'dataset/Tr/emoji/i (3)/t (8).pgm'
#img2= cv2.imread(path,cv2.COLOR_BGR2GRAY)
#img2= cv2.resize(img2, (128,128), interpolation = cv2.INTER_AREA)
#tmpTs= np.array(img2).reshape(-1,1)
#featureTs= np.dot(EigenFace.T, tmpTs-meanVector).T
#labelTs= 2
tmptest = []
featureTs = []
listimgTs = []
labelTs = []
for _pred in range(1, 37):
  ranname = random.randint(1, 15)
  ranpic = random.randint(1, 8)
  path2 = 'dataset/Tr/emoji/i ('+str(ranname)+')/t ('+str(ranpic)+').pgm'
  # path2 = 'dataset/Tr/emoji/i ('+str(_pred)+')/t ('+str(ranpic)+').pgm'
  img2 = cv2.imread(path2,cv2.COLOR_BGR2GRAY)
  img2 = cv2.resize(img2, (128,128), interpolation = cv2.INTER_AREA)
  listimgTs.append(img2)
  tmptest.append(np.array(img2).reshape(-1,1))
  labelTs.append(ranname)
for i in range(36):
  featureTs.append(np.dot(EigenFace.T, tmptest[i]-meanVector).T)

print(featureTs)
# ขั้นที่7 Image Classification
out = []
classifier = sn.KNeighborsClassifier(n_neighbors=1)
classifier.fit(featureTr, labelTr)
for a in range(36):
  out.append(classifier.predict(featureTs[a]))

#print(len(labelTs))
#print(len(out))
#print(labelTs[1])
#print(out[1][0])

fig, ax=plt.subplots(6, 6)
count = 0

for i , axi in enumerate(ax.flat):
  axi.imshow(listimgTs[i],cmap='bone')
  axi.set(xticks=[],yticks=[])
  axi.set_xlabel("predict "+str(out[i][0]).split()[-1],color="green" if labelTs[i] == out[i][0] else "red")
  if labelTs[i] == out[i][0]:
    count += 1
  axi.set_ylabel("expected "+str(labelTs[i]).split()[-1])
count = (count / 36)*100
#plt.title("rate %.2f %%" % (count),loc="left")
fig.suptitle(("Correct Rate %.2f %%" % (count)), fontsize=16)
plt.show()



#ฝากทำ TS โดยสุ่มเลือกรูปจากใน TR มาทำเป็นเทสเคส