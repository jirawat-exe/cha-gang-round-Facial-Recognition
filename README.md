# cha-gang-round-Facial-Recognition
### สมาชิก
1. จิรวัฒน์ ประทุมถิ่น 61070025 (IT/SE)
2. เตชินท์ โค้วประเสริฐ 61070063 (IT/Multi)
3. ทิวัตถ์ ทิพย์เลขา 61070067 (IT/SE)
4. ธรรมรัตน์ หาญประสพ 61070083 (IT/Network)
5. ปรมัตถ์ สุริยะรังษี 61070113 (IT/Multi)

## Project2 : Handcraft_base

#### ไฟล์ที่ใช้ในการ Run
training และ testing
```
handcraft_based.py
```
#### วิธีดึงรูปจาก Dataset
```
ดึงรูปมา Train
  for _classname in range(1,16):
    for _id in range(1, 8):
      path_train = 'dataset/Tr/i (' + str(_classname) + ')/t (' + str(_id) + ').pgm'
      img = cv2.imread(path_train,cv2.COLOR_BGR2GRAY)
      img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
      data.append(np.array(img).reshape(-1,1))
      labelTr.append(_classname)

 ดึงรูปมา Test
  test_class = 3
  path_test = 'dataset/Ts/i (' + str(test_class) + ')/t (1).pgm'
  img2= cv2.imread(path_test,cv2.COLOR_BGR2GRAY)
  img2= cv2.resize(img2, (128,128), interpolation = cv2.INTER_AREA)
  tmpTs= np.array(img2).reshape(-1,1)
  featureTs= np.dot(EigenFace.T, tmpTs-meanVector).T
  labelTs= 2
  
```



## Project3 : Learning_base
#### วิธีดึงรูปจาก Dataset
```
    train_data = datasets.ImageFolder(root=TRAIN_ROOT, transform=TRAIN_PREPROCESS)
    train_loader = DataLoader(train_data, batch_size=40, shuffle=True, num_workers=0)
    
    ซึ่ง TRAIN_ROOT เก็บ String PATH ของ dataset
    
    test_data = datasets.ImageFolder(root=TEST_ROOT, transform=TEST_PREPROCESS)
    test_loader = DataLoader(test_data, batch_size=40, shuffle=True, num_workers=0)
    
    PATH model ที่ Train แล้ว
```
#### วิธี Run Code
```

    เรียก main()

```
