# import pandas as pd
import numpy as np
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, random_split
# from GoogLeNet import GoogLeNet
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
import torch
import random
from skimage.io import imread
import cv2
# model = GoogLeNet()

training_img = []
PRETRAINED_SIZE = 224  # define pretrained size
PRETRAINED_MEANS = [0.485, 0.456, 0.406]  # define pretrained means
PRETRAINED_STDS = [0.229, 0.224, 0.225]  # define pretrained stds
TEST_TRANSFORMS = transforms.Compose([  # define transforms for test datasets
        transforms.Resize(PRETRAINED_SIZE),
        transforms.CenterCrop(PRETRAINED_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=PRETRAINED_MEANS, std=PRETRAINED_STDS)])
def getTrainData():
    train_data = datasets.ImageFolder('Tr/emoji/i (1)', transform=TEST_TRANSFORMS)
    print(train_data)
    return train_data

getTrainData()
# for _classname in range(1,16):
#   for _id in range(1, 9):
#     path = 'Tr/emoji/i (' + str(_classname) + ')/t (' + str(_id) + ').pgm'
#     img = cv2.imread(path,cv2.COLOR_BGR2GRAY)
#     img /= 255.0
#     img = img.astype('float32')
#     # img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
#     training_img.append(img)
#     training_x = np.array(training_img)
#     training_y = datasets.ImageFolder('Tr/emoji/i (' + str(_classname) + ')/t (' + str(_id) + ').pgm', transform='')
# criterion = CrossEntropyLoss()
# optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
# if torch.cuda.is_available():
#   model = model.cuda()
#   criterion = criterion.cuda()