import pandas as pd
import numpy as np
# from Net import Net
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import load_state_dict_from_url
from typing import Optional, Tuple, List, Callable, Any

# train = pd.read
# train.head()

train_img = []
for _classname in range(1,16):
  for _id in range(1, 9):
    path = 'Tr/emoji/i (' + str(_classname) + ')/t (' + str(_id) + ').pgm'
    img = cv2.imread(path,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
    train_img.append(_classname)
