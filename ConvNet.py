# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout
class ConvNet(Module):
    def init(self):
        super(ConvNet, self).init()
        self.layer1 = Sequential(
            Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = Sequential(
            Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = Dropout()
        self.fc1 = Linear(7 * 7 * 64, 1000)
        self.fc2 = Linear(1000, 10)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = ConvNet()
torch.save(model, 'C:/ConvNet.pt')