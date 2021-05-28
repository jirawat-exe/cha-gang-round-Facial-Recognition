import numpy as np
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from GoogLeNet import GoogLeNet
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm
import os
# for someone error
import collections
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# directory
TEST_ROOT = 'dataset/Tr/'  # path for test datasets
TRAIN_ROOT = 'dataset/Tr/'  # path for train datasets
PATH = './GoogLeNet.pth'  # path for load model

# GoogLeNet Preset
TRAIN_PREPROCESS = transforms.Compose([  # define transforms for train datasets
        transforms.Resize(256),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(224, padding=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
TEST_PREPROCESS = transforms.Compose([  # define transforms for test datasets
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def main():
    model = GoogLeNet(aux_logits=False)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    train_data = datasets.ImageFolder(root=TRAIN_ROOT, transform=TRAIN_PREPROCESS)
    train_loader = DataLoader(train_data, batch_size=40, shuffle=True, num_workers=0)
    # Loop training the model
    for epoch in tqdm(range(35)):  # loop over the dataset 35 times
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            # get the inputs then list in input and labels
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print statistics
            if i % 100 == 0:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    print('Finished Training')
    torch.save(model.state_dict(), PATH)

    # Testing the model
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    test_data = datasets.ImageFolder(root=TEST_ROOT, transform=TEST_PREPROCESS)
    test_loader = DataLoader(test_data, batch_size=40, shuffle=True, num_workers=0)
    classes = test_data.classes
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        model.eval()
        for data in tqdm(test_loader):
            images, labels = data
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions
            n_images = len(images)
            rows = int(np.sqrt(n_images))
            cols = int(np.sqrt(n_images))
            fig = plt.figure(figsize=(15, 15))
            for i in range(rows * cols):
                ax = fig.add_subplot(rows, cols, i + 1)
                image = images[i]
                image.clamp_(min=image.min(), max=image.max())
                image.add_(-image.min()).div_(image.max() - image.min() + 1e-5)
                ax.imshow(image.permute(1, 2, 0).cpu().numpy())
                label = classes[labels[i]]
                predict = classes[predictions[i]]
                ax.set_title(f'Expect: {label} \n Predict: {predict}')
                ax.axis('off')
            plt.show()
            # break
            for label, prediction, image in zip(labels, predictions, images):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    # print accuracy for each class
    print(correct_pred)
    od = collections.OrderedDict(sorted(correct_pred.items()))
    print(od)
    for classname, correct_count in od.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy of class {:5s} = {:.1f} %".format(classname,accuracy))

main()