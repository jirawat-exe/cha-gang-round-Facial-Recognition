from tqdm import tqdm_notebook, trange
# from tqdm.notebook import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from pathlib import Path
import PIL, mimetypes, os
Path.ls = lambda x:list(x.iterdir())
import torchvision.models as models
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
data_path = Path('images2/Tr/emoji')
# print(data_path/'i (1)').ls())
import matplotlib.image as mpimg

# randomly select class
rand_imgs = (data_path/f'i ({np.random.randint(1,16)})').ls()
print(rand_imgs)

# show all images within class
for i,img in enumerate(rand_imgs):
    img_arr = mpimg.imread(img)
    if i%5 ==0: fig, axs = plt.subplots(1,5, figsize=(10,10))
    _ = axs[i%5].imshow(img_arr)
    _ = axs[i%5].set_title(img_arr.shape)

# define image transformation
img_tfm = transforms.Compose([
    transforms.Resize([105,105]),
    transforms.ToTensor(),
])

# 30 faces for training 10 faces for testing
img_arr_train = torch.empty(10,8,1,105,105)
img_arr_test = torch.empty(5,8,1,105,105)

train_idxs = np.random.choice(range(1,16),size=10, replace=False)
for i,idx in enumerate(train_idxs):
    for j,img in enumerate((data_path/f'i ({idx})').ls()):
        img_arr_train[i,j] = img_tfm(PIL.Image.open(img))

# test 10 faces for testing
test_idxs = [i for i in range(1,16) if not i in train_idxs]
for i,idx in enumerate(test_idxs):
    for j,img in enumerate((data_path/f'i ({idx})').ls()):
        img_arr_test[i,j] = img_tfm(PIL.Image.open(img))

# print(img_arr_train)
# print(img_arr_test)


# define triplet image loader
class Triplet_Image_Loader:
    # code modified from pairwise image loader in https://sorenbouma.github.io/blog/oneshot/
    
    def __init__(self, img_arr_train, img_arr_val):
        self.img_arr_train = img_arr_train
        self.img_arr_val = img_arr_val
        self.n_classes, self.n_examples, self.n_ch, self.h, self.w = img_arr_train.shape
        self.n_val = img_arr_val.size(0)
        
    def get_batch(self, n):
        # 1 - randomly pick n classes
        categories = np.random.choice(self.n_classes, size=(n,), replace=False)
        triplets = [torch.zeros((n, self.n_ch, self.h, self.w)) for i in range(3)]
        for i in range(n):
            # record anchor class
            category = categories[i]
            # sample 2 examples from class (one as anchor & one as positive example)
            idxs = np.random.choice(self.n_examples,size=2,replace=False)
            idx_anchor, idx_pos = idxs[0], idxs[1]
            triplets[0][i] = self.img_arr_train[category,idx_anchor]
            triplets[1][i] = self.img_arr_train[category,idx_pos]
            
            category_neg = (category + np.random.randint(1,self.n_classes)) % self.n_classes
            idx_neg = np.random.randint(0, self.n_examples)
            triplets[2][i] = self.img_arr_train[category_neg,idx_neg]

        return triplets
    
    def make_oneshot_task(self, N):
        
        categories = np.random.choice(self.n_val, size=(N,), replace=True)
        indices = np.random.randint(0,self.n_examples, size=(N,))
        true_cat = categories[0]
        ex1, ex2 = np.random.choice(self.n_examples, replace=False, size=(2,))
        test_image = torch.stack([self.img_arr_val[true_cat,ex1]]*N)
        support_set = self.img_arr_val[categories,indices]
        support_set[0] = self.img_arr_val[true_cat, ex2]
        pairs = [test_image, support_set]
        targets = torch.zeros((N,))
        targets[0] = 1
        
        return pairs, targets
    
    def test_oneshot(self,model, N, k):
        
        model.eval()
        n_correct = 0
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N)
            dists = model.get_distance(*inputs).cpu().detach().numpy()
            if np.argmin(dists) == 0:
                n_correct += 1
        pct_correct = (100*n_correct / k)
        
        return pct_correct

# initialize image loader
triplet_dl = Triplet_Image_Loader(img_arr_train,img_arr_test)

# test image loader
# triplets = triplet_dl.get_batch(8)
# for anchor, pos, neg in zip(triplets[0], triplets[1], triplets[2]):
#     fig, axs = plt.subplots(1,3)
#     _=axs[0].imshow(anchor.numpy().squeeze(0))
#     _=axs[0].set_title('anchor')
#     _=axs[1].imshow(pos.numpy().squeeze(0))
#     _=axs[1].set_title('positive')
#     _=axs[2].imshow(neg.numpy().squeeze(0))
#     _=axs[2].set_title('negative')
#     plt.show()

def get_fc_layers(fc_sizes, ps):
    fc_layers_list = []
    for ni,nf,p in zip(fc_sizes[:-1], fc_sizes[1:], ps):
        fc_layers_list.append(nn.Linear(ni, nf))
        fc_layers_list.append(nn.ReLU(inplace=True))
        fc_layers_list.append(nn.BatchNorm1d(nf))
        fc_layers_list.append(nn.Dropout(p=p))
    return nn.Sequential(*fc_layers_list)
    
class Resnet34FeatureExtractor(nn.Module):
    def __init__(self,n_ch=3,feat_dim=128,pretrained=True):
        super().__init__()
        
        # validate input channel
        assert n_ch in [1,3]
        
        self.feat_dim = feat_dim
        resnet34 = models.resnet34(pretrained=pretrained)
        # change input channel according to the input data
        resnet34.conv1 = nn.Conv2d(n_ch,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.body = nn.Sequential(*nn.ModuleList(resnet34.children())[:-1])
        self.dense = get_fc_layers(fc_sizes=[512,1024,512,feat_dim],ps=[.5,.5,.5])

    def forward(self, input):
        output = self.body(input)
        output = torch.flatten(output,1)
        output = self.dense(output)
        return output

'''
define architecture of NN that use CNN as image feature extractor (as defined above) and
output distance of positive and negative example to anchor image
'''
class TripletNet(nn.Module):
    
    def __init__(self, feature_extractor_module: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor_module
    
    def get_distance(self,img1,img2):
        img1, img2 = img1.to(device), img2.to(device)
        img1_feat, img2_feat = self.feature_extractor(img1), self.feature_extractor(img2)
        return F.pairwise_distance(img1_feat,img2_feat,p=2.0,keepdim=True)
    
    def forward(self,input):
        anchor, pos, neg = input
        p_dist = self.get_distance(anchor, pos)
        n_dist = self.get_distance(anchor, neg)
        return (p_dist, n_dist)

# define triplet loss function as originally defined in https://arxiv.org/pdf/1503.03832
class TripletLoss(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, input):
        p_dist, n_dist = input
        return torch.mean(torch.max(p_dist - n_dist + self.alpha, torch.zeros_like(p_dist)),0)

# find optimal learning rate by creating learning finder class
class CancelTrainException(Exception): pass

class LRFinderTriplet:
    
    def __init__(self, model, data_loader, bs, loss_func, opt, lr_range, max_iter):
        self.lrs = []
        self.losses = []
        self.model = model
        self.data_loader = data_loader
        self.bs = bs
        self.loss_func = loss_func
        self.opt = opt
        self.lr_range = lr_range
        self.max_iter = max_iter
    
    def run(self):
        best_loss = 1e9
        for i in trange(self.max_iter):
            # begin batch
            pos = i/self.max_iter
            lr = self.lr_range[0]*(self.lr_range[1]/self.lr_range[0])**pos
            for pg in self.opt.param_groups: pg['lr'] = lr

            xb = self.data_loader.get_batch(self.bs)
            loss = self.loss_func(self.model(xb))
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            # after step
            if loss.item() > best_loss*10: raise CancelTrainException()
            if loss.item() < best_loss: best_loss = loss.item() 

            # after batch
            self.lrs.append(lr)
            self.losses.append(loss.item())
    
    def plot_lr(self):
        # plot lr x loss
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')

# define function for training model
def train_n_batch_triplet(n, data_loader, loss_func, model, opt, bs, eval_every, loss_every, N_way, n_val, model_path):
    
    for i in tqdm_notebook(range(n)):
        xb = data_loader.get_batch(bs)
        loss = loss_func(model(xb))
        loss.backward()
        opt.step()
        opt.zero_grad()
        best_acc = 0
        # evaluate
        if (i%eval_every==0) & (i!=0):
            val_acc = data_loader.test_oneshot(model,N_way,n_val)
            print(f"validation accuracy on {N_way} supports of total {n_val} set:{val_acc}")
            if val_acc >= best_acc:
                print("saving")
                torch.save(model.state_dict(),model_path)
                best=val_acc
        if i % loss_every == 0:
            print("iteration {}, training loss: {:.2f},".format(i,loss.item()))

# find optimal lr
triplet_dl = Triplet_Image_Loader(img_arr_train, img_arr_test)
model = TripletNet(feature_extractor_module=Resnet34FeatureExtractor(n_ch=1,feat_dim=128,pretrained=False)).to(device)
loss_func = TripletLoss(.8)
# optimizer = optim.Adam(model.parameters(), lr=0.07)
optimizer = optim.SGD(model.parameters(), lr=0.07, momentum=0.9)
lr_finder = LRFinderTriplet(model=model, data_loader=triplet_dl, bs=8, loss_func=loss_func, opt=optimizer,
                    lr_range=[1e-7,1], max_iter=100)
lr_finder.run()

triplet_dl = Triplet_Image_Loader(img_arr_train,img_arr_test)
model = TripletNet(feature_extractor_module=Resnet34FeatureExtractor(n_ch=1,feat_dim=128,pretrained=False)).to(device)
loss_func = TripletLoss(.3)
# optimizer = optim.Adam(model.parameters(),lr=0.07)
# optimizer = optim.SGD(model.parameters(), lr=0.07, momentum=0.9)


train_n_batch_triplet(n=200, data_loader=triplet_dl, loss_func=loss_func, model=model, opt=optimizer, bs=8,
            eval_every=100, loss_every=50, N_way=8, n_val=80, model_path='model.pt')