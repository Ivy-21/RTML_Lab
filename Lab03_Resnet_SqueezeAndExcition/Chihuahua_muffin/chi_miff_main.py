import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
from copy import copy
from copy import deepcopy
import numpy as np
from torchvision.transforms.transforms import RandomCrop

import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from chosen_gpu import get_free_gpu
from util_chi_muff import train_model


device = torch.device(get_free_gpu()) if torch.cuda.is_available() else torch.device("cpu")
print("Configured device: ", device)

#dataroot = "/Users/winwinphyo/Downloads/data/"

# dataset = dset.ImageFolder(root=dataroot,
#                        transform=transforms.Compose([
#                            transforms.Resize(image_size),
#                            transforms.CenterCrop(image_size),
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                        ]))


dataset = dset.ImageFolder('/root/Lab03/chi_muf',
                       transform=transforms.Compose([
                           transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))





folds = 8
skf = StratifiedKFold(n_splits=folds, shuffle=True)


from SEbasicblock import ResSENet18
models = []
def make_model(ResSENet18):
    model = ResSENet18()
    model.load_state_dict(torch.load('SEresnet18_bestsofar.pth'))
    model.linear = nn.Linear(512,2)
    model.eval()
    return model

n_models = 8

for i in np.arange(n_models):
    fig,ax = plt.subplots(1,2,sharex=True,figsize=(20,5))
    model_acc = 0
    for fold, (train_index, val_index) in enumerate(skf.split(dataset, dataset.targets)):
        print('********************* Fold {}/{} ******************** '.format(fold, 8 - 1), 
              file=open(f"SE_muff_chi_model{i}.txt", "a"))
        batch_size = 4
        train = torch.utils.data.Subset(dataset, train_index)
        val = torch.utils.data.Subset(dataset, val_index)
        
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, 
                                                   shuffle=True, num_workers=0, 
                                                   pin_memory=False)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, 
                                                 shuffle=True, num_workers=0, 
                                                 pin_memory=False)

        dataloaders = {'train': train_loader, 'val': val_loader}

        model = make_model(ResSENet18)
        model.to(device)
        dataloaders = {'train': train_loader, 'val': val_loader}
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer =  optim.Adam(model.parameters(), lr = 0.005 + 0.005*i)

        val_acc_history, loss_acc_history = train_model(model, dataloaders, 
                                                        criterion, optimizer, 
                                                        25, f'SE_muff_chi_model{i}')

        # ax[0].plot(np.arange(25),np.array(val_acc_history),label = f"val acc fold{fold}")
        # ax[1].plot(np.arange(25),np.array(loss_acc_history),label = f"val acc fold{fold}")
        # ax[0].set_xlabel("Epochs")
        # ax[1].set_xlabel("Epochs")
        # ax[0].set_ylabel("Accuracy")
        # ax[1].set_ylabel("Loss")    
        # ax[0].set_title(f"Accuracy vs Epochs of model{i}")
        # ax[1].set_title(f"Loss vs Epochs of model{i}")
        # ax[0].legend()
        # ax[1].legend()
        # ax[0].grid(True)
        # ax[1].grid(True)
        #print(len(val_acc_history))
        #print(sum(val_acc_history))         
        model_acc = model_acc + sum(val_acc_history)/len(val_acc_history)

    #plt.show()
    print(f'Average accuracy of model{i}: {model_acc/8}', file=open(f"muf_chi_Average_acc.txt", "a"))


