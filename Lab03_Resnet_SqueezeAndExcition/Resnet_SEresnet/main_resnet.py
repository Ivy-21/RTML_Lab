import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
from copy import copy
from copy import deepcopy
import torch.nn.functional as F
from resnet import ResNet18
from util_resnet import train_model

# Set device to GPU or CPU

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Allow augmentation transform for training set, no augementation for val/test set

# train_preprocess = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# eval_preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


## Resize to 256
### AUGMENT for train
train_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

## Resize to 224
### No AUGMENT for test
eval_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


# Download CIFAR-10 and split into training, validation, and test sets.
# The copy of the training dataset after the split allows us to keep
# the same training/validation split of the original training set but
# apply different transforms to the training set and validation set.

full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True)

train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [40000, 10000])
train_dataset.dataset = copy(full_train_dataset)
train_dataset.dataset.transform = train_preprocess
val_dataset.dataset.transform = eval_preprocess

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=eval_preprocess)

# DataLoaders for the three datasets

BATCH_SIZE=16
NUM_WORKERS=4

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=NUM_WORKERS)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=NUM_WORKERS)

dataloaders = {'train': train_dataloader, 'val': val_dataloader}




#%%
from chosen_gpu import get_free_gpu
device = torch.device(get_free_gpu()) if torch.cuda.is_available() else torch.device("cpu")
print("Configured device: ", device)

#%%

resnet = ResNet18().to(device)
#count number of params
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'number of trainable parameters: {count_parameters(resnet)}')

# Optimizer and loss function
criterion = nn.CrossEntropyLoss().to(device)
params_to_update = resnet.parameters()
optimizer = optim.Adam(params_to_update, lr=0.01)
#%% 
#resnet.is_debug = True
best_model, val_acc_history, loss_acc_history = train_model(resnet, dataloaders, criterion, optimizer, 25, 'resnet18_bestsofar')

# #%%
# import numpy as np
# np.save('history/val_acc_history_resnet18_25.npy',np.array(val_acc_history))
# np.save('history/loss_acc_history_resnet18_25.npy',np.array(loss_acc_history))


torch.save(best_model.state_dict(), 'resnet-18-cifar-10.pth')

torch.save(val_acc_history, 'resnet-18-cifar-10-val-acc.pth')
torch.save(loss_acc_history, 'resnet-18-cifar-10-loss-acc.pth')