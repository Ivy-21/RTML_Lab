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
from SEbasicblock import ResSENet18



device = torch.device(get_free_gpu()) if torch.cuda.is_available() else torch.device("cpu")
print("Configured device: ", device)


dataset_test = dset.ImageFolder('/root/Lab03/chi_muff_test',
                           transform=transforms.Compose([
                               transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
test_dataloader = torch.utils.data.DataLoader(dataset_test,batch_size = 8,shuffle = True)


def evaluate(model, iterator, criterion):
    
    total = 0
    correct = 0
    epoch_loss = 0
    epoch_acc = 0
    
    predicteds = []
    trues = []
    
    model.eval()
    
    with torch.no_grad():
    
        for batch, labels in iterator:
            
            #Move tensors to the configured device
            batch = batch.to(device)
            labels = labels.to(device)

            predictions = model(batch.float())
                
            loss = criterion(predictions, labels.long())
            
            predictions = nn.functional.softmax(predictions, dim=1)            
            _, predicted = torch.max(predictions.data, 1)  #returns max value, indices
                       
            predicteds.append(predicted)
            trues.append(labels)            
            total += labels.size(0)  #keep track of total
            correct += (predicted == labels).sum().item()  #.item() give the raw number
            acc = 100 * (correct / total)
            
            epoch_loss += loss.item()
            epoch_acc += acc
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator),predicteds, trues

model_test = ResSENet18()
model_test.linear = nn.Linear(512,2)
model_test.eval()
model_test.to(device)
model_test.load_state_dict(torch.load(f'SE_muff_chi_model0.pth'))
criterion = nn.CrossEntropyLoss()
test_loss, test_acc, test_pred_label, test_true_label  = evaluate(model_test, test_dataloader, criterion)


#print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%', file=open(f"muf_chi_test.txt", "a"))
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')

