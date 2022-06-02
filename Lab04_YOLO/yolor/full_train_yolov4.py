from __future__ import division
import time
from torch.utils.data import Subset
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
import albumentations as A
from custom_coco import CIOU_xywh_torch
from torch.nn.utils.rnn import pad_sequence
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
# from train import train_model, evaluate_model
from custom_coco import CustomCoco
import matplotlib.pyplot as plt

# Set device to GPU or CPU
gpu = "1"
device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

train_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

path2data_train="/data/COCO/train2017"
path2json_train="/data/COCO/annotations/instances_train2017.json"

path2data_val="/data/COCO/val2017"
path2json_val="/data/COCO/annotations/instances_val2017.json"



train_transform = A.Compose([
    A.SmallestMaxSize(1280),
    A.RandomCrop(width=1280, height=1280),
    # A.HorizontalFlip(p=0.5),
    # A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

eval_transform = A.Compose([
    A.SmallestMaxSize(1280),
    A.CenterCrop(width=1280, height=1280),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

# raw_train_dataset = torchvision.datasets.CocoDetection(root = path2data_train,
                                # annFile = path2json_train, transform=none_train_transform)

# train_dataset = torchvision.datasets.CocoDetection(root = path2data_train,
#                                 annFile = path2json_train, transform=train_transform)
BATCH_SIZE = 1

val_dataset = Subset(CustomCoco(root = path2data_val,
                                annFile = path2json_val, transform=eval_transform), list(range(0,20)))

def collate_fn(batch):
    return tuple(zip(*batch))

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2, collate_fn=collate_fn)

#If there's a GPU availible, put the model on GPU
# CUDA = torch.cuda.is_available()


print("Loading network.....")
model = Darknet("cfg/yolor_p6.cfg")
model.load_weights("cfg/yolor_p6.pt", backbone_only=True)
print("Network successfully loaded")


model.to(device)
# print("################ model: #############", model)

criterion = nn.CrossEntropyLoss()
params_to_update = model.parameters()
optimizer = optim.Adam(params_to_update, lr=0.001)
# for e in range(0, 40):
#     running_loss = 0.0
#     for inputs, labels, bboxes in val_dataloader:
#         # print("tuple label length: ", len(labels[0]))
#         inputs = torch.from_numpy(np.array(inputs)).squeeze(1).permute(0,3,1,2).float()
#         inputs = inputs.to(device)
#         labels = torch.stack(labels).to(device)
#         # print("labels.size(): ", labels.size())
#         running_corrects = 0

#         # zero the parameter gradients
#         # it uses for update training weights
#         optimizer.zero_grad()
#         with torch.set_grad_enabled(True):
#             outputs = model(inputs).to(device)
#             # print("outputs.size(): ", outputs.size())
#             # pred_xy = outputs[..., :2] / 224
#             # pred_wh = torch.sqrt(outputs[..., 2:4] / 224)

#             pred_xywh = outputs[..., 0:4] / 1280
#             # pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
#             pred_conf = outputs[..., 4:5]
#             pred_cls = outputs[..., 5:]


#             # label_xy = labels[..., :2] / 224
#             # label_wh = torch.sqrt(labels[..., 2:4] / 224)

#             label_xywh = labels[..., :4] / 1280

#             # label_xywh = torch.cat([label_xy, label_wh], dim=-1)
#             label_obj_mask = labels[..., 4:5]
#             label_noobj_mask = (1.0 - label_obj_mask)  # * (
#                 # iou_max < self.__iou_threshold_loss
#             # ).float()
#             lambda_coord = 0.001
#             lambda_noobj = 0.05
#             label_cls = labels[..., 5:]
#             loss = nn.MSELoss()
#             loss_bce = nn.BCELoss()

#             # ciou = CIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)

#             loss_coord = lambda_coord * label_obj_mask * loss(input=pred_xywh, target=label_xywh)
#             loss_conf = (label_obj_mask * loss_bce(input=pred_conf, target=label_obj_mask)) + \
#                         (lambda_noobj * label_noobj_mask * loss_bce(input=pred_conf, target=label_obj_mask))
#             loss_cls = label_obj_mask * loss_bce(input=pred_cls, target=label_cls)

#             loss_coord = torch.sum(loss_coord)
#             loss_conf = torch.sum(loss_conf)
#             loss_cls = torch.sum(loss_cls)

#             # print(pred_xywh.shape, label_xywh.shape)

#             ciou = CIOU_xywh_torch(pred_xywh, label_xywh)
#             # print(ciou.shape)
#             ciou = ciou.unsqueeze(-1)
#             # print(ciou.shape)
#             # print(label_obj_mask.shape)
#             loss_ciou = torch.sum(label_obj_mask * (1.0 - ciou))
#             # print(loss_coord)
#             loss =  loss_ciou +  loss_conf + loss_cls
#             loss.backward()
#             optimizer.step()
#             # statistics
#             running_loss += loss.item() * inputs.size(0)
#             # print('Running loss')
#             # print(loss_coord, loss_conf, loss_cls)
#     epoch_loss = running_loss / 750
#     print('Epoch', e)
#     print(epoch_loss)

    # print(calculate_APs(0.5, None, None))
    # break
    # print(x.shape)
    # print(y.shape)
    # print(w.shape)
    # print(h.shape)
    # print(obj.shape)
    # print(cls.shape)
    # break


########################## Testing ###################
images = "dog-cycle-car.png"
batch_size = 1
confidence = 0.5
nms_thesh = 0.4
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("data/coco.names")

inp_dim = int(model.net_info["height"])

model.eval()

read_dir = time.time()

# Detection phase

try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()
    
if not os.path.exists("des"):
    os.makedirs("des")

load_batch = time.time()
# loaded_ims = [letterbox_image(cv2.imread(x), (inp_dim, inp_dim)) for x in imlist]

img = cv2.imread(imlist[0])

print(type(img), img.shape)
img = letterbox_image(img, (inp_dim, inp_dim))
cv2.imwrite('test.jpg', img)
img = cv2.imread('test.jpg')
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
# print(img.shape)
# img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
print(img.shape)

loaded_ims = [img]


im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))


im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
print(im_dim_list)

leftover = 0
if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover            
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                        len(im_batches))]))  for i in range(num_batches)]  

write = 0

if torch.cuda.is_available():
    im_dim_list = im_dim_list.to(device)
    
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    # Load the image 
    start = time.time()
    if torch.cuda.is_available():
        batch = batch.to(device)
    with torch.no_grad():
        prediction = model(Variable(batch))

    nms = non_max_suppression(prediction)
    print("NMS #####: ", nms)
    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)

    end = time.time()

    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist 

    if not write:                      #If we have't initialised output
        output = prediction  
        write = 1
    else:
        output = torch.cat((output,prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if torch.cuda.is_available():
        torch.cuda.synchronize()       
try:
    output
except NameError:
    print ("No detections were made")
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
print("model.net_info: ", model.net_info["height"])
print("im_dim_list:", im_dim_list)
scaling_factor = torch.min(model.net_info["height"]/im_dim_list,1)[0].view(-1,1)

output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
output_recast = time.time()
class_load = time.time()
colors = [[255, 0, 0], [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255]]

draw = time.time()

def write(x, results):
    c1 = x[1:3].int().detach().cpu().numpy()
    c2 = x[3:5].int().detach().cpu().numpy()
    img = results[int(x[0])]
    print(img.shape)
    print(type(img))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('555',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


list(map(lambda x: write(x, loaded_ims), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format("des",x.split("/")[-1]))

list(map(cv2.imwrite, det_names, [cv2.cvtColor(loaded_ims[0], cv2.COLOR_BGR2RGB)]))
end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()