import cv2
import torch
from util import *
from darknet import MyDarknet

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_
#%%
#Create YOLOv3 model
model = MyDarknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
#%%
#model = MyDarknet("cfg/yolov3.cfg")
inp = get_test_input()
pred = model(inp, False)
print (pred)
print('Output tensor size :', pred.shape)

#%%
result = write_results(pred, 0.5, 80, nms_conf = 0.4)
print(result)

#%%
def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

num_classes = 80
classes = load_classes("data/coco.names")
print(classes)



