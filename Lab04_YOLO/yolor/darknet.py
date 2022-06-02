from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from activations import *
from util import * 
import cv2
from layers import *
# from layers import ImplicitA, ImplicitM, ImplicitC, ControlChannel, Reorg, FeatureConcat


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (608,608))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. Returns a tensor.
    """
    # pylint: disable=no-member
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = letterbox_image(img, (inp_dim, inp_dim))
    return torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors



def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    routs = [] # list of layers which rout to deeper layers
    
    for index, x in enumerate(blocks[1:]):
        print(f"index is {index}, x blocks[1:]: ", x)
        module = nn.Sequential()
    
        #check the type of block
        #create a new module for the block
        #append to module_list
        
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            # print("this is convolution")
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
        
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
        
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
        
            #Add the convolutional layer
            # print("prev_filters: ", prev_filters)
            # print("filters: ", filters)
            # print("kernel_size: ",kernel_size)
            # print("stride: ", stride)
            # print("pad: ", pad)
            # print("bias: ", bias)
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("Conv2d".format(index), conv)
        
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("BatchNorm2d".format(index), bn)
        
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

            elif activation == "swish":
                activn = Swish()
                module.add_module("swish_{0}".format(index), activn)

            elif activation == "mish":
                activn = Mish()
                module.add_module("mish_{0}".format(index), activn)

            elif activation == "silu":
                activn = nn.SiLU()
                module.add_module("silu_{0}".format(index), activn)
            
            #If it's an upsampling layer
            #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            # print("this is upsampling")
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{0}".format(index), upsample)
                
       
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            filters = 0

            for i in range(len(x["layers"])):
                pointer = int(x["layers"][i])
                if  pointer > 0:
                    filters += output_filters[pointer]
                else:
                    filters += output_filters[index + pointer]
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            layers = x['from']
            x["from"] = x["from"].split(',')
            layers = x['from']
            layers = [int(a) for a in layers]
            filters = output_filters[-1]
            routs.extend([index + l if l < 0 else l for l in layers])
            module = WeightedFeatureFusion(layers=layers, weight='weights_type' in x)
            # print("this is shortcut")
            # shortcut = EmptyLayer()
            # module.add_module("shortcut_{}".format(index), shortcut)

        #Yolo is the detection layer
        elif x["type"] == "yolo":
            # print("this is yolo")
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
    
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        
        # Max pooling layer
        elif x["type"] == "maxpool":
            # print("this is maxpool")
            stride = int(x["stride"])
            size = int(x["size"])
            max_pool = nn.MaxPool2d(size, stride, padding=size // 2)
            module.add_module("maxpool_{}".format(index), max_pool)
        elif x['type'] == 'implicit_add' or x['type'] == 'implicit_mul':
            filters = int(x['filters'])
            if x['type'] == 'implicit_add':
                implicit_op = ImplicitA(filters)
            else:
                implicit_op = ImplicitM(filters)
            modules = implicit_op
            module.add_module('implicit_{0}'.format(index), modules)
        # elif x["type"] == "implicit_add":
        #     print("this is implicitAdd")
        #     filters = x['filters']
        #     modules = ImplicitA(channel = filters)
        #     module.add_module("implicit_add_{}".format(index), modules)

        # elif x["type"] == "implicit_mul":
        #     print("this is implicitMul")
        #     filters = x["filters"]
        #     modules = ImplicitM(channel = filters)
        #     module.add_module("implicit_mul_{}".format(index), modules)

        elif x["type"] == "implicit_cat":
            # print("this is implicitCat")
            filters = x['filters']
            modules = ImplicitC(channel = filters)
            module.add_module("implicit_cat_{}".format(index), modules)

        elif x["type"] == "control_channels":
            layers = x['from']
            filters = output_filters[-1]
            routs.extend([index + 1 if int(l) < 0 else l for l in layers])
            module = ControlChannel(layers = layers)
            # module.addmodule("control_channels_{}".format(index), modules)
        
        elif x["type"] == "reorg":
            reorg = Reorg()
            module.add_module("reorg_{0}".format(index), reorg)
            filters = prev_filters
            # pass
        # print("module: ", module)
        # print("previous_filters end of line: ", prev_filters)
        # print("output_filters: ", output_filters)
        # print("filters end of line: ", filters)
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        # print("output_filters end of line: ", output_filters)
    print("################end of class################")    
    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        
    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer
        
        write = 0
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            # print("module: ", module)
            # print("module_number: ", i)

            if module_type in ["convolutional", "upsample", "maxpool"]:
                # print("module_type: ", module_type)
                x = self.module_list[i](x)
            
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                maps = []
                for l in range(0, len(layers)):
                    if layers[l] > 0:
                        layers[l] = layers[l] - i
                    maps.append(outputs[i + layers[l]])
                x = torch.cat((maps), 1)
            #     print("route output shape: ", x.size())
                
    
            elif  module_type == "shortcut":
                # print("module from: ", module['from'])
                from_ = int(module["from"][0])
                x = outputs[i-1] + outputs[i+from_]
                # print("shortcut output size: ", x.size())
    
            elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
        
                #Get the number of classes
                num_classes = int (module["classes"])
        
                #Transform 
                # x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.cat((detections, x), 1)
        
            outputs[i] = x
            # print("outputs last: ", outputs)
        
        return detections


    def load_weights(self, weightfile, backbone_only=False):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):

            # CSP Darknet 53 has 104 modules in the backbone
            if i > 104 and backbone_only:
                break
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


# print(Darknet('/root/lab04/yoloR/cfg/yolor_p6.cfg'))