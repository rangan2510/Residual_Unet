#%%
num_classes = 2

import torch
import torch.nn as nn
from torchvision import models


#%% ResNet18
net = models.resnet18()
net.fc = nn.Linear(512,num_classes)
net.load_state_dict(torch.load("BC_resnet18.dct", map_location=torch.device('cpu')))
torch.save(net, "resnet18.pth")
del net

#%% MNasnet
net = models.mnasnet1_0()
net.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),nn.Linear(in_features=1280, out_features=num_classes, bias=True))
net.load_state_dict(torch.load("BC_mnasnet1_0.dct", map_location=torch.device('cpu')))
torch.save(net, "mnasnet1_0.pth")
del net

#%% ResNext
net = models.resnext50_32x4d()
net.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
net.load_state_dict(torch.load("BC_resnext50_32x4d.dct", map_location=torch.device('cpu')))
torch.save(net, "resnext50_32x4d.pth")
del net

#%% ShuffleNet
net = models.shufflenet_v2_x1_0()
net.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
net.load_state_dict(torch.load("BC_shufflenet_v2_x1_0.dct", map_location=torch.device('cpu')))
torch.save(net, "shufflenet_v2_x1_0.pth")
del net

#%% DenseNet
net = models.densenet121()
net.classifier = nn.Linear(1024,num_classes)
# net.load_state_dict(torch.load("BC_densenet121.dct", map_location=torch.device('cpu')))
torch.save(net, "densenet121.pth")
del net
