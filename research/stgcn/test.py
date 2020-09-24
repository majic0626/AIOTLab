import torch
import torch.nn as nn
from collections import OrderedDict


d = OrderedDict()
d['conv1'] = nn.Conv2d(1,20,5)
d['conv2'] = nn.Conv2d(1,20,5)
d['conv3'] = nn.Conv2d(1,20,5)
d['conv4'] = nn.Conv2d(1,20,5)

model = nn.Sequential(d)

# model = nn.Sequential(OrderedDict([
#           ('conv1', nn.Conv2d(1,20,5)),
#           ('relu1', nn.ReLU()),
#           ('conv2', nn.Conv2d(20,64,5)),
#           ('relu2', nn.ReLU())
#         ]))


for name, layer in model.named_modules():
    if name == '':
        continue
    print(name, layer)
