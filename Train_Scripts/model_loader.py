import torch
from torch import nn
from torchvision import models

def model_loader(architecture, hidden_unit):
    # Choice of models
    densenet121 = models.densenet121(pretrained=True)
    vgg19 = models.vgg19(pretrained=True)       
    model_dictionary = {'densenet121': densenet121, 'vgg19': vgg19}
  
    model = model_dictionary[architecture]                     
    
    # Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Retrain classifier from collections import OrderedDict
    classifier = nn.Sequential(nn.Linear(1024, hidden_unit),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(hidden_unit, 102),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier

    return model