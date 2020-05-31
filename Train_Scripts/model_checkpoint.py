# Import modules
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
import torch.nn.functional as F

def model_checkpoint(model, train_data):
    """
    Saves the trained model into a dictionary that can be used later for
    predictions.

    Args:
    model: The trained model.
    train_data: the training dataset from the data_loader.py.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Dictionary with information to rebuild the model

    device = torch.device("cpu")
    model.to(device)

    checkpoint = {'model' : model,
                  'classifier' : model.classifier,
                  'input_size': 1024,
                  'output_size': 102,
                  'state_dict': model.state_dict(),
                  'class_to_idx': train_data.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
