# This script will train a new network on a dataset and save the model the as a checkpoint
#Imports modules
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
import torch.nn.functional as F

# Imports functions created for this program
from get_input_args import get_input_args
from data_loader import data_loader
from model_loader import model_loader
from model_train import model_train
from model_test import model_test
from model_checkpoint import model_checkpoint


## Training validation log
# The training loss, validation loss, and validation accuracy are printed out as a network trains

def main():
    in_arg = get_input_args()

    # Data loader 
    train_data, validation_data, test_data, train_loader, validation_loader, test_loader, cat_to_name = data_loader(in_arg.save_dir)
    
    # Generate model
    model = model_loader(in_arg.arch, in_arg.hidden_unit)
    
    # GPU
    if in_arg.gpu == "True":
        device = "cuda"
    else:
        device = "cpu"
    
    # Model train
    model, criterion = model_train(model, train_loader, validation_loader, in_arg.learning_rate, in_arg.epochs)
    
    # Model test
    model_test(model, criterion, test_loader, validation_loader)
    
    # Model Checkpoint
    model_checkpoint(model, train_data)
    

# Call to main function to run the program
if __name__ == "__main__":
    main()