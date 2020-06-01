## PURPOSE: This script will train a new network on a dataset and save the model
# the as a checkpoint
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py --save_dir <directory with images> --arch <model>
#             --learning_rate <learning rate value for the neural netwrok>
#             --hidden_unit <number of hidden units> --epochs <number of epochs>
#             --gpu <GPU mode>
#
#   Example call:
#    python check_images.py --dir /home/workspace/ImageClassifier/flowers
#                           --arch densenet121 --learning_rate 0.001
#                           --hidden_unit 500 --epochs 5 --gpu True
##

# Imports python modules
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
# The training loss, validation loss, and validation accuracy
# are printed out as a network trains

def main():
    """
    Runs the different modules together.
    """
    # Retrieves Command Line Arguments from user as input from the user running
    # the program from a terminal window
    in_arg = get_input_args()

    # Loads the dataset into different variables
    train_data, validation_data, test_data, train_loader, validation_loader,
    test_loader, cat_to_name = data_loader(in_arg.save_dir)

    # Generates deep learning model
    model = model_loader(in_arg.arch, in_arg.hidden_unit)

    # GPU
    if in_arg.gpu == "True":
        device = "cuda"
    else:
        device = "cpu"

    # Trains model
    model, criterion = model_train(model, train_loader, validation_loader,
                                   in_arg.learning_rate, in_arg.epochs)

    # Tests models against a validation dataset
    model_test(model, criterion, test_loader, validation_loader)

    # Saves the trained model
    model_checkpoint(model, train_data)

# Call to main function to run the program
if __name__ == "__main__":
    main()
