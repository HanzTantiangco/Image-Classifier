## Model architecture
# The training script allows users to choose from at least two different
# architectures available from torchvision.models

## Model hyperparameters
# The training script allows users to set hyperparameters for learning rate,
# number of hidden units, and training epochs

## Training with GPU
# The training script allows users to choose training the model on a GPU

import argparse

def get_input_args():
    """
    This function takes a user input and returns a data structure which stores
    the command line arguments
    """

    # Create parser using ArgumentParser
    parser = argparse.ArgumentParser(description='Prepare for training.')

    # Arguments
    parser.add_argument('--save_dir', type = str, 
                        default = '/home/workspace/ImageClassifier/flowers',
                        help = 'Set directory')
    parser.add_argument('--arch', type = str, default = 'densenet121',
                        help = 'Choose architecture')
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                        help = 'Choose learning rate')
    parser.add_argument('--hidden_unit', type = int, default = 500,
                        help = 'Choose number of hidden units')
    parser.add_argument('--epochs', type = int, default = 5,
                        help = 'Choose number of epochs')
    parser.add_argument('--gpu', type = bool, default=True,
                        help = 'Choose GPU for training')

    return parser.parse_args()
