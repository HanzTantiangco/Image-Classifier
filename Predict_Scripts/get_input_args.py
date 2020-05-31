import argparse

# Generates a random image to analyse
import random, os

folder_path = '/home/workspace/ImageClassifier/flowers/test/'
folder_number = 0
folder_number = random.randint(1,102)
folder_path = folder_path + str(folder_number)
random_select = random.choice(os.listdir(folder_path))
random_image_path = folder_path + '/' + random_select

def get_input_args():
    """
    This function takes a user input and returns a data structure which stores
    the command line arguments
    """

    parser = argparse.ArgumentParser(description='Prepare for prediction.')

    parser.add_argument('--gpu', type = bool, default=True, help =
    'Choose GPU for training')
    parser.add_argument('--save_dir', type = str, default = random_image_path,
    help = 'Set directory for image.')
    parser.add_argument('--checkpoint', type = str,
    default = '/home/workspace/ImageClassifier/checkpoint.pth',
    help = 'Pretrained model checkpoint directory.')
    parser.add_argument('--json', type = str,
    default = '/home/workspace/ImageClassifier/cat_to_name.json',
    help = 'Load JSON file.')
    parser.add_argument('--top_k', type = int, default = 5, help =
    'Input top k value.')

    return parser.parse_args()
