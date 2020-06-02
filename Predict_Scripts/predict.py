## PURPOSE: This script uses a trained network to predict the class for an
# input image
#
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py --gpu <GPU mode> --save_dir <directory with images>
#                        --checkpoint <trained model directory>
#                        --json <JSON file directory> --topk <k value>
#
#   Example call:
#    python check_images.py --gpu TRUE --save_dir /home/workspace/ImageClassifier
#                           /flowers/test/ --checkpoint /home/workspace/
#                           ImageClassifier/checkpoint.pth --json /home/
#                           workspace/ImageClassifier/cat_to_name.json
#                           --topk 5
#
##

# Imports functions created for this program
from get_input_args import get_input_args
from model_load import model_load
from json_load import json_load
from class_prediction import class_prediction

def main():
    """
    Runs the different modules together
    """
    # Retrieves Comman Line Arguments from user as input from the user running
    # the program from a terminal window
    in_arg = get_input_args()

    # GPU - Allows user to use the GPU to calculate the predictions
    if in_arg.gpu == "True":
        device = "cuda"
    else:
        device = "cpu"

    # Load json
    cat_to_name = json_load(in_arg.json)

    # Load model
    model = model_load(in_arg.checkpoint)

    # Prediction
    probabilities, classes = class_prediction(in_arg.save_dir, model, in_arg.top_k)

    # Convert class IDs to flower labels
    flower_labels = []
    prediction_dictionary = {}
    for key in classes:
        flower_labels.append(cat_to_name[key])

    # Display class names
    for flower, flower_probabilitiy in zip(flower_labels, probabilities):
        prediction_dictionary[flower] = round(flower_probabilitiy, 2)

    print(prediction_dictionary)

# Call to main function to run the program
if __name__ == "__main__":
    main()
