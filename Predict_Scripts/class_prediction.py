import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
import torch.nn.functional as F
from PIL import Image

# Imports functions created for this program
from process_image import process_image

def class_prediction(image_path, model, topk):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    1. Predicting classes - the script reads in an image and a checkpoint then prints the most likely image class and its associated probability
    2. Top K classes - the script allows users to print out the top K classes along with associated probabilities
    
    '''
    # TODO: Implement the code to predict the class from an image file
    
    model.eval()
    model.to(torch.double)
    
    # Open image path and process
    image = Image.open(image_path)
    image_processed = process_image(image)
    image_torched = torch.from_numpy(image_processed)
    image_torched.unsqueeze_(0) # Prevents RuntimeError: expected stride to be a single 
                                # integer value or a list of 1 values to match the convolution 
                                # dimensions, but got stride=[2, 2]
    image_torched.double() # Prevents RuntimeError: Expected object of type torch.DoubleTensor 
                           # but found type torch.cuda.FloatTensor for argument #2 'weight'
    
    # Start prediction
    with torch.no_grad():
        output = model.forward(image_torched)
        
    # Output values convert to probabilities
    probability_prediction = torch.nn.functional.softmax(output, dim = 1)
    
    # Select top k probabilities and their classes
    top_p, top_class = torch.topk(probability_prediction, topk)
    
    # Class numbers convert to class labels
    class_to_idx = model.class_to_idx
    idx_to_class = {str(value) : int(key) for key, value in class_to_idx.items()}
    
    # Probability tensor convert to numpy array
    probabilities = top_p[0][:].tolist()
    
    # Top class numbers convert to class labels
    classes = np.array([idx_to_class[str(index)] for index in top_class.numpy()[0]])
    classes = list(classes)
    classes = [str(i) for i in classes]

    return probabilities, classes
    