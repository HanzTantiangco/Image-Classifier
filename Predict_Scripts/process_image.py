from PIL import Image
import numpy as np

def process_image(image):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array.
    '''

    # Process a PIL image for use in a PyTorch model
    width, height = image.size

    # Resize the image
    image_copy = image.copy()
    image_copy = image_copy.thumbnail((256,256))

    # Crop the center of the image
    new_width = 224
    new_height = 224

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    image_crop = image.crop((left, top, right, bottom))

    # Convert to values to float
    np_image = np.array(image_crop)/255

    # Normalize
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds
    np_image = np_image.transpose((2, 0, 1))

    return np_image
