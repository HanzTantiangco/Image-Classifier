import torch

def model_load(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.class_to_idx = checkpoint['class_to_idx']  #Load the 'class_to_idx' back into the model
    model.load_state_dict(checkpoint['state_dict'])
    
    return model    