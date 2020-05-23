def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optim_state']
    criterion = checkpoint['criterion']
    model.class_to_idx = checkpoint['class_to_idx']  #Load the 'class_to_idx' back into the model
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

model = load_checkpoint('checkpoint.pth')