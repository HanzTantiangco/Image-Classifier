import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
import torch.nn.functional as F

def model_test(model, criterion, test_loader, validation_loader):
    """
    Tests the trained network's accuracy using the test dataset, images the
    test_network has not seen.

    Args:
    model: Loads the trained model.
    criterion: Loads the the loss function used for the training.
    test_loader: Loads the test dataset from data_loader.py
    validation_loader: Loads the validation dataset from data_loader.py

    Returns:
    test_accuracy
    """


    accuracy = 0
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(device), labels.cuda(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    test_accuracy = f"Test accuracy: {accuracy/len(validation_loader):.3f}"
    print(test_accuracy)
    return test_accuracy
