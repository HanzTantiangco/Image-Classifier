
# Start Training
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
import torch.nn.functional as F

def model_train(model, train_loader, validation_loader, learning_rate, epochs):
    """
    Trains the model.

    Args:
    model: Calls the model that was processed in model_loader.py
    train_loader: Loads the train dataset from data_loader.py
    validation_loader: Loads the validation dataset from data_loader.py
    learning_rate: float. Loads the user input learning rate from get_input_args.py
    epochs: int. Loads the user input epochs from get_input_args.py

    Returns:
    model: the newly trained model
    criterion_loss: the loss function used for the training
    """
    # Define the loss function
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Start training
    #epochs = 0
    steps = 0
    running_loss = 0
    print_every = 16
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            # Training loop
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validation_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        # Keep track of validation_loss
                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validation_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validation_loader):.3f}")
                running_loss = 0
                model.train()
    return model, criterion
