import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image
import pandas as pd
import json
from get_input_args import get_input_args
from collections import OrderedDict

# Load the data
def data_loading():
    ''' Formats and loads image directories.
    '''

    in_arg = get_input_args()
    data_dir = in_arg.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #Load datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    #Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)

    return trainloader, testloader, validationloader, train_data, test_transforms


def create_model(device):
    ''' Creates a classifier network and appends to selected architecture.
    '''

    in_arg = get_input_args()
    arch = in_arg.arch

    # Allows for a choice of 3 model architectures
    if in_arg.arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif in_arg.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif in_arg.arch == 'densenet121':
        model = models.densenet121(pretrained=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu");

    # Freezing parameters for model so as to not
    # backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    in_features = {'resnet50': 2048, 'alexnet': 9216, 'densenet121': 1024}

    my_classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(in_features[in_arg.arch], in_arg.hidden_units)),
                                    ('relu', nn.ReLU()),
                                    ('dropout', nn.Dropout(0.2)),
                                    ('fc2', nn.Linear(in_arg.hidden_units, 102)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))

    # Replaces classifier in each architecture with my_classifier
    if in_arg.arch == 'resnet50':
        model.fc = my_classifier
    elif in_arg.arch == 'alexnet':
        model.classifier = my_classifier
    elif in_arg.arch == 'densenet121':
        model.classifier = my_classifier

    # Error function to use
    criterion = nn.NLLLoss()
    #Get optimizer ready for backprop against classifier parameters only
    optimizer = optim.Adam(my_classifier.parameters(), in_arg.lr)

    model.to(device)
    my_classifier.to(device)
    return model, optimizer, criterion;

def train_model(device, model, trainloader, optimizer, validationloader, criterion):
    ''' Trains model until either each epoch has been passed or target accuracy
    has been attained. Only my_classifier parameters are modified.
    '''

    in_arg = get_input_args()
    device = torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu == "gpu" else "cpu");

    model.to(device)
    epochs = in_arg.epochs
    steps = 0
    running_loss = 0
    validate_every = 5
    switch = 0

    with active_session():
        model.train()

        for epoch in range(epochs):
            if switch >= 1:
                break

            for images, labels in trainloader:
                if switch >= 1:
                    break

                steps += 1

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                logps = model.forward(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % validate_every == 0:
                    validation_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():

                        for images, labels in validationloader:

                            images, labels = images.to(device), labels.to(device)
                            logps = model.forward(images)
                            batch_loss = criterion(logps, labels)

                            validation_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                            if accuracy/len(validationloader) >= in_arg.tgt:
                                print("\nAccuracy threshold met at {} epochs and {} steps. Now breaking from training.\n".format(epoch, steps))
                                switch += 1
                                break

                    print("Epoch: {}/{}".format(epoch + 1, epochs),
                          "Training Loss: {:.3f}".format(running_loss/validate_every),
                          "Validation_loss: {:.3f}".format(validation_loss/len(validationloader)),
                          "Accuracy: {:.3f}".format(accuracy/len(validationloader)),
                          "Steps: {}".format(steps))

                    running_loss = 0
                    model.train()

    switch = 0
    steps = 0

    return model, optimizer


def save_checkpoint(train_data, model, optimizer):
    ''' Saves a checkpoint with model parameters that can be returned to at
    later time.
    '''
    in_arg = get_input_args()
    model.class_to_idx = train_data.class_to_idx

    #Define checkpoint with parameters to be saved
    checkpoint = {'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict(),
                  'model': model,
                  'optimizer': optimizer,
                  'optimizer_state_dict': optimizer.state_dict()}

    filepath = in_arg.save_dir

    #Save checkpoint
    torch.save(checkpoint, filepath)
    print("Checkpoint saved successfully\n")
