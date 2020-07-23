import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image
import pandas as pd
import json
from get_input_args import get_input_args


def load_checkpoint(model, optimizer):
    ''' Loads model with earlier trained parameters to continue training with or
    ready for use with image predictions.
    '''

    in_arg = get_input_args()
    checkpoint = torch.load(in_arg.load_point, map_location=lambda storage, loc: storage)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for param in model.parameters():
        param.requires_grad = False

    return model, optimizer, model.class_to_idx


def process_image(image, model, device):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array.
    '''
    model.to(device)

    pro_pic = Image.open(image)

    test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    pro_pic = test_transforms(pro_pic)

    return pro_pic


def predict(model, train_data, device):
    ''' Predict the class (or classes) of an image using a trained
    deep learning model.
    '''
    in_arg = get_input_args()

    image_p = process_image(in_arg.image_path, model, device).to(device)
    image_p.unsqueeze_(0)
    image_p.type(torch.FloatTensor)

    model.eval()
    with torch.no_grad():

        logps = model.forward(image_p)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(in_arg.top_k, dim=1)
        idx_to_class = {value: key for key,value in train_data.class_to_idx.items()}
        probabilities = [p.item() for p in top_p[0]]
        classes = [idx_to_class[i.item()] for i in top_class[0]]

    model.train()

    return probabilities, classes


def print_outcome(probabilities, classes, model, train_data, device):
    ''' Function to print out the top classes and associated probabilities
    for image.
    '''

    in_arg = get_input_args()

    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Getting probs/classes
    probabilities, classes = predict(model, train_data, device)

    # Formatting flower names for index
    flower_names = [cat_to_name[str(i)] for i in classes]

    # Formatting data
    predictions_data = {'Probabilities:': pd.Series(probabilities), 'flower_names:': pd.Series(flower_names)}

    # Transforming data to dataframe
    predictions_data = pd.DataFrame(predictions_data)

    print(predictions_data.to_string(index=False), "\n")
