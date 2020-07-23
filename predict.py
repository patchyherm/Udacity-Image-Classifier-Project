from predict_functions import load_checkpoint, process_image, predict, print_outcome
from training_functions import data_loading, create_model
import torch
from get_input_args import get_input_args

def prediction():
    ''' Function takes in image as command line argument and will print out it's predicted
    class (or classes) using a trained deep learning model.
    '''

    in_arg = get_input_args()
    device = torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu == "gpu" else "cpu")
    print("Loading model...\n")
    model, optimizer, criterion = create_model(device)
    load_checkpoint(model, optimizer)
    trainloader, testloader, validationloader, train_data, test_transforms = data_loading()
    probabilities, classes = predict(model, train_data, device)
    print("\nTop {} prediction/s for input image are as follows:\n".format(in_arg.top_k))
    print_outcome(probabilities, classes, model, train_data, device)

prediction()
