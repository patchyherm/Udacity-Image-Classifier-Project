from get_input_args import get_input_args
from training_functions import data_loading, create_model, train_model, save_checkpoint
import torch

def start_training():
    ''' Function takes parameters as command line arguments and allows operators to
    train and save a deep learning model for later use.
    '''

    in_arg = get_input_args()
    device = torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu == "gpu" else "cpu")
    trainloader, testloader, validationloader, train_data, test_transforms = data_loading()
    print("\nCreating model...\n")
    model, optimizer, criterion = create_model(device)
    print("\nCommencing training...\n")
    model, optimizer = train_model(device, model, trainloader, optimizer, validationloader, criterion)
    print("\nTraining Complete.\n")
    save_checkpoint(train_data, model, optimizer)

start_training()
