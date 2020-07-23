import argparse

def get_input_args():

    parser = argparse.ArgumentParser()

    # Create command line arguments using add_argument() from ArgumentParser method
    parser.add_argument('--dir', type = str, default = 'flowers', help = 'path to the folder of flower images')
    parser.add_argument('--arch', type = str, default = 'resnet50', help = 'choice of CNN Model Architecture')
    parser.add_argument('--lr', type = float, default = 0.0015, help = 'learn rate for training model')
    parser.add_argument('--epochs', type = int, default = 5, help = 'number of epochs for training')
    parser.add_argument('--save_dir', type = str, default = 'default_save.pth', help = 'name of checkpoint to save; extension .pth')
    parser.add_argument('--load_point', type = str, default = 'default_load.pth', help = 'name of checkpoint to load')
    parser.add_argument('--hidden_units', type = int, default = 500, help = 'hidden units in network')
    parser.add_argument('--gpu', type = str, default = 'cpu', help = 'use gpu for processing')
    parser.add_argument('--top_k', type = int, default = 5, help = 'number of classes in output')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'mapping of categories to names')
    parser.add_argument('--image_path', type = str, default = 'flowers/test/33/image_06460.jpg', help = 'path to the image that will be processed')
    parser.add_argument('--tgt', type = float, default = 1, help = 'optional accuracy target for training; will supersede epochs if met first; 0 <= tgt <= 1 ')

    return parser.parse_args()
