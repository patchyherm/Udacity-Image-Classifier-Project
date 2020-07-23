# Image Classifier Project

Having implemented argument parser for this project, there are a number of optional arguments that can be input through the command line so as to customise the model. See below:

•	**Change image directory:** '--dir', type = str, default = 'flowers', help = 'path to the folder of flower images'

•	**Switch CNN model architecture:** '--arch', type = str, default = 'resnet50', help = 'choice of CNN Model Architecture; 'resnet50', 'alexnet', 'densenet121''

•	**Change learn rate for training model:** '--lr', type = float, default = 0.0015, help = 'learn rate for training model'

•	**Change number of epochs for training:** '--epochs', type = int, default = 5, help = 'number of epochs for training'

•	**Create checkpoint to save:** '--save_dir', type = str, default = 'default_save.pth', help = 'name of checkpoint to save; extension .pth'

•	**Input checkpoint to load:** '--load_point', type = str, default = 'default_load.pth', help = 'name of checkpoint to load'

•	**Change number of hidden units in classifier network:** '--hidden_units', type = int, default = 500, help = 'hidden units in network'

•	**Use GPU if available:** '--gpu', type = str, default = 'cpu', help = 'use gpu for processing'

•	**Change number of classes in prediction output:** '--top_k', type = int, default = 5, help = 'number of classes in output'

•	**Change category names:** '--category_names', type = str, default = 'cat_to_name.json', help = 'mapping of categories to names'

•	**Input path to image for prediction:** '--image_path', type = str, default = 'flowers/test/33/image_06460.jpg', help = 'path to the image that will be processed'

•	**Set a target for accuracy when training:** '--tgt', type = float, default = 1, help = 'optional accuracy target for training; will supersede epochs if met first; 0 <= tgt <= 1 '

**NB**: 

The notebook included is not needed for functionality, but I've included it as it serves as an outline for the project and is where most of the code was generated and testing advanced. 

The image directory has not been included due to it's large size but can be found through ImageNet.
