import torch
import os
import cv2

# define locaiton of dataset including images and masks
DATASET_PATH = os.path.join('ice_seg_dataset')
IMAGE_PATH = os.path.join(DATASET_PATH, 'Images')
MASK_PATH = os.path.join(DATASET_PATH, 'Masks')
TEST_SIZE = 0.08
VALIDATION_SIZE = 0.2

# get the device as a torch device to move tensors on the correct device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pin memory to speed up data transfer on GPU
PIN_MEMORY = True if DEVICE == torch.device("cuda") else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 64
# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "output"
# create the output folder if it doesn't exist
os.makedirs(os.path.join("..", BASE_OUTPUT), exist_ok=True)
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_water_bodies.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])