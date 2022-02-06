from internal_files.dataset import SegmentationDataset
from internal_files.model import UNet
from internal_files import config
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
# variable that controls if the testin gset is written on a specific training run
write_testing_list = True

# sort the image and mask paths to match them up
img_paths = sorted(list(paths.list_images(config.IMAGE_PATH)))
mask_paths = sorted(list(paths.list_images(config.MASK_PATH)))

# split the data into training and testing sets. The training set will be further split into training and validation set
train_x_path, test_x_path, train_y_path, test_y_path = train_test_split(img_paths, mask_paths, test_size=config.TEST_SIZE, random_state=8)
train_x_path, valid_x_path, train_y_path, valid_y_path = train_test_split(train_x_path, train_y_path, test_size=config.VALIDATION_SIZE, random_state=8)

# write all validation images to disk
if write_testing_list:
    for i in range(len(test_x_path)):
        test_x_path[i] += ":" + test_y_path[i]

    # with open(config.TEST_PATHS) as f:
    #     for i in range(len(test_x_path)):
    #         test_x_path[i] += test_y_path[i]

    #     contents = "\n".join(test_x_path)

# print(len(img_paths), len(mask_paths))
# print(len(train_x_path), len(valid_x_path), len(test_x_path))