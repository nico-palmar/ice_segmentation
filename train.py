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
import seaborn as sns 
import pandas as pd

# variable that controls if the testin gset is written on a specific training run
write_testing_list = False

# sort the image and mask paths to match them up
img_paths = sorted(list(paths.list_images(config.IMAGE_PATH)))
mask_paths = sorted(list(paths.list_images(config.MASK_PATH)))

# split the data into training and testing sets. The training set will be further split into training and validation set
train_x_path, test_x_path, train_y_path, test_y_path = train_test_split(img_paths, mask_paths, test_size=config.TEST_SIZE, random_state=8)
train_x_path, valid_x_path, train_y_path, valid_y_path = train_test_split(train_x_path, train_y_path, test_size=config.VALIDATION_SIZE, random_state=8)

# write all validation images to disk
if write_testing_list:
    with open(config.TEST_PATHS) as f:
        for i in range(len(test_x_path)):
            test_x_path[i] += ":" + test_y_path[i]

        contents = "\n".join(test_x_path)
        f.write(contents)

im_transforms = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)), 
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))
])

mask_transforms = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)), 
    transforms.ToTensor(),
])

# create the training and valdiation datasets
train_ds = SegmentationDataset(train_x_path, train_y_path, im_transforms, mask_transforms)
valid_ds = SegmentationDataset(valid_x_path, valid_y_path, im_transforms, mask_transforms)

running_y = torch.Tensor([])
# for i, (x, y) in enumerate(train_ds):
#     # take a look at the range of pixel values given
#     x_mean = torch.mean(x, axis=0)
#     # running_y = torch.cat((running_y, y.view(-1)), axis=0)
#     # if i == 50:
#     #     break

    
#     fig, ax = plt.subplots(2)
#     ax[0].imshow(y.permute(1, 2, 0).detach())
#     ax[1].imshow(x.permute(1, 2, 0).detach())
#     plt.savefig(f'output/idx_{i}.png')
#     if i > 20:
#         break

# sns.histplot(train_ds[21][0].view(-1))
# plt.savefig(f'output/idx21_histplot.png')
idx_21_series = pd.Series(train_ds[21][0].view(-1))
descr = idx_21_series.describe()
print(descr)


# print(len(img_paths), len(mask_paths))
# print(len(train_x_path), len(valid_x_path), len(test_x_path))