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
import numpy as np

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
    transforms.ToTensor(),
])

# create a class mapping variable
# after scanning the entire dataset, realize that classes are as follows:
# [  0.,   1.,   2.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 91.,  92., 100.]
class_mappings = {
    # assume 55, 1, 2 are water (graph 1 and 2 as water)
    # 0: 4, # the current 0 values are just water, these can be kept as they are
    1: 0, # map 0, 1 as water (0)
    2: 0,
    100: 1, # mark 100 as 1 (land)
    10: 2, # mark all labels from 10 to 80 as other class (2)
    20: 2,
    30: 2,
    40: 2,
    50: 2,
    60: 2,
    70: 2,
    80: 2,
    90: 3, # map high ice concentration as 3
    91: 3, 
    92: 3, 
}

class MaskToTensor:
    def __init__(self, mapping):
        # dictionary mapping with key value pairs
        self.mapping = mapping
    def __call__(self, pic):
        # TODO: ask if I need to divide by 100
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        for old_mask_val, new_mask_val in self.mapping.items():
            # change old mask image locations to new mask image locations according to the mapping desired
            img[img==old_mask_val] = new_mask_val
        return img


mask_transforms = transforms.Compose([
    transforms.ToPILImage(), 
    MaskToTensor(class_mappings),
])

# create the training and valdiation datasets
train_ds = SegmentationDataset(train_x_path, train_y_path, im_transforms, mask_transforms, testing=False, test_val=np.array([90, 91, 92]))
valid_ds = SegmentationDataset(valid_x_path, valid_y_path, im_transforms, mask_transforms)

train_loader = DataLoader(train_ds, shuffle=True,batch_size=config.BATCH_SIZE)
valid_loader = DataLoader(valid_ds, shuffle=False, batch_size=config.BATCH_SIZE)

# create a unet model force sea ice classification
n_classes = len(set(class_mappings.values())) # the number of classes are the number of distinct classes assigned above
ice_clf = UNet(n_classes).to(config.DEVICE)
opt = optim.Adam(ice_clf.parameters(), config.INIT_LR)
ce_loss = nn.CrossEntropyLoss()

# for e in tqdm(range(config.NUM_EPOCHS)):
for i in range(1):
    # set the model in training mode
    ice_clf.train()
    # initialize the total training and validation loss
    train_loss = 0
    valid_loss = 0
    for j, (x, y) in enumerate(train_loader):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE).long()    
        pred = ice_clf(x) # shape (BS, Classes, L, W)
        # want to take loss between pred (BS, Classes, L, W) and targ (BS, 1, L, W) = (BS, L, W)
        loss = ce_loss(pred, y)
        
        # update weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss += loss.item()
    
    # TODO: At the end of epoch, get validation loss
    # TODO: At the end of epoch, get metrics
    
    