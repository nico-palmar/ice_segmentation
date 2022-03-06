from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import numpy as np

class SegmentationDataset(Dataset):
    """
    Custom dataset class for segmentation - inherits from torch.utils.data.Dataset class
    """
    def __init__(self, img_paths, mask_paths, im_transforms, mask_transforms, testing=False, test_val=np.array([])):
        # store image and mask paths
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        # store transformations on the image data that are required
        self.im_transforms = im_transforms
        self.mask_transforms = mask_transforms
        self.testing_im = 0
        self.testing = testing
        self.test_val = test_val
    
    def __len__(self):
        # get the total length of items in the dataset, required since this class is inheriting from the torch.utils.data.Dataset class
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        """
        Return the image and corresponding mask for the index required with transformations applied
        """
        # get the image and mask from index
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # read the image and go from GRB to RBG
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # read the mask while specifying grey scale image
        mask = cv2.imread(mask_path, 0) 

        # view the desired value(s) if in testing mode
        if self.testing and self.testing_im < 20 and np.any(np.isin(mask, self.test_val)):
            mask_clone = mask.copy()
            # move all non mask elements to 255. Mask values are in range (0-100) so any 255 value will be separate
            # NOTE: this means that all mask values will appear as a dark mask in the image
            mask_clone[~np.isin(mask, self.test_val)] = 255
            fig, ax = plt.subplots(2)
            plt.title(f'Mask testing for {self.test_val}')
            ax[0].imshow(mask_clone)
            ax[1].imshow(img)
            plt.savefig(f'output/mask_testing_{idx}.png')
            plt.close()
            self.testing_im += 1

        # apply trnasforms to both image and mask if there are transforms
        if self.im_transforms is not None:
            img = self.im_transforms(img)
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        
        # return the tuple of transfomed image and mask
        return (img, mask)