from torch.utils.data import Dataset
import cv2

class SegmentationDataset(Dataset):
    """
    Custom dataset class for segmentation - inherits from torch.utils.data.Dataset class
    """
    def __init__(self, img_paths, mask_paths, im_transforms, mask_transforms):
        # store image and mask paths
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        # store transformations on the image data that are required
        self.im_transforms = im_transforms
        self.mask_transforms = mask_transforms
    
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

        # apply trnasforms to both image and mask if there are transforms
        if self.im_transforms is not None:
            img = self.im_transforms(img)
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        
        # return the tuple of transfomed image and mask
        return (img, mask)