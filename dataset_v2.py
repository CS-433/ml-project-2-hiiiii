import os
import cv2
from torch.utils.data import Dataset
import numpy as np

class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=[]):
        # at least the normalization should be applied
        if transforms == []:
            raise ValueError("transforms cannot be empty")
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_names) * len(self.transforms)

    def __getitem__(self, idx):
        # get image and mask path
        image_idx = idx % len(self.image_names)
        image_path = os.path.join(self.image_dir, self.image_names[image_idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[image_idx])
        # read image
        image = cv2.imread(image_path)
        # read mask and map to 0-1
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        # convert to numpy array
        image = np.array(image)
        mask = np.array(mask, dtype=np.float32)
        # apply the corresponding transform
        transform_idx = idx // len(self.image_names)
        transform = self.transforms[transform_idx]
        augmentations = transform(image=image, mask=mask)
        image = augmentations['image']
        mask = augmentations['mask']
        return image, mask
