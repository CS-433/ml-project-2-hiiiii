import os
import cv2
from torch.utils.data import Dataset
import numpy as np

class RoadDataset(Dataset):
    def __init__(self, image_dir, groundtruth_dir, transforms=None):
        self.image_dir = image_dir
        self.groundtruth_dir = groundtruth_dir
        self.transforms = transforms
        self.image_names = sorted(os.listdir(image_dir))
        self.groundtruth_names = sorted(os.listdir(groundtruth_dir))

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        groundtruth_path = os.path.join(self.groundtruth_dir, self.groundtruth_names[idx])
        # read image
        image = cv2.imread(image_path)
        # read groundtruth and map to 0-1
        groundtruth = cv2.imread(groundtruth_path, cv2.IMREAD_GRAYSCALE)
        groundtruth[groundtruth > 127] = 1
        groundtruth[groundtruth <= 127] = 0
        # convert to numpy array
        image = np.array(image)
        groundtruth = np.array(groundtruth)
        # transform
        if self.transforms is not None:
            augmentations = self.transforms(image, groundtruth)
            image = augmentations['image']
            groundtruth = augmentations['groundtruth']

        return image, groundtruth
