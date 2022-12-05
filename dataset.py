import os
import cv2
from torch.utils.data import Dataset
import numpy as np

class RoadDataset(Dataset):
    def __init__(self, image_dir, groundtruth_dir, transform=None):
        self.image_dir = image_dir
        self.groundtruth_dir = groundtruth_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(image_dir))
        self.groundtruth_names = sorted(os.listdir(groundtruth_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        groundtruth_path = os.path.join(self.groundtruth_dir, self.groundtruth_names[idx])
        # read image
        image = cv2.imread(image_path)
        # read groundtruth and map to 0-1
        groundtruth = cv2.imread(groundtruth_path, cv2.IMREAD_GRAYSCALE)
        groundtruth[groundtruth <= 127] = 0
        groundtruth[groundtruth > 127] = 1
        # convert to numpy array
        image = np.array(image)
        groundtruth = np.array(groundtruth, dtype=np.float32)
        # transform
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=groundtruth)
            image = augmentations['image']
            groundtruth = augmentations['mask']

        return image, groundtruth
