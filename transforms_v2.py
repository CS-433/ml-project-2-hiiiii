import albumentations as A
from albumentations.pytorch import ToTensorV2
import constants as cst

################################################################################
# TRAIN AUGMENTATIONS
################################################################################

list_of_transforms = [
    A.Compose([]),
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Transpose(p=1),
    A.GaussianBlur(p=1),
    A.GaussNoise(p=1),
    A.RandomBrightnessContrast(p=1),
    A.RandomGamma(p=1),
    A.ChannelShuffle(p=1),
    A.ChannelDropout(channel_drop_range=(1, 2), fill_value=0, p=1),
]

list_of_random_rotations = [
    (0, 45),
    (45, 90),
    (90, 135),
    (135, 180),
    (180, 225),
    (225, 270),
    (270, 315),
    (315, 360),
]

def add_rotations(transforms):
    '''Combine all transforms with all rotations'''
    new_transforms = [] #transforms.copy()
    for transform in transforms:
        for rotation in list_of_random_rotations:
            new_transforms.append([A.Rotate(limit=rotation, p=1.0), *transform])
    return new_transforms

def add_resize_and_normalization(transforms):
    '''Add resize and normalization to all transforms'''
    new_transforms = []
    for transform in transforms:
        new_transforms.append(A.Compose(
            [
                A.Resize(height=cst.IMAGE_HEIGHT, width=cst.IMAGE_WIDTH),
                *transform,
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        ))
    return new_transforms

def get_train_transforms():
    '''Return all possible combinations of transforms'''
    transforms = []
    for transform in list_of_transforms:
        transforms.append([transform])
    transforms = add_rotations(transforms)
    transforms = add_resize_and_normalization(transforms)
    return transforms

################################################################################
# VALIDATION/TEST AUGMENTATIONS
################################################################################

list_of_val_transforms = [
    A.Compose([]),
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Transpose(p=1),
    A.Rotate(limit=(90, 90), p=1.0),
    A.Rotate(limit=(180, 180), p=1.0),
    A.Rotate(limit=(270, 270), p=1.0),
]

list_of_inverse_val_transforms = [
    A.Compose([]),
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Transpose(p=1),
    A.Rotate(limit=(-90, -90), p=1.0),
    A.Rotate(limit=(-180, -180), p=1.0),
    A.Rotate(limit=(-270, -270), p=1.0),
]

def get_test_transforms():
    '''Return all validation/test transforms'''
    transforms = []
    for transform in list_of_val_transforms:
        transforms.append(A.Compose(
            [
                A.Resize(height=cst.IMAGE_HEIGHT, width=cst.IMAGE_WIDTH),
                transform,
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        ))
    return transforms

def get_inverse_test_transforms():
    '''Return all inverse validation/test transforms'''
    transforms = []
    for transform in list_of_inverse_val_transforms:
        transforms.append(A.Compose(
            [
                A.Resize(height=cst.IMAGE_HEIGHT, width=cst.IMAGE_WIDTH),
                transform,
                ToTensorV2(),
            ]
        ))
    return transforms
