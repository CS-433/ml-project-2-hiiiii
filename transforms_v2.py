import albumentations as A
from albumentations.pytorch import ToTensorV2
from itertools import chain, combinations
import constants as cst

list_of_transforms = [
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Transpose(p=1),
]

list_of_rotations = [
    (15, 15),
    (-15, -15),
    (30, 30),
    (-30, -30),
    (45, 45),
    (-45, -45),
    (60, 60),
    (-60, -60),
    (75, 75),
    (-75, -75),
    (90, 90),
    (-90, -90),
    (105, 105),
    (-105, -105),
    (120, 120),
    (-120, -120),
    (135, 135),
    (-135, -135),
    (150, 150),
    (-150, -150),
    (165, 165),
    (-165, -165),
    (180, 180),
]

list_of_random_rotations = [
    (0, 45),
    (45, 90),
    (90, 135),
    (135, 180),
    (180, 225),
    (225, 270),
    (270, 315),
    (315, 360)
]

# combine all transforms
def powerset(iterable):
    s = list(iterable)
    comb = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    for c in comb:
        yield A.Compose(c)

def add_rotations(transforms):
    new_transforms = transforms.copy()
    for transform in transforms:
        for rotation in list_of_random_rotations:
            new_transforms.append([A.Rotate(limit=rotation, p=1.0), *transform])
    return new_transforms

def rotations(transforms=[]):
    for rotation in list_of_rotations:
        transforms.append([A.Rotate(limit=rotation, p=1.0)])
    return transforms

def add_resize_and_normalization(transforms):
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

def get_transforms():
    transforms = list(powerset(list_of_transforms))
    transforms = add_rotations(transforms)
    # transforms = []
    # for transform in list_of_transforms:
    #     transforms.append([transform])
    # transforms = rotations(transforms)
    transforms = add_resize_and_normalization(transforms)
    return transforms

val_transforms = A.Compose(
    [
        A.Resize(height=cst.IMAGE_HEIGHT, width=cst.IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

test_transforms = A.Compose(
    [
        A.Resize(height=cst.TEST_IMAGE_HEIGHT, width=cst.TEST_IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)
