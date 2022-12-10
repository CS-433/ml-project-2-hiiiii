import torch

IMAGE_HEIGHT = 304
TEST_IMAGE_HEIGHT = 608
IMAGE_WIDTH = 304
TEST_IMAGE_WIDTH = 608
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
TRAIN_IMAGE_DIR = "train_data_x4/train_images/"
TRAIN_MASK_DIR = "train_data_x4/train_masks/"
VAL_IMAGE_DIR = "train_data_x4/val_images/"
VAL_MASK_DIR = "train_data_x4/val_masks/"
TEST_IMAGE_DIR = "test_data/"
BATCH_SIZE = 36
NUM_WORKERS = 4
PIN_MEMORY = True
NUM_EPOCHS = 5
