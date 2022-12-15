import torch

IMAGE_HEIGHT = 400
TEST_IMAGE_HEIGHT = 608
IMAGE_WIDTH = 400
TEST_IMAGE_WIDTH = 608
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
TRAIN_IMAGE_DIR = "train_data/train_images/"
TRAIN_MASK_DIR = "train_data/train_masks/"
VAL_IMAGE_DIR = "train_data/val_images/"
VAL_MASK_DIR = "train_data/val_masks/"
TEST_IMAGE_DIR = "test_data/"
BATCH_SIZE = 18
NUM_WORKERS = 4
PIN_MEMORY = True
NUM_EPOCHS = 15
PROB_DROPOUT = 0.5
