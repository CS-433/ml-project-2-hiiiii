import torch
import os
import sys

from model import UNET
import constants as cst
from utils import *
from transforms_v2 import *
from train import *

def main():
    # read arguments
    print("Reading arguments...")
    args = sys.argv
    load_model = False
    model_path = ""
    if len(args) == 2 or len(args) > 3:
        raise ValueError("Too many arguments")
    elif len(args) == 3:
        if args[1] == "load":
            load_model = True
        else:
            raise ValueError("Invalid argument")
        model_path = args[2]
    # create model
    print("Creating model...")
    model = UNET(in_channels=3, out_channels=1).to(cst.DEVICE)
    if load_model:
        load_checkpoint(torch.load(model_path), model)
        predict_test_images(model)
        exit()
    # define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cst.LEARNING_RATE)
    # load data
    print("Loading data...")
    train_loader, val_loader = get_loaders(
        train_image_dir=cst.TRAIN_IMAGE_DIR,
        train_mask_dir=cst.TRAIN_MASK_DIR,
        val_image_dir=cst.VAL_IMAGE_DIR,
        val_mask_dir=cst.VAL_MASK_DIR,
        batch_size=cst.BATCH_SIZE,
        train_transform=get_transforms(),
        val_transform=[val_transforms],
        num_workers=cst.NUM_WORKERS,
        pin_memory=cst.PIN_MEMORY,
    )
    # check that cuda is available
    print("Cuda is available: ", torch.cuda.is_available())
    # train model
    print("Training model...")
    train_loss_history, train_f1_history, val_loss_history, val_f1_history = train(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader
    )
    # save losses and f1 scores
    print("Saving losses and f1 scores...")
    train_f1_history = [f.cpu() for f in train_f1_history]
    val_f1_history = [f.cpu() for f in val_f1_history]
    save_array_data(train_loss_history, "train_loss_history.dat")
    save_array_data(val_loss_history, "val_loss_history.dat")
    save_array_data(train_f1_history, "train_f1_history.dat")
    save_array_data(val_f1_history, "val_f1_history.dat")
    # load best model
    print("Loading best model...")
    load_checkpoint(torch.load("checkpoints/my_checkpoint.pth.tar"), model)
    # predict on test data
    print("Predicting on test data...")
    predict_test_images(model)
    # run mask_to_submission.py
    print("Running mask_to_submission.py...")
    os.system("python3 mask_to_submission.py")

if __name__ == "__main__":
    main()
