import torch
import os
import sys

import constants as cst
from utils import *
from transforms_v2 import *
from train import *
from prediction import *
from loss import *

################################################################################
# ENTRY POINT OF THE PROGRAM
################################################################################

def main():
    torch.cuda.empty_cache()
    # read arguments
    print("Reading arguments...")
    args = sys.argv
    load_model, model_path, predict = read_args(args)
    print("Done")
    # create model
    print("Creating model...")
    model = cst.MODEL(in_channels=3, out_channels=1).to(cst.DEVICE)
    if load_model:
        # load model
        load_checkpoint(torch.load(model_path), model)
    if predict:
        # predict on test data
        print("Predicting on test data...")
        predict_test_images(model)
        print("Done")
        # run mask_to_submission.py
        print("Running mask_to_submission.py...")
        os.system("python3 mask_to_submission.py")
        print("Done")
        return
    print("Done")
    # load data
    print("Loading data...")
    train_loader, val_loader = get_loaders(
        train_image_dir=cst.TRAIN_IMAGE_DIR,
        train_mask_dir=cst.TRAIN_MASK_DIR,
        val_image_dir=cst.VAL_IMAGE_DIR,
        val_mask_dir=cst.VAL_MASK_DIR,
        batch_size=cst.BATCH_SIZE,
        train_transform=get_train_transforms(),
        val_transform=get_test_transforms(),
        num_workers=cst.NUM_WORKERS,
        pin_memory=cst.PIN_MEMORY,
    )
    # define loss function, optimizer and schedulers
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cst.LEARNING_RATE, weight_decay=cst.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(cst.NUM_EPOCHS * len(train_loader.dataset)) // cst.BATCH_SIZE,
    )
    print("Done")
    # check that cuda is available
    if not torch.cuda.is_available():
        print("CUDA is not available")
        raise ValueError("CUDA is not available, please use a GPU")
    # train model
    print("Training model...")
    train_loss_history, train_f1_history, val_loss_history, val_f1_history = train(
        model,
        optimizer,
        criterion,
        scheduler,
        train_loader,
        val_loader
    )
    print("Done")
    # save losses and f1 scores
    print("Saving losses and f1 scores...")
    train_f1_history = [f.cpu() for f in train_f1_history]
    val_f1_history = [f.cpu() for f in val_f1_history]
    save_array_data(train_loss_history, "train_loss_history.dat")
    save_array_data(val_loss_history, "val_loss_history.dat")
    save_array_data(train_f1_history, "train_f1_history.dat")
    save_array_data(val_f1_history, "val_f1_history.dat")
    print("Done")
    # load best model
    print("Loading best model...")
    load_checkpoint(torch.load("checkpoints/max_val_f1.pth.tar"), model)
    print("Done")
    # predict on test data
    print("Predicting on test data...")
    predict_test_images(model)
    print("Done")
    # run mask_to_submission.py
    print("Running mask_to_submission.py...")
    os.system("python3 mask_to_submission.py")
    print("Done")

if __name__ == "__main__":
    main()
