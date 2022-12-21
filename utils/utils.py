import torch
import numpy as np
from data.dataset import RoadDataset
from torch.utils.data import DataLoader
from data.transforms import *

################################################################################
# UTILS
################################################################################

def read_args(args):
    load_model = False
    model_path = ""
    predict = False
    if len(args) == 2 or len(args) > 4:
        raise ValueError("Invalid number of arguments")
    elif len(args) == 3 or len(args) == 4:
        if args[1] == "load":
            load_model = True
        else:
            raise ValueError("Invalid argument")
        model_path = args[2]
    if len(args) == 4:
        if args[3] == "predict":
            predict = True
        else:
            raise ValueError("Invalid argument")
    return load_model, model_path, predict

def save_checkpoint(state, filename="checkpoints/my_checkpoint.pth.tar"):
    '''Save the state of the model'''
    print("=> Saving checkpoint\n")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    '''Load the state of the model'''
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_image_dir,
    train_mask_dir,
    val_image_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    '''Return train and validation dataloaders'''
    train_ds = RoadDataset(train_image_dir, train_mask_dir, train_transform)
    val_ds = RoadDataset(val_image_dir, val_mask_dir, val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def f1_score(predictions, targets):
    '''Calculate the f1 score'''
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    tp = (predictions * targets).sum()
    fp = ((1 - targets) * predictions).sum()
    fn = (targets * (1 - predictions)).sum()
    f1 = (2 * tp) / (2 * tp + fp + fn)
    return f1

def save_array_data(data, filename):
    '''Save the data of an array'''
    np.savetxt("data_history/" + filename, data, delimiter="\n")
