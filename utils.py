import torch
import torchvision
import os
import cv2
import numpy as np
from dataset_v2 import RoadDataset
from torch.utils.data import DataLoader
import constants as cst
from transforms_v2 import *

def save_checkpoint(state, filename="checkpoints/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
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

@torch.no_grad()
def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        predictions = model(x)
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5).float()
        predictions = predictions.cpu().numpy()
        torchvision.utils.save_image(
            torch.tensor(predictions), f"{folder}pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

def f1_score(predictions, targets):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    tp = (predictions * targets).sum()
    fp = ((1 - targets) * predictions).sum()
    fn = (targets * (1 - predictions)).sum()
    f1 = (2 * tp) / (2 * tp + fp + fn)
    return f1

def split_test_image(image):
    """Split a 608x608 image into 4 304x304 images"""
    image_parts = []
    image_parts.append(image[:304, :304])
    image_parts.append(image[:304, 304:])
    image_parts.append(image[304:, :304])
    image_parts.append(image[304:, 304:])
    return image_parts

def predict_image(image, image_folder, model, device):
    model.eval()
    image = test_transforms(image=image)["image"]
    image = image.unsqueeze(0).to(device)
    prediction = model(image)
    prediction = torch.sigmoid(prediction)
    prediction = (prediction > 0.5).float()
    prediction = prediction.cpu().numpy()
    torchvision.utils.save_image(
        torch.tensor(prediction), cst.TEST_IMAGE_DIR + image_folder + "/" + image_folder + "_pred.png"
    )

def predict_test_images(model):
    for image_folder in os.listdir(cst.TEST_IMAGE_DIR):
        img = cv2.imread(cst.TEST_IMAGE_DIR + image_folder + "/" + image_folder + ".png")
        img = np.array(img)
        predict_image(img, image_folder, model, cst.DEVICE)

def save_array_data(data, filename):
    np.savetxt("data_history/" + filename + ".dat", data, delimiter="\n")
