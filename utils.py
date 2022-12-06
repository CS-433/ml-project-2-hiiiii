import torch
import torchvision
from dataset import RoadDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_images_dir,
    train_groundtruth_dir,
    val_images_dir,
    val_groundtruth_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = RoadDataset(train_images_dir, train_groundtruth_dir, train_transform)
    val_ds = RoadDataset(val_images_dir, val_groundtruth_dir, val_transform)

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
        predictions = (predictions > 0.5).float()
        predictions = predictions.cpu().numpy()
        predictions[predictions > 0.5] = 255
        torchvision.utils.save_image(
            torch.tensor(predictions), f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
