from dataset import RoadDataset
from torch.utils.data import DataLoader

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