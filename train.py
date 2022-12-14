from tqdm import tqdm
from utils import *
import constants as cst

def train_epoch(model, optimizer, criterion, train_loader, epoch, device):
    print(f"Epoch {epoch+1}/{cst.NUM_EPOCHS}")
    model.train()
    train_loss = 0
    train_f1 = 0
    for _, (data, target) in enumerate(tqdm(train_loader)):
        # move data to device
        data, target = data.to(device), target.unsqueeze(1).to(device)
        # predict
        predictions = model(data)
        # compute loss
        loss = criterion(predictions, target)
        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
        # compute loss and f1 score
        train_loss += loss.item()
        train_f1 += f1_score(predictions, target)
    # compute average loss and f1 score
    train_loss /= len(train_loader)
    train_f1 /= len(train_loader)
    print('\nTrain set: Average loss: {:.4f}\tAverage F1: {:.4f}'.format(train_loss, train_f1))
    return train_loss, train_f1

@torch.no_grad()
def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    val_f1 = 0
    for _, (data, target) in enumerate(val_loader):
        # move data to device
        data, target = data.to(device), target.unsqueeze(1).to(device)
        # predict
        predictions = model(data)
        # compute loss and f1 score
        val_loss += criterion(predictions, target).item()
        val_f1 += f1_score(predictions, target)
    # compute average loss and f1 score
    val_loss /= len(val_loader)
    val_f1 /= len(val_loader)
    print('Validation set: Average loss: {:.4f}\tAverage F1: {:.4f}'.format(val_loss, val_f1))
    model.train()
    return val_loss, val_f1

def train(model, optimizer, criterion, train_loader, val_loader):
    train_loss_history = []
    train_f1_history = []
    val_loss_history = []
    val_f1_history = []

    for epoch in range(cst.NUM_EPOCHS):

        train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_loader, epoch, cst.DEVICE)

        train_loss_history.append(train_loss)
        train_f1_history.append(train_f1)

        val_loss, val_f1 = validate(model, criterion, val_loader, cst.DEVICE)

        val_loss_history.append(val_loss)
        val_f1_history.append(val_f1)

        # save model if the validation loss is the lowest so far
        if val_loss == min(val_loss_history):
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=cst.DEVICE
            )
    
    return train_loss_history, train_f1_history, val_loss_history, val_f1_history
