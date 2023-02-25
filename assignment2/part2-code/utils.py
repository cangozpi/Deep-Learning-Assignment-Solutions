import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_model(model, train_dataloader, val_dataloader, epochs, optimizer, loss_func):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    train_loss_hist = []
    train_accuracy_hist = []
    val_loss_hist = []
    val_accuracy_hist = []
    for epoch in range(1, epochs+1):
        # Train Model
        model.train()
        train_epoch_losses_hist = []
        train_epoch_accuracy_hist = []
        with tqdm(train_dataloader, unit="batch") as tEpoch:
            for data, target in tEpoch:
                tEpoch.set_description(f"Training: Epoch {epoch}/{epochs}")
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                preds = model(data) # [B, C]
                target_ids = target.argmax(dim=1, keepdim=True).squeeze() # [B,]
                loss = loss_func(preds, target_ids.type(torch.long))
                loss.backward()
                optimizer.step()

                # Track metrics
                prediction_ids = preds.argmax(dim=1, keepdim=True).squeeze() # [B,]
                correct = (prediction_ids == target_ids).sum().item()
                accuracy = correct / data.shape[0]
                # Record metrics
                train_epoch_losses_hist.append(loss.item() * data.shape[0])
                train_epoch_accuracy_hist.append(accuracy * data.shape[0])
                tEpoch.set_postfix(train_loss=loss.item(), train_accuracy=accuracy) # metrics for current epoch, NOT running averages
        
        avg_epoch_loss = sum(train_epoch_losses_hist) / len(train_dataloader.dataset)
        avg_epoch_accuracy = sum(train_epoch_accuracy_hist) / len(train_dataloader.dataset)
        train_loss_hist.append(avg_epoch_loss)
        train_accuracy_hist.append(avg_epoch_accuracy)
    


        # Validate Model
        model.eval()
        val_epoch_losses_hist = []
        val_epoch_accuracy_hist = []
        with tqdm(val_dataloader, unit="batch") as tEpoch:
            for data, target in tEpoch:
                tEpoch.set_description(f"Validating: Epoch {epoch}/{epochs}")
                data, target = data.to(device), target.to(device)

                with torch.no_grad():
                    preds = model(data) # [B, C]
                    target_ids = target.argmax(dim=1, keepdim=True).squeeze() # [B,]
                    loss = loss_func(preds, target_ids.type(torch.long))

                # Track metrics
                prediction_ids = preds.argmax(dim=1, keepdim=True).squeeze() # [B,]
                correct = (prediction_ids == target_ids).sum().item()
                accuracy = correct / data.shape[0]
                # Record metrics
                val_epoch_losses_hist.append(loss.item() * data.shape[0])
                val_epoch_accuracy_hist.append(accuracy * data.shape[0])
                tEpoch.set_postfix(val_loss=loss.item(), val_accuracy=accuracy) # metrics for current epoch, NOT running averages
        
        avg_epoch_loss = sum(val_epoch_losses_hist) / len(val_dataloader.dataset)
        avg_epoch_accuracy = sum(val_epoch_accuracy_hist) / len(val_dataloader.dataset)
        val_loss_hist.append(avg_epoch_loss)
        val_accuracy_hist.append(avg_epoch_accuracy)


    return train_loss_hist, train_accuracy_hist, val_loss_hist, val_accuracy_hist


def test_model(model, test_dataloader, loss_func):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    test_epoch_losses_hist = []
    test_epoch_accuracy_hist = []
    with tqdm(test_dataloader, unit="batch") as tEpoch:
        for data, target in tEpoch:
            tEpoch.set_description(f"Testing:")
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                preds = model(data) # [B, C]
                target_ids = target.argmax(dim=1, keepdim=True).squeeze() # [B,]
                loss = loss_func(preds, target_ids.type(torch.long))

            # Track metrics
            prediction_ids = preds.argmax(dim=1, keepdim=True).squeeze() # [B,]
            correct = (prediction_ids == target_ids).sum().item()
            accuracy = correct / data.shape[0]
            # Record metrics
            test_epoch_losses_hist.append(loss.item() * data.shape[0])
            test_epoch_accuracy_hist.append(accuracy * data.shape[0])
            tEpoch.set_postfix(test_loss=loss.item(), test_accuracy=accuracy) # metrics for current epoch, NOT running averages
    
    avg_epoch_loss = sum(test_epoch_losses_hist) / len(test_dataloader.dataset)
    avg_epoch_accuracy = sum(test_epoch_accuracy_hist) / len(test_dataloader.dataset)
    
    return avg_epoch_loss, avg_epoch_accuracy
    

def plot_results(train_loss_hist, train_accuracy_hist, val_loss_hist, val_accuracy_hist):
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(train_loss_hist)), train_loss_hist, label="training")
    plt.plot(np.arange(len(val_loss_hist)), val_loss_hist, label="validation")
    plt.title("Loss History")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(train_accuracy_hist)),train_accuracy_hist, label="training")
    plt.plot(np.arange(len(val_accuracy_hist)), val_accuracy_hist, label="validation")
    plt.title("Accuracy History")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()

    plt.show()
