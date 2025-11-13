import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dunet import DUNet
from att_dunet import Att_DUNet
from dataloader import CustomDataset
from tqdm import tqdm
import os
import torch.optim.lr_scheduler as lr_scheduler

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def train(model, train_dataloader, val_dataloader, epochs, lr, device="cuda"):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    stepLR = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    model.to(device)
    
    epoch_losses = []
    val_losses = []

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True,
                  ascii=True) as pbar:
            for mat_batch, img_batch in train_dataloader:
                mat_batch, img_batch = mat_batch.to(device), img_batch.to(device)

                optimizer.zero_grad()
                outputs = model(mat_batch)

                loss = criterion(outputs, img_batch)
                running_loss += loss.item()

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=running_loss / (pbar.n + 1))
                pbar.update(1)

        epoch_loss = running_loss / len(train_dataloader)
        epoch_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mat_batch, img_batch in val_dataloader:
                mat_batch, img_batch = mat_batch.to(device), img_batch.to(device)

                outputs = model(mat_batch)
                loss = criterion(outputs, img_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_attdunet_model.pth")

        stepLR.step()

    torch.save(model.state_dict(), "attdunet_model_last_epoch.pth")
    print("Model saved as 'attdunet_model_last_epoch.pth'.")

    plt.figure()
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='b', label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, marker='x', color='r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Progression')
    plt.legend()
    plt.grid(True)
    plt.savefig('attdunet_training_loss.png')
    plt.show()


if __name__ == "__main__":
    train_pa_data_dir = "pa_data"
    train_ground_true_dir = "ground_truth"
    val_pa_data_dir = "validation_pa_data"
    val_ground_true_dir = "validation_ground_truth"

    train_dataset = CustomDataset(train_pa_data_dir, train_ground_true_dir)
    val_dataset = CustomDataset(val_pa_data_dir, val_ground_true_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1)

    model = Att_DUNet()
    model.load_state_dict(torch.load('1.pth'))

    train(model, train_dataloader, val_dataloader, epochs=300, lr=0.01, device="cuda")
