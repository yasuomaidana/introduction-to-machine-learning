import torch
from torch import Generator
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from model.u_net import UNet
from data.caravana_dataset import CaravanDataset


def train_model(data_path: str = "./data/raw",
                directory_name="train", split_ratio=0.8,
                learning_rate=3e-4, batch_size=16, num_epochs=2) -> UNet:

    dataset = CaravanDataset(data_path, directory_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    generator = Generator().manual_seed(42)

    train_dataset, valid_dataset = random_split(dataset, [split_ratio, 1-split_ratio], generator=generator)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(3, 1).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = BCEWithLogitsLoss()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_running_loss = 0
        for images, masks in tqdm(train_data_loader):

            images = images.float().to(device)
            masks = masks.float().to(device)

            y_predicted = model(images)
            optimizer.zero_grad()

            loss = criterion(y_predicted, masks)
            train_running_loss += loss.item()

            loss.backward()
            optimizer.step()

        average_train_loss = train_running_loss/len(train_data_loader)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for images, masks in tqdm(valid_data_loader):
                images, masks = images.to(device), masks.to(device)

                y_predicted = model(images)
                loss = criterion(y_predicted, masks)
                val_running_loss += loss.item()
            average_val_loss = val_running_loss / len(valid_data_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {average_train_loss:.4f} - Validation Loss: {average_val_loss:.4f}")

    return model


if __name__ == "__main__":
    trained_model = train_model()
    model_save_path = "./easy_unet_model.pth"
    torch.save(trained_model.state_dict(), model_save_path)
