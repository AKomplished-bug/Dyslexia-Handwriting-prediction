import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import create_model

class HandwritingDataset(Dataset):
    """
    Custom dataset for loading handwriting data from .npz files.
    """
    def __init__(self, data_path):
        data = np.load(data_path)
        self.images = data['images']
        self.labels = data['labels']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert image to tensor and add a channel dimension
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

def train_model():
    """
    Train the MobileNet V3 model using the preprocessed dataset.
    """
    # Paths to datasets
    train_data_path = os.path.abspath("Data/processed/train_data.npz")
    test_data_path = os.path.abspath("Data/processed/test_data.npz")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 32
    epochs = 10
    learning_rate = 0.001

    # Create datasets and dataloaders
    train_dataset = HandwritingDataset(train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = HandwritingDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the model
    model = create_model(num_classes=3)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # Save the trained model
    model_save_path = os.path.abspath("../Data/processed/mobilenet_v3.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()
