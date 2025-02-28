import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score
from typing import List

class BinaryAlexNet(nn.Module):
    def __init__(self, dropout: float = 0.5) -> None:
        """
        Initializes the Binary AlexNet model for binary classification.

        Args:
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()

        self.features: nn.Sequential = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.avgpool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Output logits (before applying sigmoid).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_model(self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, epochs: int = 10, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> None:
        """
        Trains the model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            optimizer (optim.Optimizer): Optimizer for updating weights.
            criterion (nn.Module): Loss function.
            epochs (int): Number of epochs for training.
            device (torch.device): Device to run training on.
        """
        self.to(device)
        self.train()
        loss_values: List[float] = []

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            loss_values.append(avg_loss)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        self.plot_loss(loss_values)

    def evaluate_model(self, test_loader: DataLoader, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> float:
        """
        Evaluates the model on test data.

        Args:
            test_loader (DataLoader): DataLoader for test data.
            device (torch.device): Device to run evaluation on.

        Returns:
            float: Accuracy score.
        """
        self.to(device)
        self.eval()
        all_preds: List[float] = []
        all_labels: List[float] = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = self(inputs)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy: float = accuracy_score(all_labels, all_preds)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def plot_loss(self, loss_values: List[float]) -> None:
        """
        Plots the loss curve.

        Args:
            loss_values (List[float]): List of loss values per epoch.
        """
        plt.plot(loss_values, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def save_model(self, path: str) -> None:
        """
        Saves the model state dictionary.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> None:
        """
        Loads the model state dictionary.

        Args:
            path (str): Path to load the model from.
            device (torch.device): Device to load the model onto.
        """
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        self.eval()
        print(f"Model loaded from {path}")


# ---- MAIN SCRIPT ----
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 10

    # Data Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Dummy dataset (replace with real dataset)
    dataset = datasets.FakeData(transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss Function, Optimizer
    model = BinaryAlexNet().to(device)
    criterion = nn.BCEWithLogitsLoss()  # More numerically stable than BCELoss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train and Evaluate
    model.train_model(train_loader, optimizer, criterion, num_epochs)
    model.evaluate_model(test_loader)

    # Save and Load Example
    model.save_model("binary_alexnet.pth")
    model.load_model("binary_alexnet.pth")