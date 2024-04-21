# Train Simple Neural Network
# Utilizes transfer learning; 
# weights taken from Resnet 18. 

# NOTE -- THIS IS A COMMAND LINE TOOL. 
# YOU MUST PROVIDE THE PATH TO YOUR SKYVIEW
# DATA IN THE COMMAND LINE INPUT FOR IT TO RUN. 

# TODO -- There are several possible optimizations:
# - Experiment with optimizers -- currently uses SGD
# - Experiment with transformations -- currently resizes for Resnet input
# - Experiment with learning rate -- defaults to .001 (LR decay?)
# - Increase epochs -- defaults to 5
# - Experiment with Batch -- defaults to 32
# - MANY OTHERS!

import argparse  # Command Line args
import torch  # For Math, Cuda access, Load
from tqdm import tqdm  # Progress Bar
from torch import nn  # Model Layers, Class
from torch import optim  # SGD
from torch.utils.data import DataLoader, random_split  # DataLoader, random_split func.
from torchvision import models, datasets, transforms  # Resnet18 weights


# Define a simple neural network model
class SimpleNet(nn.Module):
    """
    Initialize the SimpleNet model.

    Args:
        num_classes (int): Number of output classes.
    """
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.features = models.resnet18(weights=None)
        self.features.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.features(x)
        return x


def train_model(model, train_loader, val_loader, num_epochs=5, learning_rate=0.001):
    """
    Train the specified model.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of training epochs. Default is 5.
        learning_rate (float): Learning rate for optimizer. Default is 0.001.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        with tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
        ) as train_bar:
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                train_bar.set_postfix(loss=loss.item())
                train_bar.update()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        for val_inputs, val_labels in val_loader:
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)

            val_outputs = model(val_inputs)
            _, val_preds = torch.max(val_outputs, 1)
            val_loss = criterion(val_outputs, val_labels)

            val_running_loss += val_loss.item() * val_inputs.size(0)
            val_running_corrects += torch.sum(val_preds == val_labels.data)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)

        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a neural network on the Skyview dataset"
    )
    parser.add_argument("input_file", type=str, help="Path to Skyview dataset")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=7, help="Number of epochs (default: 7)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=15,
        help="Number of data classes (default: 15)",
    )
    args = parser.parse_args()

    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Load image dataset from folder
    dataset = datasets.ImageFolder(root=args.input_file, transform=data_transform)

    # Split dataset into train and validation sets
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print("Total train size:", len(train_loader.dataset))
    print("Total val size:", len(val_loader.dataset))

    # Define the model
    model = SimpleNet(num_classes=args.num_classes)

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=args.num_epochs)
