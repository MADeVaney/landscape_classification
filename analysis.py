import argparse  # Command Line args
import torch  # For Math, Cuda access, Load
from tqdm import tqdm  # Progress Bar
from torch import nn  # Model Layers, Class
from torch import optim  # SGD
from torch.utils.data import DataLoader, random_split  # DataLoader, random_split func.
from torchvision import models, datasets, transforms  # Resnet18 weights

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


classes = ['Agriculture', 'Airport', 'Beach', 'City', 'Desert', 'Forest', 'Grassland', 'Highway', 'Lake', 'Mountain', 'Parking', 'Port', 'Railway', 'Residential', 'River']




model = SimpleNet(num_classes=15)
model.load_state_dict(torch.load("torchmodel.pth"))

data_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Load image dataset from folder
dataset = datasets.ImageFolder(root="../Aerial_Landscapes", transform=data_transform)

# Split dataset into train and validation sets
val_size = int(len(dataset) * 0.2)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

val_loader = DataLoader(val_dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()


results = {}

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

    vl = int(val_labels[0])
    vp = int(val_preds[0])

    if vl in results.keys():
        if vp in results[vl].keys():
            results[vl][vp] += 1
        else:
            results[vl][vp] = 1
    else:
        results[vl] = {vp: 1}

val_epoch_loss = val_running_loss / len(val_loader.dataset)
val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)

print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")


for k1 in results.keys():
    '''for k2 in results[k1].keys():
        print("Actual was " + classes[k1] + " and " + classes[k2] + " was predicted " + str(round((results[k1][k2] / sum(results[k1].values())) * 100, 2)) + " percent of the time")'''
    print(classes[k1] + ": " + str(round((results[k1][k1] / sum(results[k1].values())) * 100, 2)) + "%")

print(f"Total test set accuracy was {val_epoch_acc*100:.4f}%")
print("Accuracy without Airport data was " + str(round(float((val_running_corrects.double() - results[1][1]) / (len(val_loader.dataset) - sum(results[1].values()))), 4) * 100) + "%")