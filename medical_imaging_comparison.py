import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import medmnist
from medmnist import INFO
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


from ndlinear import NdLinear

# Configuration
DATA_NAME = 'organmnist3d'
BATCH_SIZE = 64
LR = 0.01
EPOCHS = 50  # Keep epochs low for a quick comparison
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Prepare data
info = INFO[DATA_NAME]
NUM_CLASSES = len(info['label'])
N_CHANNELS = info['n_channels']
task = info['task']

DataClass = getattr(medmnist, info['python_class'])

# Preprocessing
# Note: MedMNIST3D datasets are channels-first (C, D, H, W)

data_transform = Compose([
    # Convert numpy array (assumed C, D, H, W) to tensor and normalize to [0, 1]
    lambda x: torch.tensor(x).float() / 255.0,
    # Normalize using mean and std for grayscale images (0.5, 0.5)
    Normalize(mean=[.5], std=[.5])
])

# Use local data instead of downloading
train_dataset = DataClass(split='train', transform=data_transform, download=False, root='.')
val_dataset = DataClass(split='val', transform=data_transform, download=False, root='.')

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#  Model Definitions 

# Simple3DConv is used as a feature extractor for the Baseline and NdLinear models. 


class Simple3DConv(nn.Module):
    """A very simple 3D CNN feature extractor"""
    def __init__(self, out_channels=32):
        super().__init__()
        self.conv1 = nn.Conv3d(N_CHANNELS, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(2) # Output: 14x14x14
        self.conv2 = nn.Conv3d(16, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(2) # Output: 7x7x7
        self.out_channels = out_channels


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        return x


"""

Baseline Model with a simple 3D CNN feature extractor followed by a Flatten and Linear layer. The flattened size is 16 * 7 * 7 * 7 = 5488.


"""
class BaselineModel(nn.Module):
    """Simple CNN followed by Flatten and Linear layers"""
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = Simple3DConv(out_channels=32)
        self.flatten = nn.Flatten()
        # Calculate flattened size: 32 * 7 * 7 * 7
        flattened_size = 32 * 7 * 7 * 7
        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Add dropout layer
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x


"""

NdLinear Model with a simple 3D CNN feature extractor followed by a NdLinear layer.

"""
class NdLinearModel(nn.Module):
    """Simple CNN followed by NdLinear layer"""
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = Simple3DConv(out_channels=32)
        # Feature extractor output: [batch, 32, 7, 7, 7]
        # Adjust hidden_size maybe? Let's keep it for now.
        hidden_size = (16, 4, 4, 4) # Let's increase first dim slightly to match increased channels
        self.ndlinear = NdLinear(input_dims=(32, 7, 7, 7), hidden_size=hidden_size)
        # Output of NdLinear: [batch, 8, 4, 4, 4]
        self.flatten = nn.Flatten()
        flattened_size = np.prod(hidden_size) # 16 * 4 * 4 * 4 = 1024
        self.fc = nn.Linear(flattened_size, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x) # Output: [batch, 32, 7, 7, 7]
        x = self.ndlinear(x) # Output: [batch, 8, 4, 4, 4]
        x = self.flatten(x)
        x = self.fc(x)
        return x

# --- Training and Evaluation ---

"""
Training involves a forward pass, loss calculation, backward pass, and optimizer step.
"""

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.squeeze().long().to(device) # Ensure labels are 1D LongTensor

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

"""

Evaluation takes the same inputs as training, but without the backward pass.

"""
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.squeeze().long().to(device) # Ensure labels are 1D LongTensor
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



# Count the number of parameters in the model.

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Run the experiment ---

models_to_test = {
    "Baseline (nn.Linear)": BaselineModel(NUM_CLASSES),
    "NdLinear": NdLinearModel(NUM_CLASSES)
}

# Empty results dictionary

results = {}

# Define the criterion. Using CrossEntropyLoss since this is a classification task.

criterion = nn.CrossEntropyLoss()

for name, model in models_to_test.items():
    print(f"--- Training {name} ---")
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    param_count = count_parameters(model)
    print(f"Parameters: {param_count:,}")

    start_time = time.time()
    best_val_acc = 0.0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
        best_val_acc = max(best_val_acc, val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

    end_time = time.time()
    results[name] = {
        "best_val_acc": best_val_acc,
        "parameters": param_count,
        "training_time": end_time - start_time
    }
    # plt.figure(figsize=(8,6))
    # plt.plot(range(1, EPOCHS+1), train_loss_history, label='Train Loss')
    # plt.plot(range(1, EPOCHS+1), val_loss_history, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title(f'Loss Curves for {name}')
    # plt.legend()
    # plt.show()
    # print("-" * 20)


# Print the results.

print("##Comparison Summary #")
for name, res in results.items():
    print(f"{name}:")
    print(f"  Best Validation Accuracy: {res['best_val_acc']:.4f}")
    print(f"  Trainable Parameters:   {res['parameters']:,}")
    print(f"  Training Time:          {res['training_time']:.2f}s")
print("-" * 26) 



# Summary of the results.

"""
The number of parameters and the accuracy is influenced by the number of epochs runs. The program was run for 10, 20, 30, 40, and 50 epochs.
It was observed that the validation accuracy of the baseline model didn't improve much with more epochs. The validation accuracy of the NdLinear model improved with more epochs.
For 50 epochs, NdLinear has a validation accuracy of 0.944 while the baseline model has a validation accuracy of 0.0994. The reason for choosing 50 epochs is that for lower epochs,
the models were overfitting; they had high accuracy on the training set but low accuracy on the validation set.

The number of trainable parameters also increased with the number of epochs. For 50 epochs, NdLinear has 26,299 trainable parameters while the baseline model had 1,420,875.
NdLinear takes slightly longer to train (185.16 vs 187.76s) than the baseline model. 

"""