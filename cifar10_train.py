import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Check if GPU is available (cuda for gpu, mps for mac m1, cpu for others)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using {} device".format(device))

# Hyperparameters
learning_rate = 0.0001
epochs = 100
batch_size = 128
num_classes = 10  # 10 classes for CIFAR10

# Define the transform function for trainset
transform_train = transforms.Compose(
    [
        # data augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.TrivialAugmentWide(),
        # data normalization
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Define the transform function for val
transform_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# Load the CIFAR10 dataset
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_train)
valset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_val)

# Define the dataloader
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

# Load pretrained VGG19 model with batch normalization
# model = torchvision.models.vgg19_bn(pretrained=True)
# Change the last layer to fit the CIFAR10 dataset
# model.classifier[6] = nn.Linear(4096, num_classes)

# Load VGG_bn model without pretrained weights (need to train for longer time to get good result)
model = torchvision.models.vgg19_bn(num_classes=num_classes)

# Move the model to GPU for calculation
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Record the loss and accuracy for training and validation set
train_loss, train_acc = [], []
val_loss, val_acc = [], []

# Train the model
print("Training the model...")
for epoch in tqdm(range(epochs)):
    model.train()
    print("\nEpoch: ", epoch + 1)
    batch_train_loss, batch_train_acc = [], []
    for i, data in enumerate(train_loader):
        # Prepare data
        inputs, labels = data
        # Load data to GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record the loss and accuracy for every batch
        batch_train_loss.append(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        batch_train_acc.append((predicted == labels).sum().item() / batch_size)

    # Record the loss and accuracy for every epoch
    loss = np.mean(batch_train_loss)
    acc = np.mean(batch_train_acc)
    train_loss.append(loss)
    train_acc.append(acc)
    print(f"Training Loss: {loss} | Training Accuracy: {acc}")

    # Validate the model
    model.eval()
    with torch.no_grad():
        batch_val_loss, batch_val_acc = [], []
        for i, data in enumerate(val_loader):
            # Prepare data
            inputs, labels = data
            # Load data to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Record the loss and accuracy for every batch
            batch_val_loss.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            batch_val_acc.append((predicted == labels).sum().item() / batch_size)

        # Record the loss and accuracy for every epoch
        loss = np.mean(batch_val_loss)
        acc = np.mean(batch_val_acc)
        val_loss.append(loss)
        val_acc.append(acc)
        print(f"Validation Loss: {loss} | Validation Accuracy: {acc}")

# Plot the loss and accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label="train_acc")
plt.plot(val_acc, label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()

# Save the image
plt.savefig(f"./epoch_{epochs}.png")
# Show the image
plt.show()

# Save the model
torch.save(model.state_dict(), f"./model.pth")
