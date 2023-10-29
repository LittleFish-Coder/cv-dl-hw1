import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Check if GPU is available (cuda for gpu, mps for mac m1, cpu for others)
device = "cude" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Hyperparameters
learning_rate = 0.001
epochs = 40
batch_size = 128
num_classes = 10  # 10 classes for CIFAR10

# Define the transform function for trainset
transform_train = transforms.Compose(
    [
        # data augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        # data normalization
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
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
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
valset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_val)

# Define the dataloader
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

# Load pretrained VGG19 model with batch normalization, and change the last layer to 10 classes
model = torchvision.models.vgg19_bn(pretrained=True, num_classes=num_classes)

# Move the model to GPU for calculation
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, momentum=0.9)

# Record the loss and accuracy for training and validation set
train_loss, train_acc = [], []
val_loss, val_acc = [], []

# Train the model
print("Training the model...")
for epoch in range(epochs):
    model.train()
    print("\nEpoch: ", epoch + 1)
    batch_train_loss, batch_train_acc = [], []
    for i, data in enumerate(train_loader, 0):
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
        for i, data in enumerate(val_loader, 0):
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
plt.plot(train_loss, label="Train")
plt.plot(val_loss, label="Validation")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label="Train")
plt.plot(val_acc, label="Validation")
plt.title("Accuracy")
plt.legend()

plt.show()

# Save the model
torch.save(model.state_dict(), "./model.pth")
