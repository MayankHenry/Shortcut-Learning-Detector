import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import random

# Fix for MNIST download issues
datasets.MNIST.mirrors = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/"
]

# 1. Define the exact same Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 2. The Fix: Completely RANDOM background colors
def colorize_unbiased(image):
    image = image.repeat(3, 1, 1) # Make it RGB
    
    # Generate a random RGB color
    color = torch.rand(3)
    
    # Apply the random color to the background
    mask = image[0] < 0.5
    for i in range(3):
        image[i][mask] = color[i]
        
    return image

# 3. Custom Dataset
class UnbiasedMNIST(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset):
        self.mnist = mnist_dataset

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        colored_image = colorize_unbiased(image)
        return colored_image, label

def train_model():
    print("Loading MNIST and applying RANDOM colors...")
    transform = transforms.ToTensor()
    raw_train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    unbiased_train_data = UnbiasedMNIST(raw_train_data)
    
    train_loader = torch.utils.data.DataLoader(unbiased_train_data, batch_size=64, shuffle=True)
    
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training the Fixed Model (forced to learn shapes!)...")
    model.train()
    # Training for 2 epochs
    for epoch in range(2): 
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Training complete! Saving unbiased model...")
    torch.save(model.state_dict(), "unbiased_mnist_model.pth")
    print("Saved to 'unbiased_mnist_model.pth'")

if __name__ == "__main__":
    train_model()